import asyncio
import gc
import threading
import time
import uuid
from collections import Counter
from pathlib import Path
from urllib.parse import unquote, urlparse
from urllib.request import Request, urlopen

import cv2
import numpy as np
import openmmlab_service.core.events as events_module
from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect

from openmmlab_service.api.schemas import (
    ImageRequest,
    StopRequest,
    StreamRequest,
    TaskQueryRequest,
    TaskStatusRequest,
    UpdateClassesRequest,
)
from openmmlab_service.config import (
    INTERNAL_RTMP_HOST,
    OUTPUT_FPS,
    PUBLIC_HOST,
    PUBLIC_WS_BASE,
    PUBLIC_ZLM_BASE,
    RTMP_PORT,
    RTSP_PORT,
    get_label_display_name,
    resolve_path,
)
from openmmlab_service.core.draw import cv2_draw_chinese_batch
from openmmlab_service.core.engine import push_stream_worker
from openmmlab_service.core.events import active_websockets
from openmmlab_service.core.pipeline import execute_pipeline, normalize_pipeline
from openmmlab_service.core.pool import model_pool
from openmmlab_service.core.state import active_tasks
from openmmlab_service.core.storage import persist_image_array

router = APIRouter()


def parse_names_dict(names_dict: str) -> dict:
    """解析 'drone_people:人群,drone_car:汽车' 格式的类目翻译字典字符串"""
    result = {}
    for raw_item in (names_dict or "").split(","):
        item = raw_item.strip()
        if not item:
            continue
        if ":" in item or "：" in item:
            parts = item.replace("：", ":").split(":", 1)
            label = parts[0].strip()
            if label:
                display_name = parts[1].strip()
                result[label] = get_label_display_name(label) if not display_name or display_name == label else display_name
        else:
            result[item] = get_label_display_name(item)
    return result


def build_task_urls(task_id: str) -> dict:
    return {
        "rtmp_url": f"rtmp://{PUBLIC_HOST}:{RTMP_PORT}/live/{task_id}",
        "webrtc_url": f"{PUBLIC_ZLM_BASE}/index/api/webrtc?app=live&stream={task_id}&type=play",
        "flv_url": f"{PUBLIC_ZLM_BASE}/live/{task_id}.live.flv",
        "m3u8_url": f"{PUBLIC_ZLM_BASE}/live/{task_id}/hls.m3u8",
        "rtsp_url": f"rtsp://{PUBLIC_HOST}:{RTSP_PORT}/live/{task_id}",
        "ws_url": f"{PUBLIC_WS_BASE}/ws/{task_id}",
    }


def build_task_summary(task_id: str, info: dict) -> dict:
    data = {
        "task_id": task_id,
        "type": info.get("type", ""),
        "status": info.get("status", ""),
        "source": info.get("source", ""),
        "alias": info.get("alias", ""),
        "mode": info.get("mode", ""),
        "device": info.get("device", ""),
        "rtmp_url": info.get("rtmp_url", ""),
        "target_classes": info.get("target_classes", {}),
        "selected_models": info.get("selected_models", []),
        "pipeline": info.get("pipeline", []),
        "conf_thres": info.get("conf_thres", 0.25),
        "source_fps": info.get("source_fps", 0.0),
        "current_fps": info.get("current_fps", 0.0),
        "output_fps": info.get("output_fps", 0.0),
        "frame_count": info.get("frame_count", 0),
        "queue_wait_ms": info.get("queue_wait_ms", 0.0),
        "model_exec_ms": info.get("model_exec_ms", 0.0),
        "draw_ms": info.get("draw_ms", 0.0),
        "write_ms": info.get("write_ms", 0.0),
    }
    if task_id and data["type"] == "stream":
        data.update(build_task_urls(task_id))
    return data


def _read_remote_bytes(source: str) -> bytes:
    request = Request(source, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(request, timeout=15) as response:
        return response.read()


def _read_local_bytes(source: str) -> bytes:
    parsed = urlparse(source)
    if parsed.scheme == "file":
        path = Path(unquote(parsed.path))
    else:
        path = Path(resolve_path(source))

    if not path.exists():
        raise FileNotFoundError(f"找不到图片文件: {source}")
    return path.read_bytes()


def load_image_from_source(source: str) -> np.ndarray:
    parsed = urlparse(source)
    if parsed.scheme in {"http", "https"}:
        raw_bytes = _read_remote_bytes(source)
    else:
        raw_bytes = _read_local_bytes(source)

    image = cv2.imdecode(np.frombuffer(raw_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("图片解码失败，请检查文件格式是否正确")
    return image


@router.get("/models")
async def list_models():
    models = model_pool.describe_models()
    labels = []
    for item in models:
        for label in item["labels"]:
            labels.append(
                {
                    "id": label,
                    "name": get_label_display_name(label),
                    "model_id": item["model_id"],
                }
            )

    labels.sort(key=lambda item: (item["model_id"], item["id"]))
    return {
        "code": 200,
        "message": "成功获取模型目录",
        "data": {
            "models": models,
            "labels": labels,
        },
    }


@router.post("/detect-stream")
async def detect_stream(req: StreamRequest):
    stream_url = req.stream_url or req.m3u8_url or req.url
    if not stream_url:
        raise HTTPException(status_code=400, detail="未提供有效流地址(需要 stream_url, m3u8_url 或 url 其中一个)")

    model_obj = model_pool.acquire(timeout=3.0)
    if not model_obj:
        raise HTTPException(status_code=429, detail="服务器并发达到满载，当前停车场没有多余的模型算力车位，请稍后再试或杀掉现存任务")

    target_classes = parse_names_dict(req.names_dict)
    try:
        pipeline_info = normalize_pipeline(model_pool, target_classes, req.pipeline, req.conf_thres)
    except ValueError as exc:
        model_pool.release(model_obj)
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    task_id = f"{req.alias}-{uuid.uuid4()}"
    urls = build_task_urls(task_id)
    internal_push_rtmp = f"rtmp://{INTERNAL_RTMP_HOST}:{RTMP_PORT}/live/{task_id}"

    active_tasks[task_id] = {
        "type": "stream",
        "status": "starting",
        "source": stream_url,
        "rtmp_url": urls["rtmp_url"],
        "device": model_obj["device"],
        "alias": req.alias,
        "confidence_display": bool(req.confidence_display),
        "conf_thres": req.conf_thres,
        "target_classes": pipeline_info["target_classes"],
        "selected_models": pipeline_info["selected_models"],
        "pipeline": pipeline_info["pipeline"],
        "mode": req.mode,
        "current_fps": 0.0,
        "output_fps": 0.0,
        "frame_count": 0,
        "queue_wait_ms": 0.0,
        "model_exec_ms": 0.0,
        "draw_ms": 0.0,
        "write_ms": 0.0,
    }

    thread = threading.Thread(
        target=push_stream_worker,
        args=(
            task_id,
            stream_url,
            internal_push_rtmp,
            OUTPUT_FPS,
            model_obj,
            pipeline_info["target_classes"],
            req.conf_thres,
            req.mode,
            pipeline_info["pipeline"],
            bool(req.confidence_display),
        ),
        daemon=True,
        name=f"stream-{task_id}",
    )
    active_tasks.update_fields(task_id, thread=thread)
    thread.start()

    return {
        "code": 200,
        "message": "直播流处理任务已启动",
        "data": {
            "status": "starting",
            "device": model_obj["device"],
            "source_fps": 0.0,
            "current_fps": 0.0,
            "output_fps": 0.0,
            "frame_count": 0,
            "queue_wait_ms": 0.0,
            "model_exec_ms": 0.0,
            "draw_ms": 0.0,
            "write_ms": 0.0,
            "task_id": task_id,
            "target_classes": pipeline_info["target_classes"],
            "selected_models": pipeline_info["selected_models"],
            "pipeline": pipeline_info["pipeline"],
            **urls,
        },
    }


@router.post("/detect-image")
async def detect_image(req: ImageRequest):
    img_url = req.image_url or req.file_url or req.url
    if not img_url:
        raise HTTPException(status_code=400, detail="缺少有效图片地址参数")

    target_classes = parse_names_dict(req.names_dict)
    try:
        pipeline_info = normalize_pipeline(model_pool, target_classes, req.pipeline, req.conf_thres)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    try:
        image = load_image_from_source(img_url)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"图片读取失败: {exc}") from exc

    model_obj = model_pool.acquire(timeout=3.0)
    if not model_obj:
        raise HTTPException(status_code=429, detail="服务器并发达到满载，请稍后再试")

    try:
        service = model_obj["service"]
        pipeline_result = execute_pipeline(service, image.copy(), pipeline_info["pipeline"], 15.0)
        boxes = pipeline_result["boxes"]
        rendered, _ = cv2_draw_chinese_batch(image.copy(), boxes, pipeline_info["target_classes"])

        input_saved = persist_image_array(image, "inputs", "source")
        output_saved = persist_image_array(rendered, "processed", "detect")

        counts = Counter(label_en for _, label_en, _ in boxes)
        statistics = [
            {
                "label": label,
                "label_name": pipeline_info["target_classes"].get(label, label),
                "count": count,
            }
            for label, count in counts.most_common()
        ]

        return {
            "code": 200,
            "message": "图像处理成功",
            "data": {
                "source_image_url": input_saved["url"],
                "image_url": output_saved["url"],
                "statistics": statistics,
                "detection_count": len(boxes),
                "selected_models": pipeline_info["selected_models"],
                "pipeline": pipeline_info["pipeline"],
            },
        }
    finally:
        model_pool.release(model_obj)


@router.post("/ai/stop")
async def stop_tasks(req: StopRequest):
    target = req.task_id
    results = []

    def kill_task(tid):
        task = active_tasks.get(tid)
        if not task:
            return
        active_tasks.update_fields(tid, status="stopping")
        if task.get("process"):
            try:
                task["process"].terminate()
            except Exception:
                pass
        results.append({"task_id": tid, "code": 200, "message": f"任务 {tid} 停止"})

    if target == "all":
        for tid in active_tasks.keys():
            kill_task(tid)
    elif "," in target:
        for tid in target.split(","):
            kill_task(tid.strip())
    else:
        killed_any = False
        for tid, info in active_tasks.items():
            if tid == target or info.get("alias") == target:
                kill_task(tid)
                killed_any = True
        if not killed_any:
            results.append({"task_id": target, "code": 404, "message": "未找到任务"})

    gc.collect()
    return {
        "code": 200,
        "message": "操作完成",
        "results": results,
        "stopped_count": len([r for r in results if r["code"] == 200]),
        "failed_count": len([r for r in results if r["code"] != 200]),
    }


@router.post("/ai/reload-models")
async def reload_models():
    """模型热插拔热加载接口：重新装载权重，实现零宕机更新"""
    model_pool.reload()
    return {"code": 200, "message": "模型热更新重载挂载完毕，后续推流将自动无缝使用新模型权重。"}


@router.post("/tasks")
async def query_tasks(req: TaskQueryRequest):
    alias = req.alias
    data = {
        tid: build_task_summary(tid, info)
        for tid, info in active_tasks.items()
        if alias == "all" or info.get("alias") == alias
    }
    return {"code": 200, "message": "成功获取所有任务状态列表", "data": data}


@router.post("/task-status")
async def task_status(req: TaskStatusRequest):
    tid = req.task_id
    if tid not in active_tasks:
        return {"code": 404, "message": "任务不存在"}
    info = active_tasks.get(tid)
    return {"code": 200, "message": "成功", "data": build_task_summary(tid, info)}


@router.post("/update-classes")
async def update_classes(req: UpdateClassesRequest):
    """
    流直播过程中动态修改目标检测字典，不需要中断视频流。
    """
    tid = req.task_id
    if tid not in active_tasks:
        return {"code": 404, "message": "任务不存在"}

    target_classes = parse_names_dict(req.names_dict) if req.names_dict else {}
    current_conf = active_tasks.get_field(tid, "conf_thres", 0.25)
    try:
        pipeline_info = normalize_pipeline(
            model_pool,
            target_classes,
            getattr(req, "pipeline", None),
            current_conf,
            allow_empty=True,
        )
    except ValueError as exc:
        return {"code": 400, "message": str(exc)}
    active_tasks.set_field(tid, "target_classes", pipeline_info["target_classes"])
    active_tasks.set_field(tid, "selected_models", pipeline_info["selected_models"])
    active_tasks.set_field(tid, "pipeline", pipeline_info["pipeline"])
    return {
        "code": 200,
        "message": "动态修改检测目标成功",
        "data": {
            "task_id": tid,
            "target_classes": pipeline_info["target_classes"],
            "selected_models": pipeline_info["selected_models"],
            "pipeline": pipeline_info["pipeline"],
        },
    }


@router.websocket("/ws/{task_id}")
async def websocket_endpoint(websocket: WebSocket, task_id: str):
    """
    长连接监控流异常端点（防抖式预警）：
    前端或手机 App 直接以 ws:// 强连接上该地址，AI 判定有高价值抓拍事件将自动送回告警信息与照片回放。
    """
    await websocket.accept()

    if events_module.main_loop is None:
        events_module.main_loop = asyncio.get_running_loop()

    active_websockets[task_id].append(websocket)

    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        if websocket in active_websockets[task_id]:
            active_websockets[task_id].remove(websocket)
