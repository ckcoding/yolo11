import uuid
import logging
import asyncio
import threading
from contextlib import asynccontextmanager
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import queue
import os
import uvicorn

# 设定基础日志输出格式
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 导入自定义的核心处理逻辑模块
from stream_worker import StreamWorker
from config import config
from minio_client import minio_uploader
from datetime import datetime, timedelta


# 维护当前正在运行的任务字典（线程安全）
_active_tasks_lock = threading.RLock()
active_tasks = {}


def safe_get_task(task_id):
    with _active_tasks_lock:
        return active_tasks.get(task_id)


def safe_pop_task(task_id):
    with _active_tasks_lock:
        return active_tasks.pop(task_id, None)


def safe_set_task(task_id, worker):
    with _active_tasks_lock:
        active_tasks[task_id] = worker

from model_manager import model_manager

# --- 请求与响应模型 ---
class ApiResponse(BaseModel):
    code: int = 200
    message: str = "success"
    data: dict = None

class ModelClassesRequest(BaseModel):
    model_name: str

class StopRequest(BaseModel):
    task_id: str

class StartRequest(BaseModel):
    stream_url: str
    model_name: str
    names_dict: str
    alias: str
    confidence_display: bool = False
    conf_thres: float = 0.5
    inference_fps: int = 10 
    fight_detection: bool = False # 新增：是否开启打架识别

class UpdateRequest(BaseModel):
    task_id: str
    model_name: str
    names_dict: str
    alias: str
    confidence_display: bool
    conf_thres: float
    inference_fps: int = 10 
    fight_detection: bool = False # 新增

# --- 周期性后台定时任务 ---
async def daily_cleanup_task():
    """后台常驻协程：每天凌晨 2 点准时执行 MinIO 清理计划"""
    while True:
        now = datetime.now()
        target_time = now.replace(hour=2, minute=0, second=0, microsecond=0)
        if now >= target_time:
            target_time += timedelta(days=1)
            
        wait_seconds = (target_time - now).total_seconds()
        logger.info(f">>> MinIO 存储巡检已就绪，将在 {wait_seconds/3600:.2f} 小时后执行自动清理(凌晨2点)")
        await asyncio.sleep(wait_seconds)
        
        try:
            minio_uploader.cleanup_old_images(days_ago=1)
        except Exception as e:
            logger.error(f"定时清理任务失败: {e}")
        
        await asyncio.sleep(60)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """初始化硬件预热并挂载后台任务"""
    logger.info(">>> Uvicorn API 挂载准备：引导系统分配全部核显算力预热本地 AI 模型池...")
    model_manager.preload_all()
    cleanup_task = asyncio.create_task(daily_cleanup_task())
    try:
        yield
    finally:
        cleanup_task.cancel()
        # 优雅关闭所有活跃任务
        with _active_tasks_lock:
            for tid in list(active_tasks.keys()):
                worker = active_tasks.pop(tid, None)
                if worker:
                    worker.stop()
        logger.info("所有推理任务已安全退出")

app = FastAPI(title="YOLO11 ONNX 视频流检测接口", lifespan=lifespan)

# 允许跨域请求
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/api/models")
async def get_models():
    """获取所有预制模型列表 (全 POST)"""
    return ApiResponse(
        message="模型列表获取成功",
        data={"models": model_manager.available_models}
    )

@app.post("/api/model_classes")
async def get_model_classes(req: ModelClassesRequest):
    """获取指定模型的类别列表 (JSON 入参)"""
    model = model_manager.get_model(req.model_name, "cpu")
    if not model:
        return ApiResponse(code=404, message="未找到该模型", data={})
    
    classes = list(model.names.values())
    return ApiResponse(
        message="模型类别查询成功",
        data={"model_name": req.model_name, "classes": classes}
    )

@app.post("/api/update_detection")
async def update_detection(req: UpdateRequest):
    """热切换/更新任务参数 (JSON 整体入参，URL 无动态拼接)"""
    task_id = req.task_id
    worker = safe_get_task(task_id)
    if worker is None:
        return ApiResponse(code=404, message="任务不存在", data={})
    
    worker.names_dict = req.names_dict
    worker.alias = req.alias
    worker.conf_thres = req.conf_thres
    worker.confidence_display = req.confidence_display
    worker.fight_detection = req.fight_detection # 新增
    
    if worker.model_name != req.model_name:
        worker.model_name = req.model_name
        worker.detector.model = model_manager.get_model(req.model_name, worker.device)
        logger.info(f"热切换任务模型: {task_id} -> {req.model_name}")
        
    return ApiResponse(message="热切换成功")

@app.get("/", response_class=HTMLResponse)
async def index():
    frontend_path = os.path.join(os.path.dirname(__file__), "frontend", "index.html")
    with open(frontend_path, "r", encoding="utf-8") as f:
        html = f.read()
    return html

@app.post("/api/start_detection")
async def start_detection(req: StartRequest):
    """启动新的检测任务"""
    base_id = str(uuid.uuid4())
    task_id = f"{req.alias}_{base_id}" if req.alias else base_id
    
    worker = StreamWorker(task_id, req)
    try:
        worker.start()
        safe_set_task(task_id, worker)
        return ApiResponse(
            message="推流任务已启动",
            data={"task_id": task_id}
        )
    except Exception as e:
        logger.error(f"启动失败: {e}")
        return ApiResponse(code=500, message=f"启动失败: {str(e)}", data={})

@app.post("/api/stop_detection")
async def stop_detection(req: StopRequest):
    """停止检测任务 (JSON 整体入参，URL 无动态拼接)"""
    task_id = req.task_id
    worker = safe_pop_task(task_id)
    if worker is None:
        return ApiResponse(code=404, message="任务不存在或已停止", data={})

    worker.stop()
    return ApiResponse(message="任务已成功截断并释放")

@app.websocket("/ws/stream/{task_id}")
async def stream_websocket(websocket: WebSocket, task_id: str):
    """建立 WebSocket 接口服务，接收后端的识别结果画面持续向前端推送实时的编码图像"""
    await websocket.accept()
    worker = safe_get_task(task_id)
    if worker is None:
        await websocket.close(code=1008, reason="无法查找对应的任务ID")
        return

    last_activity = asyncio.get_event_loop().time()
    
    try:
        while True:
            # 若控制标识被修改或线程被截停，则主动挂断与前台的大屏交互套接字
            if not worker.is_running:
                await websocket.close(code=1000, reason="侦测任务生命周期结束，正常停止连接。")
                break
                
            try:
                # 使用非阻塞方式获取帧，减少延迟
                b64_frame = await asyncio.get_event_loop().run_in_executor(None, worker.q.get, True, 0.1)
                await websocket.send_text(b64_frame)
                last_activity = asyncio.get_event_loop().time()
            except queue.Empty:
                # 队列为空时短暂休眠
                await asyncio.sleep(0.005)
                
                # 检查连接是否超时（30秒无数据则认为异常）
                if asyncio.get_event_loop().time() - last_activity > 30:
                    logger.warning(f"WebSocket 连接 {task_id} 超过30秒无数据传输，主动断开")
                    break
                
    except WebSocketDisconnect:
        logger.info(f"WebSocket 的收看页面遭遇主动或异常刷新失联，侦测推流随之中断: {task_id}")
    except Exception as e:
        logger.error(f"由于抛出未预期的报错而退出了对套接字推送连接，任务号 {task_id}，错误详文: {e}")
        try:
            await websocket.close()
        except:
            pass
    finally:
        # 【新增特性】当检测到哪怕是关闭/刷新网页造成的 Socket 脱线，都强制自动帮用户截停并在后台销毁这个耗时推理任务！
        if safe_get_task(task_id) is not None:
            logger.info(f"触发 WebSocket 离开安全策略：正在自动挂断并释放由于脱线闲置的任务 {task_id} 的庞大算力...")
            worker_to_stop = safe_pop_task(task_id)
            if worker_to_stop:
                worker_to_stop.stop()

if __name__ == "__main__":
    uvicorn.run(
        "main:app", 
        host="0.0.0.0", 
        port=8089,
        reload=False,
        ws_ping_interval=None,   # 禁用服务器端 ping，避免与数据流冲突导致 AssertionError
        ws_ping_timeout=None     # 禁用 ping 超时检测radical
    )
