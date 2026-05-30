import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import cv2
cv2.setNumThreads(1)

import torch
torch.set_num_threads(1)

import logging
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
from app.config import (
    ARTIFACTS_DIR, ARTIFACTS_URL_PREFIX, CORS_ALLOW_CREDENTIALS,
    CORS_ALLOW_ORIGINS, CUDA_AVAILABLE, GPU_COUNT, MODEL_HALF,
    MODEL_IMGSZ, NMS_FREE, SAHI_ENABLED, SAHI_SLICE_SIZE,
    SERVER_HOST, SERVER_PORT, USE_TENSORRT, UVICORN_RELOAD,
)
from app.api.router import router
from app.core.pool import model_pool


class QuietPollFilter(logging.Filter):
    """过滤掉高频轮询请求的访问日志，避免淹没关键诊断信息"""

    _NOISY_FRAGMENTS = (
        "POST /task-status",
        "OPTIONS /task-status",
        "OPTIONS /detect-stream",
    )

    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        return not any(frag in msg for frag in self._NOISY_FRAGMENTS)


def _build_speed_tag() -> str:
    """生成当前推理速度档位标识"""
    parts = []
    if MODEL_HALF:
        parts.append("FP16")
    if USE_TENSORRT:
        parts.append("TensorRT")
    if NMS_FREE:
        parts.append("NMS-Free")
    return "+".join(parts) if parts else "FP32标准"


def _print_startup_banner():
    """打印 UAV 小目标检测服务启动诊断信息"""
    speed_tag = _build_speed_tag()
    gpu_info = f"{GPU_COUNT}x GPU" if CUDA_AVAILABLE else "CPU 模式"
    sahi_info = f"SAHI 切片推理(slice={SAHI_SLICE_SIZE})" if SAHI_ENABLED else "SAHI 关闭"

    banner = (
        "\n"
        "╔══════════════════════════════════════════════════════════╗\n"
        "║       🛸 UAV 小目标实时检测引擎 · 已就绪              ║\n"
        "╠══════════════════════════════════════════════════════════╣\n"
        f"║  硬件:     {gpu_info:<44s} ║\n"
        f"║  速度档:   {speed_tag:<44s} ║\n"
        f"║  输入分辨:  {MODEL_IMGSZ}x{MODEL_IMGSZ:<39} ║\n"
        f"║  切片推理:  {sahi_info:<43s} ║\n"
        "╚══════════════════════════════════════════════════════════╝"
    )
    print(banner)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动前：初始化停车场对象池，预热所有 GPU 算力资源放入列队中
    model_pool.init_pool()
    _print_startup_banner()
    try:
        yield
    finally:
        # 关机时：先安全停止所有活跃推流任务
        from app.core.state import active_tasks
        for tid in active_tasks.keys():
            active_tasks.update_fields(tid, status="stopping")
        # 等待任务线程感知 stopping 状态并退出
        import time
        time.sleep(2)
        # 最终释放模型池资源
        model_pool.shutdown()
        print("[Shutdown] 所有推流任务和模型资源已安全释放")

app = FastAPI(
    title="UAV 小目标实时检测与视频流服务",
    description=(
        "无人机视角智能视觉分析平台 · 支持多 GPU 并发推理, "
        "FP16/TensorRT 加速, NMS-Free(YOLO26+), SAHI 切片推理"
    ),
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ALLOW_ORIGINS,
    allow_credentials=CORS_ALLOW_CREDENTIALS,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 挂载中心路由
app.include_router(router)
app.mount(ARTIFACTS_URL_PREFIX, StaticFiles(directory=str(ARTIFACTS_DIR)), name="artifacts")

if __name__ == "__main__":
    # 将轮询噪音过滤器注入 uvicorn 的 access 日志通道
    logging.getLogger("uvicorn.access").addFilter(QuietPollFilter())
    uvicorn.run("app.main:app", host=SERVER_HOST, port=SERVER_PORT, reload=UVICORN_RELOAD)

