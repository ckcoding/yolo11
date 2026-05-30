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
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from openmmlab_service.api.router import router
from openmmlab_service.config import (
    ARTIFACTS_DIR,
    ARTIFACTS_URL_PREFIX,
    CORS_ALLOW_CREDENTIALS,
    CORS_ALLOW_ORIGINS,
    SERVER_HOST,
    SERVER_PORT,
    UVICORN_RELOAD,
)
from openmmlab_service.core.pool import model_pool


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


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动前：初始化停车场对象池，预热所有 GPU 算力资源放入列队中
    model_pool.init_pool()
    try:
        yield
    finally:
        # 关机后：释放清理
        model_pool.shutdown()

app = FastAPI(
    title="OpenMMLab 视频流与图像识别服务",
    description="兼容现有接口的 OpenMMLab 推理服务",
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
    uvicorn.run("openmmlab_service.main:app", host=SERVER_HOST, port=SERVER_PORT, reload=UVICORN_RELOAD)
