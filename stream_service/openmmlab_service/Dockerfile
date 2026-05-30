FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

# 设置非交互环境变量，防止 apt 安装时交互卡死
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV YOLO_CONFIG_DIR=/tmp
ENV PIP_DISABLE_PIP_VERSION_CHECK=1
ENV PYTHONPATH=/workspace

# 替换 apt 源为阿里云镜像源，安装包含完整 libx264 支持的 ubuntu 系统级 ffmpeg
RUN sed -i 's/archive.ubuntu.com/mirrors.aliyun.com/g' /etc/apt/sources.list && \
    sed -i 's/security.ubuntu.com/mirrors.aliyun.com/g' /etc/apt/sources.list && \
    apt-get update && apt-get install -y \
    tzdata ffmpeg \
    libsm6 libxext6 libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 强制卸载 conda 自带的不完整的骨架 ffmpeg（防止系统层和 conda 层命令冲突）
RUN conda remove -y ffmpeg || true

WORKDIR /workspace/openmmlab_service

COPY constraints.txt .
RUN pip install --no-cache-dir -c constraints.txt -i https://mirrors.aliyun.com/pypi/simple/ \
    fastapi uvicorn numpy opencv-python ffmpeg-python loguru openmim

# OpenMMLab 官方组件优先通过 mim 安装，减少 CUDA/torch 轮子不匹配问题
RUN mim install "mmengine>=0.10.3" "mmcv>=2.0.1,<2.2.0" "mmdet>=3.2.0,<3.4.0"

COPY requirements.txt .
RUN pip install --no-cache-dir -c constraints.txt -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/

COPY . /workspace/openmmlab_service

# 走 openmmlab_service.main 内部的 uvicorn 启动逻辑
CMD ["python", "-m", "openmmlab_service.main"]
