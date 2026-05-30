FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04

# 设置非交互环境变量，防止 apt 安装时交互卡死
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV YOLO_CONFIG_DIR=/tmp
ENV PIP_DISABLE_PIP_VERSION_CHECK=1
ENV NVIDIA_DISABLE_REQUIRE=1

# 替换 apt 源为阿里云镜像源，安装 Python、curl、ffmpeg 和 OpenCV 运行时依赖
RUN sed -i 's/archive.ubuntu.com/mirrors.aliyun.com/g' /etc/apt/sources.list && \
    sed -i 's/security.ubuntu.com/mirrors.aliyun.com/g' /etc/apt/sources.list && \
    apt-get update && apt-get install -y \
    python3 python3-pip python3-dev python3-setuptools python3-wheel \
    tzdata curl ffmpeg \
    libsm6 libxext6 libgl1 libglib2.0-0 \
    && ln -sf /usr/bin/python3 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# 【优化1：分离出永远不太可能改变、且极大的核心库】
# 这样做的好处是，Docker 永远会把这一层缓存起来。只有当你后续修改 requirements.txt 时，才只会重装里面的小库。
COPY constraints.txt .
RUN python -m pip install --no-cache-dir --upgrade pip setuptools wheel && \
    python -m pip install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cu128 \
    torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0 && \
    python -m pip install --no-cache-dir -c constraints.txt -i https://mirrors.aliyun.com/pypi/simple/ \
    fastapi uvicorn "numpy<2.0" opencv-python-headless ultralytics ffmpeg-python loguru

# 【优化2：然后装业务线经常热更新的小依赖】
COPY requirements.txt .
RUN python -m pip install --no-cache-dir -c constraints.txt -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/

# 走 app.main 内部的 uvicorn 启动逻辑，是否开启 reload 由 config.yml / 环境变量控制。
CMD ["python", "-m", "app.main"]
