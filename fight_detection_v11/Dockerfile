FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

# 设置非交互环境变量，防止 apt 安装时交互卡死
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV YOLO_CONFIG_DIR=/tmp
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

# 替换 apt 源为阿里云镜像源，安装包含完整 libx264 支持的 ubuntu 系统级 ffmpeg 等必须库
RUN sed -i 's/archive.ubuntu.com/mirrors.aliyun.com/g' /etc/apt/sources.list && \
    sed -i 's/security.ubuntu.com/mirrors.aliyun.com/g' /etc/apt/sources.list && \
    apt-get update && apt-get install -y \
    tzdata ffmpeg \
    libsm6 libxext6 libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 强制卸载 conda 自带的不完整的骨架 ffmpeg（防止系统层和 conda 层命令冲突）
RUN conda remove -y ffmpeg || true

WORKDIR /workspace

# 【优化1：分离出永远不太可能改变、且极大的核心库】
# 这样做的好处是，Docker 永远会把这一层缓存起来。只有当你后续修改 requirements.txt 时，才只会重装里面的小库。
COPY constraints.txt .
RUN pip install --no-cache-dir -c constraints.txt -i https://mirrors.aliyun.com/pypi/simple/ \
    fastapi uvicorn numpy opencv-python ultralytics ffmpeg-python loguru

# 【优化2：然后装业务线经常热更新的小依赖】
COPY requirements.txt .
RUN pip install --no-cache-dir -c constraints.txt -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/

# 拷贝项目源代码和目录资源（若通过 volume 动态挂载运行时请视需求调整）
COPY . /workspace

# 暴露端口，供 main.py 中的 uvicorn 服务使用
EXPOSE 8089

# 走项目入口文件 main.py 的启动逻辑
CMD ["python", "main.py"]
