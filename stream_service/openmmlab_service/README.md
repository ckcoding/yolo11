# OpenMMLab Service

这是基于现有流式识别服务改出来的 OpenMMLab 版本，新目录独立运行，不影响原有 `app` 服务。

## 主要变化

- 保留原有 FastAPI 接口、任务状态、推流和图片识别流程。
- 推理后端从 `ultralytics` 改为 OpenMMLab，模型配置改为 `config + checkpoint`。
- 支持 MMDetection / MMYOLO 风格模型，启动时会自动注册 OpenMMLab 模块。

## 配置方式

编辑 [config.yml](/Users/ck/Downloads/yolo11_backup/stream_service/openmmlab_service/config.yml)：

```yaml
models:
  items:
    - id: "drone_person"
      config: "/workspace/models/drone_person.py"
      checkpoint: "/workspace/models/drone_person.pth"
      classes: ["drone_people"]
```

`classes` 建议显式填写。这样即使 checkpoint 缺少元信息，接口层也能正确完成标签映射。

## 启动

先安装 OpenMMLab 依赖：

```bash
pip install -r openmmlab_service/requirements.txt
mim install "mmengine>=0.10.3" "mmcv>=2.0.1,<2.2.0" "mmdet>=3.2.0,<3.4.0"
```

本地模块方式：

```bash
python -m openmmlab_service.main
```

容器方式：

```bash
cd openmmlab_service
docker compose up --build
```

## Docker 说明

- 容器工作目录是 `/workspace/openmmlab_service`
- 代码目录会挂载到 `/workspace/openmmlab_service`
- 模型配置和权重从宿主机 `../models` 挂载到容器 `/workspace/models`
- 字体目录从宿主机 `../fonts` 挂载到容器 `/workspace/fonts`

因此 [config.yml](/Users/ck/Downloads/yolo11_backup/stream_service/openmmlab_service/config.yml) 里的 `config` 和 `checkpoint` 路径保持写成 `/workspace/models/...` 即可。
