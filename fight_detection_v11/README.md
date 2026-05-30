# YOLO11 无人机打架识别监控系统 (V3.0-行为分析版)

本项目是基于 YOLO11-Pose 开发的专项行为识别监控系统，特别针对**无人机高空视角**下的打斗、冲突行为进行了逻辑优化。通过人体姿态关键点（Keypoints）结合多目标追踪（Tracking）实现高精度的实时预警。

## 0. 打架识别核心技术
- **姿态估计**: 采用 `yolo11n-pose.pt` 提取人体 17 个关键点。
- **冲突模型**: 基于人际距离（Proximity）与瞬时运动强度（Motion Intensity）的热力学判定算法。
- **无人机优化**: 引入了运动补偿逻辑，有效过滤因无人机自身飞行导致的画面抖动误报。

## 1. 核心 API 接口清单


默认服务端口：`8089`

### 1.1 获取可用模型列表
- **请求方式**: `POST`
- **路径**: `/api/models`
- **请求 Payload**: `{}`
- **响应示例**:
```json
{
  "code": 200,
  "message": "模型列表获取成功",
  "data": { "models": ["drone_car.onnx", "..."] }
}
```

### 1.2 获取模型类别清单
- **请求方式**: `POST`
- **路径**: `/api/model_classes`
- **请求 Payload**: `{"model_name": "..."}`

### 1.3 启动检测任务 (支持性能调优)
- **请求方式**: `POST`
- **路径**: `/api/start_detection`
- **请求 Payload**:
```json
{
  "stream_url": "rtmp://...",
  "model_name": "drone_car.onnx",
  "names_dict": "car",
  "alias": "gate_01",
  "conf_thres": 0.5,
  "inference_fps": 10,
  "confidence_display": true
}
```
- **关键新增字段**:
  - `inference_fps`: (Integer) 推理频率，范围 1-20。若设为 10，则每秒只进行 10 次 AI 检测。这是解决多路流导致的系统卡顿、显卡负载过高的**最有效工具**。

### 1.4 停止检测任务
- **请求方式**: `POST`
- **路径**: `/api/stop_detection`
- **请求 Payload**: `{"task_id": "..."}`

### 1.5 实时参数更新 (无感热切换)
- **请求方式**: `POST`
- **路径**: `/api/update_detection`
- **请求 Payload**:
```json
{
  "task_id": "gate_01_uuid_xxx",
  "model_name": "drone_car.onnx",
  "names_dict": "car",
  "alias": "gate_01_updated",
  "conf_thres": 0.6,
  "inference_fps": 5,
  "confidence_display": false
}
```

## 2. 实时数据流控制 (WebSocket)
- **URL**: `ws://[HOST]:8089/ws/stream/{task_id}`

## 3. 性能优化建议 (针对卡顿)
1. **调低 FPS**: 对于安防监控，`inference_fps` 调至 `5-10` 即可在大幅降低 GPU 压力的同时保持画面的实时流畅。
2. **硬件调度**: 任务越多，单张显卡的压力越大，建议开启多张 GPU 并行。
