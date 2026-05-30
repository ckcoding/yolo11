# MMYOLO 小目标训练工程

这个目录是独立补出来的一套 `MMYOLO / OpenMMLab` 训练版本，重点针对无人机、远距离监控这类小目标场景。

当前默认策略不是盲目追求最复杂模型，而是先上更稳的基线：

1. 首选 `RTMDet` 做无人机小目标基线。
2. 训练前把大图切成有重叠的小块，等价于放大小目标。
3. 后续如果还要继续压榨极小目标，再扩展到 `YOLOv8-P2` 和 `NWD Loss`。

这样比单纯把输入分辨率拉高更稳，也比一开始就改骨干结构更容易把第一版效果跑出来。

## 目录说明

- `configs/rtmdet_tiny_smallobj_template.py`
  - 无人机小目标推荐模板，默认基于 `mmyolo::rtmdet/rtmdet_tiny_syncbn_fast_8xb32-300e_coco.py`
- `configs/yolov8_s_smallobj_template.py`
  - `YOLOv8` 备选模板
- `docs/SMALL_OBJECT_STRATEGY.md`
  - 无人机小目标训练策略说明
- `tools/prepare_small_object_dataset.py`
  - 支持 `YOLO -> COCO -> 切片数据集` 一步处理
- `tools/generate_config.py`
  - 根据 COCO 标注自动生成训练配置，默认生成 `RTMDet`
- `tools/check_env.py`
  - 检查本机 Python 和 MMYOLO 依赖是否可用
- `scripts/train_smallobj.sh`
  - 训练入口
- `scripts/test_smallobj.sh`
  - 评估入口

## 环境要求

当前工作区本机 Python 是 `3.14.3`，这不适合直接装 MMYOLO。

建议单独建一个 Python `3.10` 或 `3.11` 环境，然后再安装依赖。`MMYOLO 0.6.0` 对 `mmcv/mmdet/mmengine` 的版本要求比较严格，直接在高版本 Python 上装通常会踩兼容性问题。

先检查环境：

```bash
python tools/check_env.py
```

推荐安装顺序：

```bash
pip install -r requirements.txt
python -m mim install "mmengine>=0.10.3" "mmcv>=2.0.1,<2.2.0" "mmdet>=3.2.0,<3.4.0"
pip install mmyolo==0.6.0
```

如果你用的是 GPU 环境，`torch` / `torchvision` / CUDA 请按你机器实际版本单独装，不要盲目直接复制。

## 数据准备

这个工程默认训练数据用 COCO 格式：

```text
your_dataset/
  annotations/
    instances_train.json
    instances_val.json
  train/
  val/
```

如果你现有是 YOLO 标注目录，可以直接一步转换并切片：

```bash
python tools/prepare_small_object_dataset.py \
  --src /path/to/yolo_dataset \
  --dst data/drone_smallobj \
  --tile-size 1280 \
  --tile-overlap 320 \
  --min-intersection-ratio 0.4 \
  --max-empty-tiles 2 \
  --overwrite
```

常用参数说明：

- `--tile-size`
  - 切片尺寸，常用 `1024` 或 `1280`
- `--tile-overlap`
  - 切片重叠，建议 `tile-size` 的 `20%~30%`
- `--min-intersection-ratio`
  - 标注框和切片相交面积占原框面积的最小比例
- `--max-empty-tiles`
  - 每张原图最多保留多少负样本切片，防止空图过多

如果你的数据已经是 COCO 格式，也可以直接输入 COCO 根目录。

## 生成训练配置

数据准备完后，根据训练集类别自动生成配置。

无人机场景默认建议生成 `RTMDet` 配置：

```bash
python tools/generate_config.py \
  --ann data/drone_smallobj/annotations/instances_train.json \
  --data-root data/drone_smallobj \
  --out configs/drone_smallobj.py \
  --arch rtmdet \
  --img-scale 1024 \
  --max-epochs 200
```

如果你想切到 `YOLOv8` 基线：

```bash
python tools/generate_config.py \
  --ann data/drone_smallobj/annotations/instances_train.json \
  --data-root data/drone_smallobj \
  --out configs/drone_smallobj_yolov8.py \
  --arch yolov8 \
  --img-scale 1024 \
  --max-epochs 200
```

生成后的配置会自动写入类别名、数据根目录和小目标相关参数。

## 开始训练

```bash
bash scripts/train_smallobj.sh configs/drone_smallobj.py
```

指定工作目录：

```bash
bash scripts/train_smallobj.sh configs/drone_smallobj.py work_dirs/drone_smallobj
```

多卡训练示例：

```bash
CUDA_VISIBLE_DEVICES=0,1 bash scripts/train_smallobj.sh configs/drone_smallobj.py work_dirs/drone_smallobj --launcher pytorch
```

## 评估

```bash
bash scripts/test_smallobj.sh \
  configs/drone_smallobj.py \
  work_dirs/drone_smallobj/best_coco_bbox_mAP_epoch_*.pth
```

## 小目标建议

- 无人机训练先用 `RTMDet + 切片` 跑基线，通常比直接改 `P2` 更稳。
- 先做切片，再训练；这比只改 `img_scale` 更有效。
- 切片后优先用 `1024` 或 `1280` 分辨率训练。
- 推理阶段如果原图很大，也建议做同样的滑窗推理，否则训练和部署分布不一致。
- 如果你后面要继续压榨效果，再考虑改成带更浅层特征的 `P2` 检测头；这一步改模型结构，成本比数据切片高。
- `NWD Loss` 适合做第二阶段增强，但当前工程里还没有默认塞入未经验证的自定义实现。

## Docker 训练控制台

现在目录里已经补了一套 Web 训练控制台，适合直接在 Docker 里跑。

控制台能力：

- 输入 `YOLO txt` 或 `COCO` 数据集路径
- 自动切片、转换、生成配置
- 启动 `MMYOLO` 训练任务
- 查看任务状态、实时日志、训练指标和检查点
- 浏览挂载进容器的宿主机目录并一键回填路径
- 停止训练任务
- 支持在界面中选择 `NPU / CUDA / CPU` 设备类型

相关文件：

- `trainer_console/main.py`
- `trainer_console/job_manager.py`
- `trainer_console/static/index.html`
- `docker-compose.console.yml`
- `Dockerfile.console`

### 启动

```bash
cd /Users/ck/Downloads/yolo11_backup/mmyolo_small_object
docker compose -f docker-compose.console.yml up --build
```

启动后访问：

```text
http://localhost:18080
```

### 路径说明

容器里默认可浏览这些宿主机挂载路径：

- `/workspace/project`
  - 当前项目目录
- `/hosthome`
  - 你的宿主机 `HOME` 目录
- `/hosttmp`
  - 宿主机 `/tmp`

所以如果你的数据集原本在宿主机：

```text
/Users/ck/Downloads/datasets/drone_people
```

那么在控制台里通常填：

```text
/hosthome/Downloads/datasets/drone_people
```

### 运行产物

控制台会把运行数据写到：

```text
mmyolo_small_object/runtime/
```

其中包括：

- `jobs/`
  - 任务元数据
- `logs/`
  - 训练日志
- `data/`
  - 自动准备后的切片数据集
- `configs/`
  - 自动生成的训练配置
- `work_dirs/`
  - MMYOLO 训练输出和检查点

### NPU 说明

当前控制台已经支持在界面里选择 `NPU`，训练时会设置：

```text
ASCEND_RT_VISIBLE_DEVICES
ASCEND_VISIBLE_DEVICES
```

但要真正跑在昇腾 NPU 上，宿主机还必须提前具备对应的 `Ascend Docker Runtime / CANN / torch_npu` 运行环境。当前这个 Dockerfile 先解决了 `mmcv` 构建缺少 `g++` 的问题，并把控制台层改成了不再绑定 NVIDIA。
