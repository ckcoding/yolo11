#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


CONFIG_TEMPLATE = """_base_ = {base_config}

# 导入独立的包装函数以确保 DataLoader 不受 registry kwargs 影响，同时通过 custom_imports 让 Config 安全 dump
custom_imports = dict(imports=['trainer_console.collate_wrappers'], allow_failed_imports=False)

data_root = {data_root!r}
class_name = {class_name}
num_classes = len(class_name)

metainfo = dict(
    classes=class_name,
    palette={palette},
)

img_scale = ({img_scale}, {img_scale})
max_epochs = {max_epochs}
save_epoch_intervals = {save_epoch_intervals}
train_batch_size_per_gpu = {train_batch_size}
train_num_workers = {train_workers}
val_batch_size_per_gpu = {val_batch_size}
val_num_workers = {val_workers}
base_lr = {base_lr}

load_from = {load_from}

model = dict(
    # Disable base-config remote backbone downloads by default.
    # If a local pretrained checkpoint is desired, pass --load-from explicitly.
    backbone=dict(init_cfg=None),
    bbox_head=dict(head_module=dict(num_classes=num_classes)),
    test_cfg=dict(score_thr=0.001, nms_pre=30000, max_per_img=300),
)

train_cfg = dict(
    max_epochs=max_epochs,
    val_begin=min(10, max_epochs),
    val_interval=save_epoch_intervals,
)

default_hooks = dict(
    checkpoint=dict(interval=save_epoch_intervals, max_keep_ckpts=3, save_best='auto'),
    logger=dict(interval=20),
)

optim_wrapper = dict(optimizer=dict(lr=base_lr))

train_dataloader = dict(
    batch_size=train_batch_size_per_gpu,
    num_workers=train_num_workers,
    collate_fn=dict(type='safe_yolo_collate'),
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='annotations/instances_train.json',
        data_prefix=dict(img='train/'),
    ),
)

val_dataloader = dict(
    batch_size=val_batch_size_per_gpu,
    num_workers=val_num_workers,
    collate_fn=dict(type='safe_pseudo_collate'),
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='annotations/instances_val.json',
        data_prefix=dict(img='val/'),
    ),
)

test_dataloader = val_dataloader

val_evaluator = dict(ann_file=data_root + 'annotations/instances_val.json')
test_evaluator = val_evaluator
"""


def load_categories(ann_path: Path) -> tuple[str, ...]:
    if not ann_path.exists():
        raise FileNotFoundError(f'标注文件不存在: {ann_path}')

    with ann_path.open('r', encoding='utf-8') as f:
        payload = json.load(f)

    categories = sorted(payload.get('categories', []), key=lambda item: item['id'])
    names = tuple(item['name'] for item in categories)
    if not names:
        raise ValueError(f'未在 {ann_path} 中找到 categories')
    return names


def build_palette(num_classes: int) -> list[tuple[int, int, int]]:
    seed = [
        (31, 119, 180),
        (255, 127, 14),
        (44, 160, 44),
        (214, 39, 40),
        (148, 103, 189),
        (140, 86, 75),
        (227, 119, 194),
        (127, 127, 127),
        (188, 189, 34),
        (23, 190, 207),
    ]
    palette = []
    for idx in range(num_classes):
        palette.append(seed[idx % len(seed)])
    return palette


def main() -> int:
    parser = argparse.ArgumentParser(description='根据 COCO 标注生成 MMYOLO 训练配置')
    parser.add_argument('--ann', required=True, help='训练集 COCO 标注，例如 annotations/instances_train.json')
    parser.add_argument('--data-root', required=True, help='数据根目录')
    parser.add_argument('--out', required=True, help='输出配置文件路径')
    parser.add_argument('--arch', choices=['rtmdet', 'yolov8'], default='rtmdet',
                        help='模型基线，默认 rtmdet，适合无人机小目标第一版')
    parser.add_argument('--img-scale', type=int, default=1024, help='训练分辨率，默认 1024')
    parser.add_argument('--max-epochs', type=int, default=200, help='最大训练轮数')
    parser.add_argument('--save-epoch-intervals', type=int, default=10, help='保存和验证间隔')
    parser.add_argument('--train-batch-size', type=int, default=8, help='单卡训练 batch size')
    parser.add_argument('--val-batch-size', type=int, default=4, help='单卡验证 batch size')
    parser.add_argument('--train-workers', type=int, default=4, help='训练 dataloader workers')
    parser.add_argument('--val-workers', type=int, default=2, help='验证 dataloader workers')
    parser.add_argument('--base-lr', type=float, default=0.004, help='基础学习率')
    parser.add_argument('--load-from', default='None', help='预训练权重路径或 None')
    args = parser.parse_args()

    ann_path = Path(args.ann).resolve()
    out_path = Path(args.out).resolve()
    class_name = load_categories(ann_path)
    palette = build_palette(len(class_name))
    load_from = args.load_from if args.load_from == 'None' else repr(args.load_from)
    base_config = {
        'rtmdet': "'mmyolo::rtmdet/rtmdet_tiny_syncbn_fast_8xb32-300e_coco.py'",
        'yolov8': "'mmyolo::yolov8/yolov8_s_syncbn_fast_8xb16-500e_coco.py'",
    }[args.arch]

    content = CONFIG_TEMPLATE.format(
        base_config=base_config,
        data_root=args.data_root if args.data_root.endswith('/') else args.data_root + '/',
        class_name=repr(class_name),
        palette=repr(palette),
        img_scale=args.img_scale,
        max_epochs=args.max_epochs,
        save_epoch_intervals=args.save_epoch_intervals,
        train_batch_size=args.train_batch_size,
        train_workers=args.train_workers,
        val_batch_size=args.val_batch_size,
        val_workers=args.val_workers,
        base_lr=args.base_lr,
        load_from=load_from,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(content, encoding='utf-8')
    print(f'[OK] 配置已生成: {out_path}')
    print(f'[OK] 架构: {args.arch}')
    print(f'[OK] 类别: {class_name}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
