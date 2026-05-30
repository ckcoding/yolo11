_base_ = 'mmyolo::rtmdet/rtmdet_tiny_syncbn_fast_8xb32-300e_coco.py'

# ⚠️ 此文件为手动 CLI 训练的静态模板。
# Web Console 使用 tools/generate_config.py 动态生成等价配置。
# 修改此文件时请同步 generate_config.py 的 CONFIG_TEMPLATE，反之亦然。

# 导入独立的包装函数以确保 DataLoader 不受 registry kwargs 影响，同时通过 custom_imports 让 Config 安全 dump
custom_imports = dict(imports=['trainer_console.collate_wrappers'], allow_failed_imports=False)

data_root = 'data/your_dataset/'
class_name = ('small_object', )
num_classes = len(class_name)

metainfo = dict(
    classes=class_name,
    palette=[(31, 119, 180)],
)

# Drone small-object oriented defaults.
img_scale = (1024, 1024)
max_epochs = 200
save_epoch_intervals = 10
train_batch_size_per_gpu = 8
train_num_workers = 4
val_batch_size_per_gpu = 4
val_num_workers = 2
base_lr = 0.004

load_from = None

model = dict(
    backbone=dict(init_cfg=None),
    bbox_head=dict(head_module=dict(num_classes=num_classes)),
    # Keep more candidates and lower the confidence floor for tiny objects.
    test_cfg=dict(score_thr=0.001, nms_pre=30000, max_per_img=300),
)

train_cfg = dict(
    max_epochs=max_epochs,
    val_begin=10,
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
