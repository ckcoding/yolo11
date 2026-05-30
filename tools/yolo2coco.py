#!/usr/bin/env python3
"""
YOLO 格式数据集 → COCO 格式数据集 转换脚本
用法: python yolo2coco.py --src /home/dataList/drone/drone_car --dst /home/dataList/drone/drone_car_coco
"""
import os
import json
import argparse
import shutil
from PIL import Image
from pathlib import Path


def find_dataset_structure(src_root):
    """自动探测 YOLO 数据集的目录结构"""
    src = Path(src_root)
    
    # 常见结构1: src/images/train, src/labels/train
    # 常见结构2: src/train/images, src/train/labels
    # 常见结构3: src/images, src/labels (无 train/val 子目录)
    
    splits = {}
    
    # 检查是否有 data.yaml
    yaml_path = None
    for name in ["data.yaml", "dataset.yaml", "data.yml"]:
        p = src / name
        if p.exists():
            yaml_path = p
            break
    
    # 尝试结构1: images/train, labels/train
    images_dir = src / "images"
    labels_dir = src / "labels"
    if images_dir.exists() and labels_dir.exists():
        # 检查子目录
        for split_name in ["train", "val", "test"]:
            img_split = images_dir / split_name
            lbl_split = labels_dir / split_name
            if img_split.exists():
                splits[split_name] = {
                    "images": img_split,
                    "labels": lbl_split if lbl_split.exists() else None,
                }
        # 如果没有子目录，说明所有图片直接在 images/ 下
        if not splits:
            splits["train"] = {
                "images": images_dir,
                "labels": labels_dir,
            }
    
    # 尝试结构2: train/images, train/labels
    if not splits:
        for split_name in ["train", "val", "test"]:
            split_dir = src / split_name
            img_dir = split_dir / "images"
            lbl_dir = split_dir / "labels"
            if img_dir.exists():
                splits[split_name] = {
                    "images": img_dir,
                    "labels": lbl_dir if lbl_dir.exists() else None,
                }
    
    return splits, yaml_path


def parse_classes_from_yaml(yaml_path):
    """从 data.yaml 中解析类别名"""
    classes = []
    if yaml_path is None:
        return classes
    
    try:
        # 简单解析，不依赖 pyyaml
        with open(yaml_path, "r", encoding="utf-8") as f:
            content = f.read()
        
        # 查找 names: [...] 或 names:\n  - xxx
        in_names = False
        for line in content.split("\n"):
            stripped = line.strip()
            if stripped.startswith("names:"):
                rest = stripped[6:].strip()
                if rest.startswith("["):
                    # 内联列表: names: ['car', 'truck']
                    rest = rest.strip("[]")
                    for item in rest.split(","):
                        item = item.strip().strip("'\"")
                        if item:
                            classes.append(item)
                    return classes
                elif rest.startswith("{"):
                    # 内联字典: names: {0: 'car', 1: 'truck'}
                    rest = rest.strip("{}")
                    for item in rest.split(","):
                        if ":" in item:
                            val = item.split(":", 1)[1].strip().strip("'\"")
                            if val:
                                classes.append(val)
                    return classes
                else:
                    in_names = True
                    continue
            
            if in_names:
                if stripped.startswith("- "):
                    classes.append(stripped[2:].strip().strip("'\""))
                elif stripped and ":" in stripped and not stripped.startswith("#"):
                    # 字典格式: 0: car
                    val = stripped.split(":", 1)[1].strip().strip("'\"")
                    if val:
                        classes.append(val)
                elif stripped and not stripped.startswith("#") and not stripped.startswith("-"):
                    # 遇到非 names 的其他 key，结束
                    break
    except Exception as e:
        print(f"[WARN] 解析 {yaml_path} 失败: {e}")
    
    return classes


def scan_classes_from_labels(labels_dir):
    """从标签文件中扫描所有出现的类别 ID"""
    max_cls = -1
    if labels_dir is None or not labels_dir.exists():
        return max_cls
    
    for txt_file in labels_dir.glob("*.txt"):
        try:
            with open(txt_file, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cls_id = int(parts[0])
                        if cls_id > max_cls:
                            max_cls = cls_id
        except Exception:
            pass
    return max_cls


def convert_split(split_name, img_dir, lbl_dir, dst_root, categories, start_img_id=1, start_ann_id=1):
    """转换一个 split（train/val/test）"""
    dst = Path(dst_root)
    dst_img_dir = dst / split_name
    dst_img_dir.mkdir(parents=True, exist_ok=True)
    
    images_list = []
    annotations_list = []
    
    img_id = start_img_id
    ann_id = start_ann_id
    
    # 收集所有图片
    img_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    img_files = sorted([
        f for f in img_dir.iterdir()
        if f.suffix.lower() in img_extensions
    ])
    
    total = len(img_files)
    converted = 0
    skipped = 0
    
    for idx, img_path in enumerate(img_files):
        # 获取图片尺寸
        try:
            with Image.open(img_path) as im:
                width, height = im.size
        except Exception as e:
            print(f"  [SKIP] 无法读取图片 {img_path.name}: {e}")
            skipped += 1
            continue
        
        # 复制图片到目标目录
        dst_img_path = dst_img_dir / img_path.name
        if not dst_img_path.exists():
            shutil.copy2(img_path, dst_img_path)
        
        images_list.append({
            "id": img_id,
            "file_name": img_path.name,
            "width": width,
            "height": height,
        })
        
        # 查找对应的标签文件
        if lbl_dir is not None:
            lbl_path = lbl_dir / (img_path.stem + ".txt")
            if lbl_path.exists():
                with open(lbl_path, "r") as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) < 5:
                            continue
                        
                        cls_id = int(parts[0])
                        x_center = float(parts[1])
                        y_center = float(parts[2])
                        w_norm = float(parts[3])
                        h_norm = float(parts[4])
                        
                        # YOLO 归一化坐标 → COCO 绝对坐标 (x, y, w, h)
                        abs_w = w_norm * width
                        abs_h = h_norm * height
                        abs_x = x_center * width - abs_w / 2
                        abs_y = y_center * height - abs_h / 2
                        
                        # COCO category_id 从 1 开始
                        annotations_list.append({
                            "id": ann_id,
                            "image_id": img_id,
                            "category_id": cls_id + 1,  # YOLO 从 0 开始，COCO 从 1 开始
                            "bbox": [round(abs_x, 2), round(abs_y, 2), round(abs_w, 2), round(abs_h, 2)],
                            "area": round(abs_w * abs_h, 2),
                            "iscrowd": 0,
                        })
                        ann_id += 1
        
        img_id += 1
        converted += 1
        
        if (idx + 1) % 500 == 0:
            print(f"  [{split_name}] 已处理 {idx+1}/{total} 张图片...")
    
    print(f"  [{split_name}] 完成: {converted} 张图片, {len(annotations_list)} 个标注, 跳过 {skipped} 张")
    
    # 生成 COCO JSON
    coco_json = {
        "images": images_list,
        "annotations": annotations_list,
        "categories": categories,
    }
    
    ann_dir = dst / "annotations"
    ann_dir.mkdir(parents=True, exist_ok=True)
    json_path = ann_dir / f"instances_{split_name}.json"
    
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(coco_json, f, ensure_ascii=False, indent=2)
    
    print(f"  [{split_name}] COCO JSON 已保存: {json_path}")
    return img_id, ann_id


def main():
    parser = argparse.ArgumentParser(description="YOLO → COCO 格式转换")
    parser.add_argument("--src", required=True, help="YOLO 数据集根目录")
    parser.add_argument("--dst", required=True, help="COCO 数据集输出目录")
    args = parser.parse_args()
    
    src = Path(args.src)
    dst = Path(args.dst)
    
    if not src.exists():
        print(f"[ERROR] 源目录不存在: {src}")
        return
    
    print(f"=== YOLO → COCO 转换 ===")
    print(f"源目录: {src}")
    print(f"目标目录: {dst}")
    
    # 1. 探测目录结构
    print(f"\n[1/3] 探测数据集结构...")
    splits, yaml_path = find_dataset_structure(src)
    
    if not splits:
        print(f"[ERROR] 无法识别数据集结构，请确认 {src} 下有 images/ 和 labels/ 目录")
        # 列出目录内容帮助诊断
        print(f"\n{src} 的内容:")
        for item in sorted(src.iterdir()):
            print(f"  {'[DIR] ' if item.is_dir() else '      '}{item.name}")
        return
    
    for split_name, info in splits.items():
        img_count = len(list(info["images"].glob("*"))) if info["images"].exists() else 0
        lbl_count = len(list(info["labels"].glob("*.txt"))) if info["labels"] and info["labels"].exists() else 0
        print(f"  [{split_name}] 图片: {img_count}, 标签: {lbl_count}")
        print(f"    图片目录: {info['images']}")
        print(f"    标签目录: {info['labels']}")
    
    if yaml_path:
        print(f"  配置文件: {yaml_path}")
    
    # 2. 解析类别
    print(f"\n[2/3] 解析类别信息...")
    classes = parse_classes_from_yaml(yaml_path)
    
    if not classes:
        # 从标签文件扫描最大类别 ID
        max_cls = -1
        for info in splits.values():
            mc = scan_classes_from_labels(info["labels"])
            if mc > max_cls:
                max_cls = mc
        
        if max_cls >= 0:
            classes = [f"class_{i}" for i in range(max_cls + 1)]
            print(f"  [WARN] 未找到类别名，使用自动生成的 {len(classes)} 个类别: {classes}")
        else:
            print(f"  [ERROR] 无法确定类别数量")
            return
    else:
        print(f"  找到 {len(classes)} 个类别: {classes}")
    
    # COCO categories（id 从 1 开始）
    categories = [
        {"id": i + 1, "name": name, "supercategory": "object"}
        for i, name in enumerate(classes)
    ]
    
    # 3. 转换每个 split
    print(f"\n[3/3] 开始转换...")
    dst.mkdir(parents=True, exist_ok=True)
    
    img_id = 1
    ann_id = 1
    for split_name, info in splits.items():
        if info["images"].exists():
            img_id, ann_id = convert_split(
                split_name,
                info["images"],
                info["labels"],
                dst,
                categories,
                start_img_id=img_id,
                start_ann_id=ann_id,
            )
    
    print(f"\n=== 转换完成! ===")
    print(f"COCO 数据集目录: {dst}")
    print(f"  annotations/  → COCO JSON 标注文件")
    for split_name in splits:
        print(f"  {split_name}/  → {split_name} 图片")


if __name__ == "__main__":
    main()
