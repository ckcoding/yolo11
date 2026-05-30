"""
自动扫描 /home/dataList 目录，发现所有可用数据集。

扫描策略：
1. 遍历 DATASET_SCAN_ROOT 下最多 2 层子目录
2. 对每个目录尝试判断是否是有效数据集（YOLO txt / COCO）
3. 收集元信息：名称、路径、类型、split、标签类别数等
"""
from __future__ import annotations

import json
from pathlib import Path

from trainer_console.config import DATASET_SCAN_ROOT
from trainer_console.dataset_utils import detect_dataset_type, inspect_dataset_structure


def _count_images(directory: Path) -> int:
    """快速统计目录下图片文件数量（不递归过深）。"""
    if not directory.exists():
        return 0
    count = 0
    suffixes = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}
    for item in directory.rglob('*'):
        if item.suffix.lower() in suffixes:
            count += 1
        if count > 99999:
            break
    return count


def _count_labels(directory: Path) -> int:
    """统计标签文件数量。"""
    if not directory.exists():
        return 0
    count = 0
    for item in directory.rglob('*'):
        if item.suffix.lower() in {'.txt', '.json', '.xml'}:
            count += 1
        if count > 99999:
            break
    return count


def _parse_yaml_classes(path: Path) -> list[str]:
    """尝试从 data.yaml / dataset.yaml 中读取类别名称。"""
    try:
        import yaml
    except ImportError:
        return []
    yaml_candidates = ['data.yaml', 'dataset.yaml', 'data.yml']
    for name in yaml_candidates:
        candidate = path / name
        if candidate.exists():
            try:
                with open(candidate, 'r', encoding='utf-8') as f:
                    data = yaml.safe_load(f)
                names = data.get('names', [])
                if isinstance(names, dict):
                    return list(names.values())
                if isinstance(names, list):
                    return names
            except Exception:
                pass
    return []


def _infer_yolo_classes_from_labels(path: Path) -> list[str]:
    """当 data.yaml 不存在时，从 YOLO 标签里反推类别 id。"""
    info = inspect_dataset_structure(path)
    class_ids: set[int] = set()

    for split_info in info.get('splits', {}).values():
        labels_dir = split_info.get('labels')
        if not labels_dir:
            continue
        labels_path = Path(labels_dir)
        if not labels_path.exists():
            continue

        for item in labels_path.rglob('*.txt'):
            try:
                text = item.read_text(encoding='utf-8', errors='ignore')
            except OSError:
                continue
            for line in text.splitlines():
                parts = line.strip().split()
                if not parts:
                    continue
                try:
                    class_id = int(float(parts[0]))
                except ValueError:
                    continue
                class_ids.add(class_id)
            if len(class_ids) > 256:
                break

    return [f'class_{idx}' for idx in sorted(class_ids)]


def _parse_coco_classes(path: Path) -> list[str]:
    """尝试从 COCO annotations JSON 中读取类别名称。"""
    ann_dir = path / 'annotations'
    if not ann_dir.exists():
        return []
    for name in ['instances_train.json', 'instances_val.json']:
        candidate = ann_dir / name
        if candidate.exists():
            try:
                with open(candidate, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                categories = data.get('categories', [])
                return [c.get('name', '') for c in categories if c.get('name')]
            except Exception:
                pass
    return []


def _get_dataset_size_mb(path: Path) -> float:
    """估算数据集大小（MB），只统计一层以节省时间。"""
    total = 0
    try:
        for item in path.rglob('*'):
            if item.is_file():
                total += item.stat().st_size
            if total > 50 * 1024 * 1024 * 1024:  # 超过 50GB 就不继续了
                break
    except OSError:
        pass
    return round(total / (1024 * 1024), 1)


def _build_split_stats(dataset_type: str, splits: dict) -> tuple[dict, int, int]:
    split_stats = {}
    total_images = 0
    total_labels = 0

    for split_name in ['train', 'val', 'test']:
        split_info = splits.get(split_name, {})
        image_count = _count_images(Path(split_info['images'])) if split_info.get('images') else 0
        if dataset_type == 'coco':
            label_count = 1 if split_info.get('annotation') else 0
        else:
            label_count = _count_labels(Path(split_info['labels'])) if split_info.get('labels') else 0

        if image_count or label_count or split_info:
            split_stats[split_name] = {
                'images': image_count,
                'labels': label_count,
                'has_images': bool(split_info.get('images')),
                'has_labels': bool(split_info.get('labels') or split_info.get('annotation')),
            }

        total_images += image_count
        total_labels += label_count

    return split_stats, total_images, total_labels


def scan_datasets(root: Path | None = None, max_depth: int = 2) -> list[dict]:
    """
    扫描指定目录下的所有数据集。
    
    Args:
        root: 扫描根目录，默认为 DATASET_SCAN_ROOT (/home/dataList)
        max_depth: 最大扫描深度
        
    Returns:
        数据集列表，每个元素包含 name, path, dataset_type, splits, classes 等信息
    """
    scan_root = (root or DATASET_SCAN_ROOT).resolve()
    if not scan_root.exists():
        return []

    datasets: list[dict] = []
    visited: set[str] = set()

    def _scan(directory: Path, depth: int) -> None:
        if depth > max_depth:
            return
        resolved = str(directory.resolve())
        if resolved in visited:
            return
        visited.add(resolved)

        if not directory.is_dir():
            return

        # 尝试检测当前目录是否为数据集
        dataset_type = detect_dataset_type(directory)
        if dataset_type in ('coco', 'yolo'):
            info = inspect_dataset_structure(directory)
            if dataset_type == 'yolo':
                classes = _parse_yaml_classes(directory)
                if not classes:
                    classes = _infer_yolo_classes_from_labels(directory)
            else:
                classes = _parse_coco_classes(directory)

            split_stats, total_images, total_labels = _build_split_stats(dataset_type, info.get('splits', {}))

            datasets.append({
                'name': directory.name,
                'path': str(directory),
                'dataset_type': dataset_type,
                'splits': info.get('splits', {}),
                'split_stats': split_stats,
                'yaml_path': info.get('yaml_path'),
                'classes': classes,
                'num_classes': len(classes),
                'total_images': total_images,
                'total_labels': total_labels,
                'message': info.get('message', ''),
            })
            return  # 已识别为数据集就不再往下递归

        # 否则继续递归子目录
        try:
            children = sorted(directory.iterdir(), key=lambda p: p.name.lower())
        except PermissionError:
            return

        for child in children:
            if child.name.startswith('.'):
                continue
            if child.is_dir():
                _scan(child, depth + 1)

    _scan(scan_root, 0)
    return datasets
