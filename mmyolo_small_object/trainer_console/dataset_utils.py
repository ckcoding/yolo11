from __future__ import annotations

from pathlib import Path


def detect_dataset_type(path: Path) -> str:
    if not path.exists():
        return 'missing'
    if (path / 'annotations').exists():
        return 'coco'

    yolo_markers = [
        path / 'images',
        path / 'labels',
        path / 'train' / 'images',
        path / 'val' / 'images',
        path / 'test' / 'images',
    ]
    if any(item.exists() for item in yolo_markers):
        return 'yolo'
    return 'unknown'


def inspect_dataset_structure(path: Path) -> dict:
    dataset_type = detect_dataset_type(path)
    result = {
        'path': str(path),
        'exists': path.exists(),
        'dataset_type': dataset_type,
        'splits': {},
        'yaml_path': None,
        'message': '',
    }
    if not path.exists():
        result['message'] = '路径不存在'
        return result

    if dataset_type == 'coco':
        ann_dir = path / 'annotations'
        for split in ['train', 'val', 'test']:
            ann = ann_dir / f'instances_{split}.json'
            image_dir = path / split
            if ann.exists() or image_dir.exists():
                result['splits'][split] = {
                    'annotation': str(ann) if ann.exists() else '',
                    'images': str(image_dir) if image_dir.exists() else '',
                }
        result['message'] = '检测到 COCO 数据集'
        return result

    yaml_candidates = ['data.yaml', 'dataset.yaml', 'data.yml']
    for name in yaml_candidates:
        candidate = path / name
        if candidate.exists():
            result['yaml_path'] = str(candidate)
            break

    structures = [
        ('images/labels', path / 'images', path / 'labels'),
        ('split/images', path / 'train' / 'images', path / 'train' / 'labels'),
    ]
    if dataset_type == 'yolo':
        for split in ['train', 'val', 'test']:
            image_dir = path / 'images' / split
            label_dir = path / 'labels' / split
            if image_dir.exists() or label_dir.exists():
                result['splits'][split] = {
                    'images': str(image_dir) if image_dir.exists() else '',
                    'labels': str(label_dir) if label_dir.exists() else '',
                }

        if not result['splits']:
            for split in ['train', 'val', 'test']:
                image_dir = path / split / 'images'
                label_dir = path / split / 'labels'
                if image_dir.exists() or label_dir.exists():
                    result['splits'][split] = {
                        'images': str(image_dir) if image_dir.exists() else '',
                        'labels': str(label_dir) if label_dir.exists() else '',
                    }

        if not result['splits'] and (path / 'images').exists():
            result['splits']['train'] = {
                'images': str(path / 'images'),
                'labels': str(path / 'labels') if (path / 'labels').exists() else '',
            }
        result['message'] = '检测到 YOLO txt 数据集'
        return result

    result['message'] = '未识别为标准 YOLO/COCO 数据集，请检查目录结构'
    return result
