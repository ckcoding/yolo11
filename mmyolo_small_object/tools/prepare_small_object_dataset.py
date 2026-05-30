#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

from PIL import Image


IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='把 YOLO/COCO 数据集整理成适合 MMYOLO 小目标训练的 COCO 切片数据集')
    parser.add_argument('--src', required=True, help='源数据集根目录，支持 YOLO 或 COCO')
    parser.add_argument('--dst', required=True, help='输出目录')
    parser.add_argument('--tile-size', type=int, default=1280, help='切片尺寸，默认 1280')
    parser.add_argument('--tile-overlap', type=int, default=320, help='切片重叠像素，默认 320')
    parser.add_argument('--min-intersection-ratio', type=float, default=0.4,
                        help='标注与切片相交面积占原框面积的最小比例，默认 0.4')
    parser.add_argument('--min-bbox-side', type=float, default=2.0,
                        help='切片后保留标注框最小边长，默认 2 像素')
    parser.add_argument('--max-empty-tiles', type=int, default=2,
                        help='每张原图最多保留多少张空切片，默认 2')
    parser.add_argument('--splits', nargs='+', default=['train', 'val'],
                        help='需要处理的 split，默认 train val')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--overwrite', action='store_true', help='覆盖输出目录内已有 train/val/test 和 annotations')
    return parser.parse_args()


def detect_dataset_type(src_root: Path) -> str:
    if (src_root / 'annotations').exists():
        return 'coco'

    yolo_markers = [
        src_root / 'images',
        src_root / 'labels',
        src_root / 'train' / 'images',
        src_root / 'val' / 'images',
    ]
    if any(path.exists() for path in yolo_markers):
        return 'yolo'

    raise FileNotFoundError(f'无法识别数据集结构: {src_root}')


def ensure_clean_output(dst_root: Path, overwrite: bool) -> None:
    targets = [dst_root / 'train', dst_root / 'val', dst_root / 'test', dst_root / 'annotations']
    existing = [path for path in targets if path.exists()]

    if existing and not overwrite:
        names = ', '.join(path.name for path in existing)
        raise FileExistsError(f'输出目录已存在内容: {names}，如需覆盖请加 --overwrite')

    if overwrite:
        for path in existing:
            shutil.rmtree(path)


def maybe_convert_yolo(src_root: Path, dst_root: Path) -> Path:
    repo_root = Path(__file__).resolve().parents[2]
    converter = repo_root / 'tools' / 'yolo2coco.py'
    if not converter.exists():
        raise FileNotFoundError(f'未找到转换脚本: {converter}')

    raw_root = dst_root / '_raw_coco'
    if raw_root.exists():
        shutil.rmtree(raw_root)
    raw_root.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(converter),
        '--src',
        str(src_root),
        '--dst',
        str(raw_root),
    ]
    print('[INFO] 检测到 YOLO 数据集，先转换为 COCO...')
    subprocess.run(cmd, check=True)
    return raw_root


def resolve_image_path(image_dir: Path, file_name: str) -> Path:
    direct = image_dir / file_name
    if direct.exists():
        return direct

    alt = image_dir.parent / file_name
    if alt.exists():
        return alt

    name_only = image_dir / Path(file_name).name
    if name_only.exists():
        return name_only

    raise FileNotFoundError(f'找不到图片: {file_name} (搜索目录: {image_dir})')


def sliding_starts(length: int, tile_size: int, overlap: int) -> list[int]:
    if length <= tile_size:
        return [0]

    stride = max(1, tile_size - overlap)
    starts = list(range(0, max(length - tile_size, 0) + 1, stride))
    tail = length - tile_size
    if starts[-1] != tail:
        starts.append(tail)
    return sorted(set(starts))


def clip_bbox_to_tile(bbox: list[float], x0: int, y0: int, x1: int, y1: int) -> tuple[list[float], float] | tuple[None, float]:
    bx, by, bw, bh = bbox
    bx2 = bx + bw
    by2 = by + bh

    inter_x1 = max(bx, x0)
    inter_y1 = max(by, y0)
    inter_x2 = min(bx2, x1)
    inter_y2 = min(by2, y1)

    inter_w = inter_x2 - inter_x1
    inter_h = inter_y2 - inter_y1
    if inter_w <= 0 or inter_h <= 0:
        return None, 0.0

    clipped = [inter_x1 - x0, inter_y1 - y0, inter_w, inter_h]
    ratio = (inter_w * inter_h) / max(bw * bh, 1e-6)
    return clipped, ratio


def load_coco_split(dataset_root: Path, split: str) -> tuple[dict, Path]:
    ann_path = dataset_root / 'annotations' / f'instances_{split}.json'
    if not ann_path.exists():
        raise FileNotFoundError(f'缺少标注文件: {ann_path}')

    image_dir = dataset_root / split
    if not image_dir.exists():
        alt_dir = dataset_root / 'images' / split
        if alt_dir.exists():
            image_dir = alt_dir
        else:
            raise FileNotFoundError(f'缺少图片目录: {image_dir}')

    with ann_path.open('r', encoding='utf-8') as f:
        coco = json.load(f)

    return coco, image_dir


def summarize_annotations(annotations: list[dict]) -> tuple[int, int, int]:
    small = medium = large = 0
    for ann in annotations:
        area = ann['bbox'][2] * ann['bbox'][3]
        if area < 32 * 32:
            small += 1
        elif area < 96 * 96:
            medium += 1
        else:
            large += 1
    return small, medium, large


def save_crop(image: Image.Image, crop_box: tuple[int, int, int, int], out_path: Path) -> None:
    cropped = image.crop(crop_box)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    suffix = out_path.suffix.lower()
    if suffix in {'.jpg', '.jpeg'} and cropped.mode not in {'RGB', 'L'}:
        cropped = cropped.convert('RGB')
    cropped.save(out_path)


def process_split(
    dataset_root: Path,
    dst_root: Path,
    split: str,
    tile_size: int,
    tile_overlap: int,
    min_intersection_ratio: float,
    min_bbox_side: float,
    max_empty_tiles: int,
    rng: random.Random,
) -> None:
    coco, image_dir = load_coco_split(dataset_root, split)
    images = coco.get('images', [])
    categories = coco.get('categories', [])
    anns_by_image: dict[int, list[dict]] = defaultdict(list)
    for ann in coco.get('annotations', []):
        anns_by_image[ann['image_id']].append(ann)

    dst_image_dir = dst_root / split
    dst_ann_dir = dst_root / 'annotations'
    dst_image_dir.mkdir(parents=True, exist_ok=True)
    dst_ann_dir.mkdir(parents=True, exist_ok=True)

    out_images = []
    out_annotations = []
    next_img_id = 1
    next_ann_id = 1

    for index, image_info in enumerate(images, start=1):
        image_path = resolve_image_path(image_dir, image_info['file_name'])
        with Image.open(image_path) as img:
            img = img.copy()
        width, height = img.size

        x_starts = sliding_starts(width, tile_size, tile_overlap)
        y_starts = sliding_starts(height, tile_size, tile_overlap)

        positive_tiles = []
        empty_tiles = []

        for y0 in y_starts:
            for x0 in x_starts:
                x1 = min(x0 + tile_size, width)
                y1 = min(y0 + tile_size, height)
                tile_annotations = []

                for ann in anns_by_image.get(image_info['id'], []):
                    clipped_bbox, ratio = clip_bbox_to_tile(ann['bbox'], x0, y0, x1, y1)
                    if clipped_bbox is None or ratio < min_intersection_ratio:
                        continue
                    if clipped_bbox[2] < min_bbox_side or clipped_bbox[3] < min_bbox_side:
                        continue

                    new_ann = {
                        'id': next_ann_id + len(tile_annotations),
                        'image_id': None,
                        'category_id': ann['category_id'],
                        'bbox': [round(v, 2) for v in clipped_bbox],
                        'area': round(clipped_bbox[2] * clipped_bbox[3], 2),
                        'iscrowd': ann.get('iscrowd', 0),
                    }
                    tile_annotations.append(new_ann)

                tile = (x0, y0, x1, y1, tile_annotations)
                if tile_annotations:
                    positive_tiles.append(tile)
                else:
                    empty_tiles.append(tile)

        if max_empty_tiles > 0 and empty_tiles:
            empty_tiles = rng.sample(empty_tiles, k=min(max_empty_tiles, len(empty_tiles)))
        else:
            empty_tiles = []

        selected_tiles = positive_tiles + empty_tiles
        if not selected_tiles:
            selected_tiles = [(0, 0, width, height, [])]

        for x0, y0, x1, y1, tile_annotations in selected_tiles:
            out_name = f"{Path(image_info['file_name']).stem}__x{x0}_y{y0}{image_path.suffix.lower()}"
            out_path = dst_image_dir / out_name
            save_crop(img, (x0, y0, x1, y1), out_path)

            out_images.append({
                'id': next_img_id,
                'file_name': out_name,
                'width': x1 - x0,
                'height': y1 - y0,
            })

            for ann in tile_annotations:
                ann['id'] = next_ann_id
                ann['image_id'] = next_img_id
                out_annotations.append(ann)
                next_ann_id += 1

            next_img_id += 1

        if index % 100 == 0:
            print(f'[INFO] [{split}] 已处理 {index}/{len(images)} 张原图')

    payload = {
        'images': out_images,
        'annotations': out_annotations,
        'categories': categories,
    }
    ann_path = dst_ann_dir / f'instances_{split}.json'
    with ann_path.open('w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    small, medium, large = summarize_annotations(out_annotations)
    print(f'[DONE] [{split}] 切片后图片数: {len(out_images)}, 标注数: {len(out_annotations)}')
    print(f'[DONE] [{split}] small={small}, medium={medium}, large={large}')
    print(f'[DONE] [{split}] 标注文件: {ann_path}')


def main() -> int:
    args = parse_args()
    rng = random.Random(args.seed)

    src_root = Path(args.src).resolve()
    dst_root = Path(args.dst).resolve()

    ensure_clean_output(dst_root, args.overwrite)
    dst_root.mkdir(parents=True, exist_ok=True)

    dataset_type = detect_dataset_type(src_root)
    if dataset_type == 'yolo':
        source_coco_root = maybe_convert_yolo(src_root, dst_root)
    else:
        source_coco_root = src_root

    print(f'[INFO] 数据集类型: {dataset_type}')
    print(f'[INFO] 输出目录: {dst_root}')

    for split in args.splits:
        try:
            process_split(
                dataset_root=source_coco_root,
                dst_root=dst_root,
                split=split,
                tile_size=args.tile_size,
                tile_overlap=args.tile_overlap,
                min_intersection_ratio=args.min_intersection_ratio,
                min_bbox_side=args.min_bbox_side,
                max_empty_tiles=args.max_empty_tiles,
                rng=rng,
            )
        except FileNotFoundError as exc:
            print(f'[WARN] 跳过 {split}: {exc}')

    print('[OK] 小目标数据集准备完成')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
