from __future__ import annotations

import os
from pathlib import Path


MMYOLO_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(os.getenv('TRAINER_PROJECT_ROOT', MMYOLO_ROOT.parent)).resolve()
STATE_ROOT = Path(os.getenv('TRAINER_STATE_ROOT', MMYOLO_ROOT / 'runtime')).resolve()
STATIC_ROOT = MMYOLO_ROOT / 'trainer_console' / 'static' / 'dist'
if not (STATIC_ROOT / 'index.html').exists():
    STATIC_ROOT = MMYOLO_ROOT / 'trainer_console' / 'static'
STATIC_INDEX = STATIC_ROOT / 'index.html'
if STATIC_ROOT.name == 'static' and (STATIC_ROOT / 'index.legacy.html').exists():
    STATIC_INDEX = STATIC_ROOT / 'index.legacy.html'

HOST = os.getenv('TRAINER_HOST', '0.0.0.0')
PORT = int(os.getenv('TRAINER_PORT', '18080'))
RELOAD = os.getenv('TRAINER_RELOAD', '0') == '1'

# 数据集自动扫描根目录（用户所有数据集的统一存放目录）
DATASET_SCAN_ROOT = Path(os.getenv('TRAINER_DATASET_SCAN_ROOT', '/home/dataList'))
DEFAULT_RTMDET_PRETRAIN = MMYOLO_ROOT / 'cspnext-tiny_imagenet_600e.pth'

DEFAULT_BROWSE_ROOTS = [
    PROJECT_ROOT,
    DATASET_SCAN_ROOT,
    Path('/workspace/project'),
    Path('/workspace'),
    Path('/data'),
    Path('/datasets'),
    Path('/home/dataList'),
    Path('/tmp'),
    Path('/hostfs'),
]


def get_browse_roots() -> list[Path]:
    raw = os.getenv('TRAINER_BROWSE_ROOTS', '')
    if raw.strip():
        roots = [Path(item.strip()).resolve() for item in raw.split(':') if item.strip()]
    else:
        roots = []

    merged: list[Path] = []
    for root in roots + DEFAULT_BROWSE_ROOTS:
        try:
            candidate = root.resolve()
        except FileNotFoundError:
            candidate = root
        if candidate.exists() and candidate not in merged:
            merged.append(candidate)
    return merged


BROWSE_ROOTS = get_browse_roots()
JOBS_ROOT = STATE_ROOT / 'jobs'
LOGS_ROOT = STATE_ROOT / 'logs'
DATA_ROOT = STATE_ROOT / 'data'
CONFIGS_ROOT = STATE_ROOT / 'configs'
WORKDIR_ROOT = STATE_ROOT / 'work_dirs'

for path in [STATE_ROOT, JOBS_ROOT, LOGS_ROOT, DATA_ROOT, CONFIGS_ROOT, WORKDIR_ROOT]:
    path.mkdir(parents=True, exist_ok=True)
