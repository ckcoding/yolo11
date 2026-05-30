from __future__ import annotations

from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from trainer_console.config import (
    BROWSE_ROOTS,
    DATASET_SCAN_ROOT,
    DEFAULT_RTMDET_PRETRAIN,
    HOST,
    PORT,
    RELOAD,
    STATIC_INDEX,
    STATIC_ROOT,
)
from trainer_console.dataset_utils import inspect_dataset_structure
from trainer_console.dataset_scanner import scan_datasets
from trainer_console.job_manager import job_manager
from trainer_console.schemas import DatasetInspectRequest, DirectoryResponse, FileEntry, TrainingRequest


app = FastAPI(
    title='MMYOLO Trainer Console',
    description='用于无人机小目标训练的 MMYOLO Docker 控制台',
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)


def resolve_default_load_from(arch: str) -> str:
    if arch == 'rtmdet' and DEFAULT_RTMDET_PRETRAIN.exists():
        return str(DEFAULT_RTMDET_PRETRAIN)
    return ''


def ensure_within_roots(path: Path) -> Path:
    candidate = path.expanduser().resolve()
    for root in BROWSE_ROOTS:
        try:
            root_resolved = root.resolve()
        except FileNotFoundError:
            root_resolved = root
        if candidate == root_resolved or root_resolved in candidate.parents:
            return candidate
    raise HTTPException(status_code=400, detail=f'路径不在允许浏览范围内: {candidate}')


@app.get('/api/health')
def health() -> dict:
    return {
        'ok': True,
        'browse_roots': [str(item) for item in BROWSE_ROOTS],
        'dataset_scan_root': str(DATASET_SCAN_ROOT),
        'jobs': len(job_manager.list_jobs()),
    }


@app.get('/api/presets')
def presets() -> dict:
    return {
        'recommended': {
            'device_type': 'cuda',
            'arch': 'rtmdet',
            'img_scale': 1024,
            'max_epochs': 200,
            'tile_size': 1280,
            'tile_overlap': 320,
            'min_intersection_ratio': 0.4,
            'min_bbox_side': 2.0,
            'max_empty_tiles': 2,
            'train_batch_size': 8,
            'val_batch_size': 4,
            'train_workers': 4,
            'val_workers': 2,
            'base_lr': 0.004,
            'load_from': resolve_default_load_from('rtmdet'),
            'launcher': 'none',
            'device_visible_ids': '0',
        }
    }


@app.get('/api/jobs')
def list_jobs() -> list[dict]:
    return job_manager.list_jobs()


@app.post('/api/jobs')
def create_job(request: TrainingRequest) -> dict:
    if not request.dataset_path and not request.prepared_dataset_path:
        raise HTTPException(status_code=400, detail='dataset_path 或 prepared_dataset_path 至少要填一个')
    if not request.prepare_dataset and not request.prepared_dataset_path:
        request.prepared_dataset_path = request.dataset_path
    if not request.load_from:
        request.load_from = resolve_default_load_from(request.arch)
    return job_manager.create_job(request)


@app.get('/api/jobs/{job_id}')
def get_job(job_id: str) -> dict:
    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail='任务不存在')
    return job


@app.post('/api/jobs/{job_id}/stop')
def stop_job(job_id: str) -> dict:
    job = job_manager.stop_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail='任务不存在')
    return job


@app.get('/api/jobs/{job_id}/logs')
def get_logs(job_id: str, max_bytes: int = Query(default=40000, ge=1000, le=500000)) -> dict:
    log = job_manager.read_log(job_id, max_bytes=max_bytes)
    if not log:
        raise HTTPException(status_code=404, detail='任务不存在')
    return log


@app.get('/api/fs')
def browse_files(path: str | None = None) -> DirectoryResponse:
    current = ensure_within_roots(Path(path)) if path else BROWSE_ROOTS[0]
    if not current.exists():
        raise HTTPException(status_code=404, detail='目录不存在')
    if not current.is_dir():
        raise HTTPException(status_code=400, detail='路径不是目录')

    entries = []
    for item in sorted(current.iterdir(), key=lambda value: (not value.is_dir(), value.name.lower())):
        try:
            stat = item.stat()
        except OSError:
            continue
        entries.append(
            FileEntry(
                name=item.name,
                path=str(item.resolve()),
                is_dir=item.is_dir(),
                size=stat.st_size if item.is_file() else 0,
            )
        )

    parent_path = None
    if current.parent != current:
        for root in BROWSE_ROOTS:
            root_resolved = root.resolve()
            if current == root_resolved or root_resolved in current.parents:
                parent_path = str(current.parent.resolve()) if current != root_resolved else None
                break

    return DirectoryResponse(
        current_path=str(current),
        parent_path=parent_path,
        roots=[str(item) for item in BROWSE_ROOTS],
        entries=entries,
    )


@app.post('/api/dataset/inspect')
def inspect_dataset(request: DatasetInspectRequest) -> dict:
    path = ensure_within_roots(Path(request.path))
    return inspect_dataset_structure(path)


@app.get('/api/datasets/scan')
def scan_all_datasets() -> dict:
    """自动扫描 /home/dataList 下所有数据集，返回发现的数据集列表。"""
    datasets = scan_datasets()
    return {
        'scan_root': str(DATASET_SCAN_ROOT),
        'scan_root_exists': DATASET_SCAN_ROOT.exists(),
        'total': len(datasets),
        'datasets': datasets,
    }


class ScanPathRequest(BaseModel):
    path: str
    max_depth: int = 2


@app.post('/api/datasets/scan-path')
def scan_datasets_at_path(request: ScanPathRequest) -> dict:
    """扫描指定路径下的所有数据集。"""
    target = ensure_within_roots(Path(request.path))
    datasets = scan_datasets(root=target, max_depth=request.max_depth)
    return {
        'scan_root': str(target),
        'scan_root_exists': target.exists(),
        'total': len(datasets),
        'datasets': datasets,
    }


@app.get('/')
def index() -> FileResponse:
    return FileResponse(STATIC_INDEX)


@app.get('/app.js')
def app_js() -> FileResponse:
    return FileResponse(STATIC_ROOT / 'app.js')


@app.get('/styles.css')
def styles_css() -> FileResponse:
    return FileResponse(STATIC_ROOT / 'styles.css')


app.mount('/static', StaticFiles(directory=str(STATIC_ROOT), html=True), name='static')


if __name__ == '__main__':
    uvicorn.run('trainer_console.main:app', host=HOST, port=PORT, reload=RELOAD)
