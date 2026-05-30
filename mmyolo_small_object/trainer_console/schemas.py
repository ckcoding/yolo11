from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, field_validator


class TrainingRequest(BaseModel):
    job_name: str = Field(default='', description='任务显示名称')
    dataset_path: str = Field(description='源数据集路径，支持 YOLO txt 或 COCO')
    prepared_dataset_path: str = Field(default='', description='如已是可训练数据集，可直接填写')
    device_type: Literal['cuda'] = 'cuda'
    arch: Literal['rtmdet', 'yolov8'] = 'rtmdet'
    img_scale: int = 1024
    max_epochs: int = 200
    save_epoch_intervals: int = 10
    tile_size: int = 1280
    tile_overlap: int = 320
    min_intersection_ratio: float = 0.4
    min_bbox_side: float = 2.0
    max_empty_tiles: int = 2
    train_batch_size: int = 8
    val_batch_size: int = 4
    train_workers: int = 4
    val_workers: int = 2
    base_lr: float = 0.004
    load_from: str = ''
    prepare_dataset: bool = True
    overwrite_prepared: bool = True
    launcher: Literal['none', 'pytorch'] = 'none'
    device_visible_ids: str = '0'
    extra_train_args: str = ''

    @field_validator('dataset_path', 'prepared_dataset_path', 'load_from', 'extra_train_args')
    @classmethod
    def strip_strings(cls, value: str) -> str:
        return value.strip()

    @field_validator('job_name')
    @classmethod
    def normalize_job_name(cls, value: str) -> str:
        return value.strip()


class JobSummary(BaseModel):
    job_id: str
    job_name: str
    status: str
    stage: str
    arch: str
    created_at: datetime
    updated_at: datetime
    dataset_path: str
    prepared_dataset_path: str
    work_dir: str
    config_path: str
    progress_epoch: int = 0
    progress_iter: int = 0
    progress_total_iter: int = 0
    metrics: dict = Field(default_factory=dict)
    latest_message: str = ''
    exit_code: int | None = None


class JobDetail(JobSummary):
    log_path: str
    checkpoints: list[str] = Field(default_factory=list)
    command_history: list[list[str]] = Field(default_factory=list)
    request: dict = Field(default_factory=dict)
    stopped_by_user: bool = False


class LogResponse(BaseModel):
    job_id: str
    text: str
    size: int


class FileEntry(BaseModel):
    name: str
    path: str
    is_dir: bool
    size: int


class DirectoryResponse(BaseModel):
    current_path: str
    parent_path: str | None
    roots: list[str]
    entries: list[FileEntry]


class DatasetInspectRequest(BaseModel):
    path: str


class DatasetInspectResponse(BaseModel):
    path: str
    exists: bool
    dataset_type: str
    splits: dict
    yaml_path: str | None = None
    message: str = ''
