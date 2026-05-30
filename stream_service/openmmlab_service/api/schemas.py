from pydantic import BaseModel
from typing import Optional


class PipelineNodeInput(BaseModel):
    id: Optional[str] = None
    model: str
    labels: Optional[list[str]] = None
    fps: Optional[float] = None
    conf: Optional[float] = None
    depends_on: Optional[str] = None
    roi_labels: Optional[list[str]] = None
    max_rois: Optional[int] = None


class StreamRequest(BaseModel):
    # 彻底兼容不同的 URL 字段叫法，防止 Pydantic 的别名机制强制拦截
    stream_url: Optional[str] = None
    m3u8_url: Optional[str] = None
    url: Optional[str] = None
    names_dict: Optional[str] = ""
    alias: Optional[str] = "default"
    mode: Optional[str] = "stream"  # "stream" 或 "websocket"
    confidence_display: Optional[bool] = False
    conf_thres: Optional[float] = 0.25
    pipeline: Optional[list[PipelineNodeInput]] = None

class ImageRequest(BaseModel):
    image_url: Optional[str] = None
    file_url: Optional[str] = None
    url: Optional[str] = None
    names_dict: Optional[str] = ""
    conf_thres: Optional[float] = 0.25
    pipeline: Optional[list[PipelineNodeInput]] = None

class StopRequest(BaseModel):
    task_id: str

class TaskStatusRequest(BaseModel):
    task_id: str

class TaskQueryRequest(BaseModel):
    alias: str

class UpdateClassesRequest(BaseModel):
    task_id: str
    names_dict: Optional[str] = ""
    pipeline: Optional[list[PipelineNodeInput]] = None
