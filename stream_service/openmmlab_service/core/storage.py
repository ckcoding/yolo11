import io
import json
import threading
import time
import uuid
from pathlib import Path

import cv2

try:
    from minio import Minio
except ImportError:
    Minio = None

from openmmlab_service.config import (
    ARTIFACTS_DIR,
    ARTIFACTS_URL_PREFIX,
    INPUTS_DIR,
    MINIO_ACCESS_KEY,
    MINIO_BUCKET,
    MINIO_ENABLED,
    MINIO_ENDPOINT,
    MINIO_PUBLIC_BASE,
    MINIO_SECRET_KEY,
    PROCESSED_DIR,
    PUBLIC_API_BASE,
    SNAPSHOT_DIR,
)

_CATEGORY_DIRS = {
    "inputs": INPUTS_DIR,
    "processed": PROCESSED_DIR,
    "snapshots": SNAPSHOT_DIR,
}

_minio_client = None
_minio_lock = threading.Lock()


def _relative_object_path(category: str, prefix: str, extension: str) -> Path:
    day = time.strftime("%Y%m%d")
    clean_ext = extension if extension.startswith(".") else f".{extension}"
    filename = f"{prefix}_{int(time.time())}_{uuid.uuid4().hex[:8]}{clean_ext}"
    return Path(category) / day / filename


def _build_local_url(relative_path: Path) -> str:
    return f"{PUBLIC_API_BASE}{ARTIFACTS_URL_PREFIX}/{relative_path.as_posix()}"


def _save_local(relative_path: Path, data: bytes) -> dict:
    full_path = ARTIFACTS_DIR / relative_path
    full_path.parent.mkdir(parents=True, exist_ok=True)
    full_path.write_bytes(data)
    return {
        "storage": "local",
        "url": _build_local_url(relative_path),
        "path": str(full_path),
        "relative_path": relative_path.as_posix(),
    }


def init_minio():
    global _minio_client
    if not MINIO_ENABLED:
        return None
    if Minio is None:
        print("[Storage] 未安装 minio 依赖，自动回退到本地存储")
        return None

    if _minio_client is not None:
        return _minio_client

    with _minio_lock:
        if _minio_client is None:
            try:
                client = Minio(
                    MINIO_ENDPOINT,
                    access_key=MINIO_ACCESS_KEY,
                    secret_key=MINIO_SECRET_KEY,
                    secure=False,
                )
                if not client.bucket_exists(MINIO_BUCKET):
                    client.make_bucket(MINIO_BUCKET)
                    policy = {
                        "Version": "2012-10-17",
                        "Statement": [
                            {
                                "Effect": "Allow",
                                "Principal": {"AWS": ["*"]},
                                "Action": ["s3:GetObject"],
                                "Resource": [f"arn:aws:s3:::{MINIO_BUCKET}/*"],
                            }
                        ],
                    }
                    client.set_bucket_policy(MINIO_BUCKET, json.dumps(policy))
                _minio_client = client
            except Exception as exc:
                print(f"[Storage] MinIO 初始化失败，将自动回退到本地存储: {exc}")
    return _minio_client


def persist_bytes(category: str, data: bytes, prefix: str, extension: str, content_type: str) -> dict:
    if category not in _CATEGORY_DIRS:
        raise ValueError(f"不支持的存储分类: {category}")

    if not MINIO_ENABLED:
        return {
            "storage": "none",
            "url": "",
            "path": "",
            "relative_path": "",
        }

    relative_path = _relative_object_path(category, prefix, extension)
    client = init_minio()
    if client is not None:
        try:
            buffer = io.BytesIO(data)
            client.put_object(
                MINIO_BUCKET,
                relative_path.as_posix(),
                buffer,
                len(data),
                content_type=content_type,
            )
            return {
                "storage": "minio",
                "url": f"{MINIO_PUBLIC_BASE}/{MINIO_BUCKET}/{relative_path.as_posix()}",
                "path": "",
                "relative_path": relative_path.as_posix(),
            }
        except Exception as exc:
            print(f"[Storage] MinIO 上传失败: {exc}")

    # 如果启用了但是上传失败，或者 minio 对象获取失败，同样不存本地磁盘
    return {
        "storage": "none",
        "url": "",
        "path": "",
        "relative_path": "",
    }


def persist_image_array(image, category: str, prefix: str, quality: int = 90) -> dict:
    success, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not success:
        raise RuntimeError("图像编码失败")
    return persist_bytes(category, encoded.tobytes(), prefix, ".jpg", "image/jpeg")
