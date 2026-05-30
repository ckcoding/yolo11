import os
from pathlib import Path

import torch
import yaml

# 服务目录在容器中通常是 /workspace，本地调试时则是 stream_service
SERVICE_DIR = Path(__file__).resolve().parent.parent
PROJECT_DIR = SERVICE_DIR.parent if (SERVICE_DIR.parent / "fonts").exists() else SERVICE_DIR
CWD_DIR = Path.cwd()

# ---- 顶级配置系统装载 ----
CONFIG_YAML_PATH = SERVICE_DIR / "config.yml"
_cfg = {}
if CONFIG_YAML_PATH.exists():
    with CONFIG_YAML_PATH.open("r", encoding="utf-8") as f:
        _cfg = yaml.safe_load(f) or {}


def get_conf(section, key, default=None):
    """从字典读取嵌套配置，不存在则回落至 default"""
    sec = _cfg.get(section)
    if not isinstance(sec, dict):
        sec = {}
    return sec.get(key, default)


def _get_env_or_conf(env_key, section, key, default=None):
    value = os.environ.get(env_key)
    if value not in (None, ""):
        return value
    return get_conf(section, key, default)


def _as_bool(value, default=False):
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _as_int(value, default):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _as_float(value, default):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_origins(value):
    if isinstance(value, str):
        items = [item.strip() for item in value.split(",") if item.strip()]
        return items or ["*"]
    if isinstance(value, list):
        items = [str(item).strip() for item in value if str(item).strip()]
        return items or ["*"]
    return ["*"]


def _normalize_base_url(value):
    return str(value).rstrip("/")


def _normalize_url_path(value, default):
    if value in (None, ""):
        return default
    cleaned = "/" + str(value).strip().strip("/")
    return cleaned if cleaned != "/" else default


TORCH_VERSION = getattr(torch, "__version__", "unknown")
TORCH_CUDA_VERSION = getattr(torch.version, "cuda", None) or "cpu"
FORCE_CPU = _as_bool(os.environ.get("FORCE_CPU"), False)


def _load_label_display_names():
    raw_mapping = get_conf("labels", "display_names", {})
    if not isinstance(raw_mapping, dict):
        return {}

    normalized = {}
    for raw_label, raw_name in raw_mapping.items():
        label = str(raw_label).strip()
        name = str(raw_name).strip()
        if label and name:
            normalized[label] = name
    return normalized


LABEL_DISPLAY_NAMES = _load_label_display_names()


def get_label_display_name(label: str) -> str:
    return LABEL_DISPLAY_NAMES.get(str(label).strip(), str(label).strip())


def resolve_path(raw_path):
    if raw_path in (None, ""):
        return ""

    raw = str(raw_path).strip()
    path = Path(raw)
    candidates = []

    normalized_raw = raw.replace("\\", "/")
    if normalized_raw.startswith("/workspace/"):
        workspace_relative = Path(normalized_raw[len("/workspace/"):])
        if normalized_raw.startswith("/workspace/media/"):
            media_relative = Path(normalized_raw[len("/workspace/media/"):])
            candidates.extend(
                [
                    PROJECT_DIR / media_relative,
                    SERVICE_DIR / media_relative,
                    CWD_DIR / media_relative,
                ]
            )
        candidates.extend(
            [
                SERVICE_DIR / workspace_relative,
                PROJECT_DIR / workspace_relative,
                CWD_DIR / workspace_relative,
            ]
        )

    if path.is_absolute():
        candidates.append(path)
        try:
            workspace_relative = path.relative_to("/workspace")
        except ValueError:
            workspace_relative = None
        if workspace_relative is not None:
            candidates.extend(
                [
                    SERVICE_DIR / workspace_relative,
                    PROJECT_DIR / workspace_relative,
                    CWD_DIR / workspace_relative,
                ]
            )
    else:
        candidates.extend(
            [
                SERVICE_DIR / path,
                PROJECT_DIR / path,
                CWD_DIR / path,
            ]
        )

    seen = set()
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate in seen:
            continue
        seen.add(candidate)
        if candidate.exists():
            return str(candidate)

    return raw


def _unique_model_paths(paths):
    cleaned = []
    duplicates = []
    seen = set()
    for raw in paths or []:
        path = str(raw).strip()
        if not path:
            continue
        if path in seen:
            duplicates.append(path)
            continue
        seen.add(path)
        cleaned.append(path)
    return cleaned, duplicates


def _load_model_specs():
    default_batch_size = max(1, _as_int(get_conf("models", "batch_size", 4), 4))
    raw_items = get_conf("models", "items", [])
    entries = []

    if raw_items:
        for idx, item in enumerate(raw_items):
            if isinstance(item, str):
                raw_path = item
                model_id = Path(item).stem
                batch_size = default_batch_size
            elif isinstance(item, dict):
                raw_path = item.get("path", "")
                model_id = str(item.get("id", "")).strip() or Path(str(raw_path)).stem
                batch_size = max(1, _as_int(item.get("batch_size", default_batch_size), default_batch_size))
            else:
                continue

            resolved_path = resolve_path(raw_path)
            if not resolved_path:
                continue

            entries.append(
                {
                    "id": model_id,
                    "path": resolved_path,
                    "batch_size": batch_size,
                }
            )
    else:
        raw_model_paths = get_conf("models", "paths", [])
        if not raw_model_paths:
            models_env = os.environ.get("MODELS", "")
            if models_env:
                raw_model_paths = [p.strip() for p in models_env.split(",") if p.strip()]

        for path in raw_model_paths:
            resolved_path = resolve_path(path)
            if not resolved_path:
                continue
            entries.append(
                {
                    "id": Path(resolved_path).stem,
                    "path": resolved_path,
                    "batch_size": default_batch_size,
                }
            )

    unique_entries = []
    duplicate_entries = []
    seen_paths = set()
    for entry in entries:
        path = entry["path"]
        if path in seen_paths:
            duplicate_entries.append(path)
            continue
        seen_paths.add(path)
        unique_entries.append(entry)
    return unique_entries, duplicate_entries


MODEL_SPECS, DUPLICATE_MODEL_PATHS = _load_model_specs()
MODEL_PATHS = [spec["path"] for spec in MODEL_SPECS]

FONT_PATH = resolve_path("fonts/Alimama_ShuHeiTi_Bold.ttf")


def _detect_cuda_runtime():
    if FORCE_CPU:
        print("[Config] 检测到 FORCE_CPU=1，已跳过 CUDA 探测并强制使用 CPU 模式")
        return False, 0, 0

    raw_gpu_count = 0
    try:
        raw_gpu_count = torch.cuda.device_count()
    except Exception as exc:
        print(f"[Config] 读取 CUDA 设备数量失败，将回退 CPU: {exc}")
        return False, 0, 0

    try:
        cuda_available = bool(torch.cuda.is_available())
    except Exception as exc:
        print(f"[Config] 检测 CUDA 可用性失败，将回退 CPU: {exc}")
        return False, 0, raw_gpu_count

    if not cuda_available:
        return False, 0, raw_gpu_count
    return True, raw_gpu_count, raw_gpu_count


CUDA_AVAILABLE, GPU_COUNT, RAW_GPU_COUNT = _detect_cuda_runtime()

SERVER_HOST = _get_env_or_conf("SERVER_HOST", "server", "host", "0.0.0.0")
SERVER_PORT = _as_int(_get_env_or_conf("SERVER_PORT", "server", "port", 8008), 8008)
PUBLIC_HOST = _get_env_or_conf("BASE_HOST", "server", "base_host", "127.0.0.1")
PUBLIC_API_BASE = _normalize_base_url(
    _get_env_or_conf("PUBLIC_API_BASE", "server", "public_api_base", f"http://{PUBLIC_HOST}:{SERVER_PORT}")
)
PUBLIC_WS_BASE = _normalize_base_url(
    _get_env_or_conf("PUBLIC_WS_BASE", "server", "public_ws_base", f"ws://{PUBLIC_HOST}:{SERVER_PORT}")
)
ZLM_HTTP_PORT = _as_int(_get_env_or_conf("ZLM_HTTP_PORT", "server", "zlm_http_port", 18080), 18080)
PUBLIC_ZLM_BASE = _normalize_base_url(
    _get_env_or_conf("PUBLIC_ZLM_BASE", "server", "public_zlm_base", f"http://{PUBLIC_HOST}:{ZLM_HTTP_PORT}")
)
INTERNAL_RTMP_HOST = _get_env_or_conf("INTERNAL_RTMP_HOST", "server", "internal_rtmp_host", "127.0.0.1")
RTMP_PORT = _as_int(_get_env_or_conf("RTMP_PORT", "server", "rtmp_port", 1935), 1935)
RTSP_PORT = _as_int(_get_env_or_conf("RTSP_PORT", "server", "rtsp_port", 554), 554)
UVICORN_RELOAD = _as_bool(_get_env_or_conf("UVICORN_RELOAD", "server", "reload", False), False)

# 全局目标输出帧率。若源帧率更低，则自动退回源帧率并全帧处理。
OUTPUT_FPS = max(1.0, _as_float(get_conf("server", "output_fps", 15), 15.0))
# 每张 GPU 最大允许的同时推流视频路数
MAX_STREAMS_PER_GPU = max(1, _as_int(get_conf("server", "max_streams_per_gpu", 6), 6))

# ── 推理精度与速度控制 ──
# 默认推理输入分辨率 (无人机场景推荐 640 + P2 头)
MODEL_IMGSZ = _as_int(get_conf("models", "imgsz", 640), 640)
# FP16 半精度推理：GPU 上可提升约 2× 速度，精度损失对小目标几乎可忽略
MODEL_HALF = _as_bool(get_conf("models", "half", True), True)
# NMS-Free 模式 (YOLO26+ 专用)：端到端直出，跳过后处理 NMS
NMS_FREE = _as_bool(get_conf("models", "nms_free", False), False)

# ── SAHI 切片推理 (高空无人机 4K 画面专用) ──
# 将大图切成 slice_size 的小块逐块检测再合并；FPS 会降到 5-10
SAHI_ENABLED = _as_bool(get_conf("models", "sahi_enabled", False), False)
SAHI_SLICE_SIZE = _as_int(get_conf("models", "sahi_slice_size", 640), 640)
SAHI_OVERLAP_RATIO = _as_float(get_conf("models", "sahi_overlap_ratio", 0.25), 0.25)

# 重编译 TensorRT Engine 相关配置参数
USE_TENSORRT = _as_bool(get_conf("models", "use_tensorrt", False), False)
ENGINE_CACHE_DIR = str(SERVICE_DIR / "engine_cache")
if USE_TENSORRT:
    os.makedirs(ENGINE_CACHE_DIR, exist_ok=True)

CORS_ALLOW_ORIGINS = _as_origins(_get_env_or_conf("CORS_ALLOW_ORIGINS", "server", "cors_allow_origins", ["*"]))
CORS_ALLOW_CREDENTIALS = "*" not in CORS_ALLOW_ORIGINS

ARTIFACTS_DIR = Path(
    resolve_path(_get_env_or_conf("ARTIFACTS_DIR", "server", "artifacts_dir", str(SERVICE_DIR / "artifacts")))
)
ARTIFACTS_URL_PREFIX = _normalize_url_path(
    _get_env_or_conf("ARTIFACTS_URL_PREFIX", "server", "artifacts_url_prefix", "/artifacts"),
    "/artifacts",
)
SNAPSHOT_DIR = ARTIFACTS_DIR / "snapshots"
PROCESSED_DIR = ARTIFACTS_DIR / "processed"
INPUTS_DIR = ARTIFACTS_DIR / "inputs"
for _dir in (ARTIFACTS_DIR, SNAPSHOT_DIR, PROCESSED_DIR, INPUTS_DIR):
    _dir.mkdir(parents=True, exist_ok=True)

# MinIO 图片抓拍归档配置
MINIO_ENABLED = _as_bool(_get_env_or_conf("MINIO_ENABLED", "minio", "enabled", True), True)
MINIO_ENDPOINT = _get_env_or_conf("MINIO_ENDPOINT", "minio", "endpoint", "127.0.0.1:9000")
MINIO_PUBLIC_BASE = _normalize_base_url(
    _get_env_or_conf("MINIO_PUBLIC_BASE", "minio", "public_base", f"http://{PUBLIC_HOST}:9000")
)
MINIO_ACCESS_KEY = _get_env_or_conf("MINIO_ACCESS_KEY", "minio", "access_key", "minioadmin")
MINIO_SECRET_KEY = _get_env_or_conf("MINIO_SECRET_KEY", "minio", "secret_key", "minioadmin")
MINIO_BUCKET = _get_env_or_conf("MINIO_BUCKET", "minio", "bucket", "yolo-snapshots")

print(f"[Config] 载入系统架构，共识别 {len(MODEL_PATHS)} 个独立模型: {MODEL_PATHS}")
print(
    f"[Config] Torch={TORCH_VERSION}, Torch CUDA={TORCH_CUDA_VERSION}, "
    f"CUDA_AVAILABLE={CUDA_AVAILABLE}, RAW_GPU_COUNT={RAW_GPU_COUNT}, FORCE_CPU={FORCE_CPU}"
)
print(
    f"[Config] UAV 速度档案: imgsz={MODEL_IMGSZ}, half(FP16)={MODEL_HALF}, "
    f"TensorRT={USE_TENSORRT}, NMS_FREE={NMS_FREE}, "
    f"SAHI={SAHI_ENABLED}(slice={SAHI_SLICE_SIZE}, overlap={SAHI_OVERLAP_RATIO})"
)
if DUPLICATE_MODEL_PATHS:
    print(f"[Config] 已忽略重复模型路径: {DUPLICATE_MODEL_PATHS}")
if not CUDA_AVAILABLE and RAW_GPU_COUNT > 0:
    print(
        "[Config] 检测到宿主机存在 GPU，但当前 PyTorch/CUDA 运行时不可用，"
        "服务将自动回退到 CPU 模式。请检查 NVIDIA 驱动版本是否与容器内 torch 匹配。"
    )
