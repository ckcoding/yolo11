import yaml
import os

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.yaml")

class Config:
    def __init__(self, config_file=CONFIG_PATH):
        # 读取并解析 YAML 配置文件
        with open(config_file, "r", encoding="utf-8") as f:
            self._config = yaml.safe_load(f)

    @property
    def weights_dir(self):
        # 默认去寻找同级目录下的 weights 文件夹
        return self._config.get("model", {}).get("weights_dir", os.path.join(os.path.dirname(__file__), "weights"))
        
    @property
    def model_weights(self):
        # 兼容老版默认参数兜底
        return self._config.get("model", {}).get("weights", "yolo11n.onnx")

    @property
    def fonts_dir(self):
        return self._config.get("fonts", {}).get("dir", "fonts")

    @property
    def default_font(self):
        return self._config.get("fonts", {}).get("default", "Arial.ttf")

    @property
    def minio_endpoint(self):
        return self._config.get("minio", {}).get("endpoint", "127.0.0.1:9000")

    @property
    def minio_access_key(self):
        return self._config.get("minio", {}).get("access_key", "minioadmin")

    @property
    def minio_secret_key(self):
        return self._config.get("minio", {}).get("secret_key", "minioadmin")

    @property
    def minio_secure(self):
        return self._config.get("minio", {}).get("secure", False)

    @property
    def minio_bucket(self):
        return self._config.get("minio", {}).get("bucket_name", "detections")

    @property
    def class_chinese(self):
        # 动态从 YAML 中提取翻译映射，如果没有则返回空字典
        return self._config.get("model", {}).get("class_names", {})

    @property
    def class_colors(self):
        # 动态从 YAML 中提取类别颜色映射，如果没有则返回空字典
        return self._config.get("model", {}).get("class_colors", {})

    @property
    def max_gpu_streams(self):
        # 默认返回 0 代表允许后续移交代码对物理硬件算力容量进行自适应测算
        return self._config.get("hardware", {}).get("max_gpu_streams", 0)

    @property
    def vram_per_stream(self):
        # 每路检测任务预估需要的显存量(GB)，用于动态分配时判断某张卡是否还有足够余量
        return self._config.get("hardware", {}).get("vram_per_stream", 1.5)

# 全局配置实例
config = Config()

# 为了保持 detector.py 等处的兼容性，这里做一个引用映射
class_chinese_legacy = config.class_chinese
# 注意：原先直接写在 py 里的字典被迁移到了 yaml，
# 后续代码如果直接引用 CLASS_CHINESE，请确保它们使用的是 config.class_chinese。
CLASS_CHINESE = config.class_chinese
