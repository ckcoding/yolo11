import os
import logging
import torch
import numpy as np
from ultralytics import YOLO
from config import config

logger = logging.getLogger(__name__)

class ModelManager:
    """集中式的底层全局模型管理器。负责在后台系统启动时，一次性把指定文件夹下所有的权重拉入常驻内存或者显存。
    支持多GPU环境：对每张可用的GPU卡和CPU均创建独立的模型实例并预热。"""
    def __init__(self):
        self.weights_dir = config.weights_dir
        os.makedirs(self.weights_dir, exist_ok=True)
        # 内部缓存字典构造: {"drone_person.onnx": {"cuda:0": <YOLO>, "cuda:1": <YOLO>, "cpu": <YOLO>}}
        self.models = {}
        self.available_models = []

    def preload_all(self):
        logger.info(f"系统启动：开始尝试扫描并预热 {self.weights_dir} 目录下的核心骨干模型（ONNX/PT）...")
        # 支持多种常见权重后缀
        self.available_models = [f for f in os.listdir(self.weights_dir) if f.endswith(".onnx") or f.endswith(".pt")]
        
        if not self.available_models:
            logger.warning(f"目录 {self.weights_dir} 中未发现任何权重文件！系统将处于停飞状态。")
            return
        
        has_gpu = torch.cuda.is_available()
        gpu_count = torch.cuda.device_count() if has_gpu else 0
        
        for model_file in self.available_models:
            path = os.path.join(self.weights_dir, model_file)
            logger.info(f"==> 正在将本地模型 [{model_file}] 多设备编纂载入极速内存池区...")
            self.models[model_file] = {}
            
            try:
                # 显式移除 task='detect'，由 YOLO 自行推导（可以是 detect/pose/segment）
                cpu_model = YOLO(path)
                dummy_img_cpu = np.zeros((320, 320, 3), dtype=np.uint8)
                cpu_model.predict(dummy_img_cpu, device="cpu", verbose=False, imgsz=320)
                self.models[model_file]["cpu"] = cpu_model
                logger.info(f"    [{model_file}] CPU 暖机完毕。")
                
                for gpu_idx in range(gpu_count):
                    device_str = f"cuda:{gpu_idx}"
                    device_obj = torch.device(device_str)
                    try:
                        gpu_model = YOLO(path)
                        dummy_img_gpu = np.zeros((640, 640, 3), dtype=np.uint8)
                        gpu_model.predict(dummy_img_gpu, device=device_obj, verbose=False, imgsz=640)
                        self.models[model_file][device_str] = gpu_model
                        logger.info(f"    [{model_file}] {device_str} ({torch.cuda.get_device_name(gpu_idx)}) 暖机完毕。")
                    except Exception as e:
                        logger.error(f"    [{model_file}] 在 {device_str} 上预热失败: {e}，该卡将不可用于此模型。")
                
                logger.info(f"    [{model_file}] 全部设备挂载暖机完成，热条用准备就绪！")
            except Exception as e:
                logger.error(f"模型 {model_file} 并行预热挂载遭遇重挫或模型破损: {e}")


    def get_model(self, model_name: str, device_str: str):
        """接口或后台流发生时被瞬间调用抓取热机。
        device_str 应为 'cuda:0', 'cuda:1', 'cpu' 等具体设备字符串。"""
        target_dict = self.models.get(model_name)
        if not target_dict:
            raise ValueError(f"系统报错：客户端请求的模型 [{model_name}] 根本不在热区就绪名单中（可能文件名打错了，或尚未部署）！")
        
        # 精确匹配优先
        model = target_dict.get(device_str)
        if model:
            return model
        
        # 降级容错：如果指定GPU上的模型不存在，尝试找其他GPU上的，最后回退到CPU
        logger.warning(f"模型 [{model_name}] 在设备 {device_str} 上未找到预热实例，尝试降级...")
        
        # 尝试其他GPU
        for key, m in target_dict.items():
            if key.startswith("cuda"):
                logger.warning(f"  降级使用 {key} 上的实例。")
                return m
        
        # 最终回退到CPU
        cpu_model = target_dict.get("cpu")
        if cpu_model:
            logger.warning(f"  最终降级至 CPU 实例。")
            return cpu_model
        
        raise ValueError(f"系统报错：模型 [{model_name}] 在所有设备上均无可用实例！")

model_manager = ModelManager()
