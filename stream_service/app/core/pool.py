import gc
import queue
import threading
from pathlib import Path

import numpy as np
import torch
from ultralytics import YOLO

from app.config import (
    CUDA_AVAILABLE, ENGINE_CACHE_DIR, GPU_COUNT, MAX_STREAMS_PER_GPU,
    MODEL_HALF, MODEL_IMGSZ, MODEL_SPECS, NMS_FREE, USE_TENSORRT,
)
from app.core.batch_infer import MultiModelBatchService, SingleModelBatchWorker


class ModelParkingLot:
    """
    多模型停车场。
    每张 GPU 上为每个模型加载一份共享实例，并为每个模型创建独立批量 Worker。
    视频流只领取流控令牌，不再独占整套模型。
    """

    def __init__(self):
        self.pool = queue.Queue()
        self.total_capacity = 0
        self._version = 0
        self._services = []
        self._main_service = None
        self._catalog = []
        self._known_labels = set()

    def init_pool(self):
        print("[ModelPool] 系统就绪，采用 多模型独立Worker + Batch 批量推理 引擎")

        if not MODEL_SPECS:
            raise RuntimeError("未配置任何可用模型路径，无法启动推理服务")

        missing_paths = [spec["path"] for spec in MODEL_SPECS if not Path(spec["path"]).exists()]
        if missing_paths:
            raise FileNotFoundError(f"以下模型文件不存在: {missing_paths}")

        if not CUDA_AVAILABLE or GPU_COUNT == 0:
            print("[ModelPool] 检测不到 GPU，将使用 CPU 模式...")
            service = self._build_device_service("cpu")
            self._services.append(service)
            self._main_service = service
            self._catalog = service.describe_models()
            self._known_labels = service.supported_labels()

            fallback_cap = max(2, MAX_STREAMS_PER_GPU)
            for _ in range(fallback_cap):
                self.pool.put({"service": service, "device": "cpu", "_pool_version": self._version})
                self.total_capacity += 1
            print(f"[ModelPool] CPU 模式下采用通用后备容量，共开放 {fallback_cap} 路并发车位")
            return

        for gpu_idx in range(GPU_COUNT):
            device = f"cuda:{gpu_idx}"
            try:
                free_mem, total_mem = torch.cuda.mem_get_info(gpu_idx)
                free_gb = free_mem / (1024 ** 3)
                total_gb = total_mem / (1024 ** 3)
                print(f"[ModelPool] GPU {gpu_idx} 显存: 空闲 {free_gb:.1f}GB / 总计 {total_gb:.1f}GB")
            except Exception as exc:
                print(f"[ModelPool] 获取 GPU {gpu_idx} 显存信息失败: {exc}")

            service = self._build_device_service(device)
            self._services.append(service)
            if not self._catalog:
                self._catalog = service.describe_models()
                self._known_labels = service.supported_labels()

        from app.core.batch_infer import ClusterBatchService
        if len(self._services) > 1:
            main_service = ClusterBatchService(self._services)
        else:
            main_service = self._services[0]
        self._main_service = main_service

        total_dynamic_capacity = 0
        for gpu_idx in range(GPU_COUNT):
            try:
                free_mem, _ = torch.cuda.mem_get_info(gpu_idx)
                # 每路视频推断上下文缓存平均保守估算占用 200MB (0.2GB)
                gpu_capacity = max(2, int((free_mem * 0.9) / (200 * 1024 * 1024)))
                total_dynamic_capacity += gpu_capacity
                print(f"[ModelPool] 资源伸缩: GPU {gpu_idx} 剩余可用显存预估可安全承载 {gpu_capacity} 层推流兵力")
            except Exception as e:
                print(f"[ModelPool] 动态寻址异常，回落默认限额: {e}")
                total_dynamic_capacity += MAX_STREAMS_PER_GPU

        if total_dynamic_capacity <= 0:
            total_dynamic_capacity = MAX_STREAMS_PER_GPU * max(1, GPU_COUNT)

        for i in range(total_dynamic_capacity):
            self.pool.put({"service": main_service, "device": main_service.device, "_pool_version": self._version})
            self.total_capacity += 1

        print(
            f"[ModelPool] 多模型集群池初始化完成，总计 {self.total_capacity} 路并发容量就绪 "
            f"(集群分发模式，负载均衡分布于 {len(self._services)} 张显卡)"
        )
        _speed_profile = []
        if MODEL_HALF:
            _speed_profile.append("FP16")
        if USE_TENSORRT:
            _speed_profile.append("TensorRT")
        if NMS_FREE:
            _speed_profile.append("NMS-Free")
        speed_tag = "+".join(_speed_profile) if _speed_profile else "FP32标准"
        print(f"[ModelPool] 速度档位: {speed_tag} | imgsz={MODEL_IMGSZ}")
        print(f"[ModelPool] 当前可识别标签: {sorted(self._known_labels)}")

    def _build_device_service(self, device):
        workers = []
        for spec in MODEL_SPECS:
            print(f"[ModelPool] 加载模型 {spec['id']} 至 {device}: {spec['path']}")

            # 默认加载 PyTorch 原生模型
            active_model_path = spec["path"]
            
            # TensorRT 自动导出：首次加载时构建 .engine 并缓存
            if USE_TENSORRT and str(device) != "cpu":
                import shutil
                # 将 imgsz 和 batch 都编码进文件名，防止配置变更导致引擎不匹配
                # 增加了 _static 强行作废之前错误生成的动态引擎
                engine_filename = f"{spec['id']}_imgsz{MODEL_IMGSZ}_b{spec['batch_size']}_static.engine"
                engine_path = Path(ENGINE_CACHE_DIR) / engine_filename
                
                if not engine_path.exists():
                    print(f"[ModelPool] → 正在导出 TensorRT 引擎 (首次可能需要几分钟)...")
                    try:
                        # 解决源目录 Read-Only 问题：拷贝到有写权限的目录进行转换
                        temp_pt = Path(ENGINE_CACHE_DIR) / f"{spec['id']}_temp.pt"
                        shutil.copy2(spec["path"], temp_pt)
                        
                        temp_model = YOLO(str(temp_pt))
                        exported_path = temp_model.export(
                            format="engine",
                            half=MODEL_HALF,
                            imgsz=MODEL_IMGSZ,
                            batch=spec["batch_size"],  # 必须指定批量大小，禁止默认的单张
                            dynamic=False,             # 彻底禁掉动态批次，使用纯粹的静态图，绕过 cuTensor Bug
                            device=device,
                            nms=NMS_FREE,
                        )
                        
                        if exported_path and Path(exported_path).exists():
                            shutil.move(exported_path, engine_path)
                            print(f"[ModelPool] ✓ TensorRT 引擎导出完成: {engine_path}")
                            active_model_path = str(engine_path)
                        else:
                            print(f"[ModelPool] ✗ TensorRT 导出未返回有效路径，回退 PyTorch")
                            
                        # 清理临时打底文件 (.pt, .onnx)
                        for ext in [".pt", ".onnx"]:
                            cache_file = Path(ENGINE_CACHE_DIR) / f"{spec['id']}_temp{ext}"
                            if cache_file.exists():
                                cache_file.unlink()
                                
                    except Exception as exc:
                        print(f"[ModelPool] ✗ TensorRT 导出彻底失败，回退原始模型: {exc}")
                else:
                    print(f"[ModelPool] → 成功复用已缓存 TensorRT 引擎: {engine_path}")
                    active_model_path = str(engine_path)

            # 加载最终模型 (若是 TensorRT 则为 .engine, 否则为 .pt)
            model = YOLO(active_model_path)
            
            self._warmup(model, device, spec["batch_size"])
            worker = SingleModelBatchWorker(
                model_id=spec["id"],
                model=model,
                model_path=spec["path"],
                device=device,
                batch_size=spec["batch_size"],
                imgsz=MODEL_IMGSZ,
                half=MODEL_HALF,
            )
            workers.append(worker)
        return MultiModelBatchService(workers, device)

    def _warmup(self, model, device, batch_size=1):
        dummy_img = np.zeros((MODEL_IMGSZ, MODEL_IMGSZ, 3), dtype=np.uint8)
        dummy_imgs = [dummy_img for _ in range(batch_size)]
        half = MODEL_HALF and str(device) != "cpu"
        model.predict(source=dummy_imgs, device=device, half=half, imgsz=MODEL_IMGSZ, verbose=False)

    def describe_models(self):
        return list(self._catalog)

    def supported_labels(self):
        return set(self._known_labels)

    def resolve_model_ids(self, target_set):
        if not self._services:
            return []
        return self._services[0].resolve_model_ids(target_set)

    def find_unknown_labels(self, target_set):
        return sorted(set(target_set or ()) - self._known_labels)

    def reload(self):
        self._version += 1
        print(f"[ModelPool] 收到重载指令 (v{self._version})，正在清理...")
        while not self.pool.empty():
            try:
                self.pool.get_nowait()
            except queue.Empty:
                break

        for svc in self._services:
            svc.stop()
        self._services.clear()
        self._catalog = []
        self._known_labels = set()
        self.total_capacity = 0

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print("[ModelPool] 碎片清理完毕，正在重新解析模型并开辟车位...")
        self.init_pool()

    def shutdown(self):
        while not self.pool.empty():
            try:
                self.pool.get_nowait()
            except queue.Empty:
                break

        for svc in self._services:
            svc.stop()
        self._services.clear()
        self._catalog = []
        self._known_labels = set()
        self.total_capacity = 0

    def acquire(self, timeout=None):
        try:
            return self.pool.get(timeout=timeout)
        except queue.Empty:
            return None

    def release(self, model_obj):
        if model_obj.get("_pool_version", -1) < self._version:
            print(
                f"[ModelPool] 检测到过期令牌 "
                f"(v{model_obj.get('_pool_version', '?')} < v{self._version})，已丢弃"
            )
            return
        self.pool.put(model_obj)

    def get_main_service(self):
        return self._main_service

model_pool = ModelParkingLot()
