import gc
import queue
from pathlib import Path

import torch

from openmmlab_service.config import CUDA_AVAILABLE, GPU_COUNT, MAX_STREAMS_PER_GPU, MODEL_SPECS
from openmmlab_service.core.batch_infer import ClusterBatchService, MultiModelBatchService, SingleModelBatchWorker


class ModelParkingLot:
    """
    OpenMMLab 多模型停车场。
    每张 GPU 上为每个模型加载一份共享实例，并为每个模型创建独立批量 Worker。
    """

    def __init__(self):
        self.pool = queue.Queue()
        self.total_capacity = 0
        self._version = 0
        self._services = []
        self._catalog = []
        self._known_labels = set()

    def init_pool(self):
        print("[ModelPool] 系统就绪，采用 OpenMMLab 多模型 Worker + Batch 推理引擎")

        if not MODEL_SPECS:
            raise RuntimeError("未配置任何 OpenMMLab 模型，无法启动推理服务")

        missing_refs = []
        for spec in MODEL_SPECS:
            if not Path(spec["config"]).exists():
                missing_refs.append({"id": spec["id"], "field": "config", "path": spec["config"]})
            if not Path(spec["checkpoint"]).exists():
                missing_refs.append({"id": spec["id"], "field": "checkpoint", "path": spec["checkpoint"]})
        if missing_refs:
            raise FileNotFoundError(f"以下模型文件不存在: {missing_refs}")

        if not CUDA_AVAILABLE or GPU_COUNT == 0:
            print("[ModelPool] 检测不到 GPU，将使用 CPU 模式...")
            service = self._build_device_service("cpu")
            self._services.append(service)
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
                free_gb = free_mem / (1024**3)
                total_gb = total_mem / (1024**3)
                print(f"[ModelPool] GPU {gpu_idx} 显存: 空闲 {free_gb:.1f}GB / 总计 {total_gb:.1f}GB")
            except Exception as exc:
                print(f"[ModelPool] 获取 GPU {gpu_idx} 显存信息失败: {exc}")

            service = self._build_device_service(device)
            self._services.append(service)
            if not self._catalog:
                self._catalog = service.describe_models()
                self._known_labels = service.supported_labels()

        main_service = ClusterBatchService(self._services) if len(self._services) > 1 else self._services[0]

        total_dynamic_capacity = 0
        for gpu_idx in range(GPU_COUNT):
            try:
                free_mem, _ = torch.cuda.mem_get_info(gpu_idx)
                gpu_capacity = max(2, int((free_mem * 0.85) / (300 * 1024 * 1024)))
                total_dynamic_capacity += gpu_capacity
                print(f"[ModelPool] GPU {gpu_idx} 剩余可用显存预估可安全承载 {gpu_capacity} 路推理任务")
            except Exception as exc:
                print(f"[ModelPool] 动态容量估算失败，回落默认限额: {exc}")
                total_dynamic_capacity += MAX_STREAMS_PER_GPU

        if total_dynamic_capacity <= 0:
            total_dynamic_capacity = MAX_STREAMS_PER_GPU * max(1, GPU_COUNT)

        for _ in range(total_dynamic_capacity):
            self.pool.put({"service": main_service, "device": main_service.device, "_pool_version": self._version})
            self.total_capacity += 1

        print(
            f"[ModelPool] OpenMMLab 集群池初始化完成，总计 {self.total_capacity} 路并发容量就绪 "
            f"(负载均衡分布于 {len(self._services)} 张显卡)"
        )
        print(f"[ModelPool] 当前可识别标签: {sorted(self._known_labels)}")

    def _build_device_service(self, device):
        workers = []
        for spec in MODEL_SPECS:
            print(
                f"[ModelPool] 加载模型 {spec['id']} 至 {device}: "
                f"config={spec['config']} checkpoint={spec['checkpoint']}"
            )
            worker = SingleModelBatchWorker(
                model_spec=spec,
                device=device,
                batch_size=spec["batch_size"],
            )
            worker.warmup()
            workers.append(worker)
        return MultiModelBatchService(workers, device)

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
        if not self._services:
            return None
        if len(self._services) > 1:
            return ClusterBatchService(self._services)
        return self._services[0]


model_pool = ModelParkingLot()
