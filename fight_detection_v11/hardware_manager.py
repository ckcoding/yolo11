import threading
import logging
import torch
import os

logger = logging.getLogger(__name__)

class HardwareManager:
    """全局物理资源集散管理器，支持多GPU环境下根据实时显存剩余量动态分配最优设备。"""
    def __init__(self, override_max_gpu=0, vram_per_stream=1.5):
        self.lock = threading.Lock()
        self.vram_per_stream = vram_per_stream  # 每路任务预估所需显存(GB)
        
        # 【全自动检测一：探明所有 GPU 显卡信息】
        self.has_gpu = torch.cuda.is_available()
        self.gpu_count = torch.cuda.device_count() if self.has_gpu else 0
        self.gpu_info = []  # 列表，每项: {"index": 0, "name": "...", "total_vram_gb": ...}
        
        # 每张卡当前被分配的任务数（用于释放时追踪）
        self.gpu_task_count = {}  # {"cuda:0": 0, "cuda:1": 0, ...}
        
        # 所有GPU合计最大可承载的流数上限
        self.max_gpu = 0
        
        # 【全自动检测二：探明 CPU 逻辑核心数量池并拆分路数】
        self.current_cpu = 0
        self.max_cpu = max(1, os.cpu_count() // 3)
        
        if self.has_gpu:
            for i in range(self.gpu_count):
                try:
                    props = torch.cuda.get_device_properties(i)
                    vram_gb = props.total_memory / (1024**3)
                    gpu_name = torch.cuda.get_device_name(i)
                    per_gpu_max = max(1, int(vram_gb // self.vram_per_stream))
                    
                    self.gpu_info.append({
                        "index": i,
                        "name": gpu_name,
                        "total_vram_gb": round(vram_gb, 1),
                        "max_streams": per_gpu_max,
                    })
                    device_str = f"cuda:{i}"
                    self.gpu_task_count[device_str] = 0
                    self.max_gpu += per_gpu_max
                    
                    logger.info(
                        f"系统智能探明器：检测到 GPU #{i} [{gpu_name}]，总显存约 {vram_gb:.1f} GB，"
                        f"单卡安全并行流上限 {per_gpu_max} 路。"
                    )
                except Exception as e:
                    logger.warning(f"获取 GPU #{i} 信息失败: {e}，跳过此卡。")
            
            if not self.gpu_info:
                logger.warning("所有 GPU 探测均失败，系统将退化为纯 CPU 模式。")
                self.has_gpu = False
        else:
            logger.warning(
                "系统智能探明器：宿主机不存在任何可用的 CUDA 显卡调度环境，"
                "因此所有流入的新任务将被直接遣送至底层纯 CPU 进行接管。"
            )
        
        logger.info(
            f"同时探明主板搭载了 {os.cpu_count()} 颗空闲逻辑核心，"
            f"推算可兜底承载并行额外上限警戒阀值为 {self.max_cpu} 路纯 CPU 检测。"
        )
        
        # 依然向用户保留人工越权调度的通道
        if override_max_gpu and override_max_gpu > 0:
            logger.info(
                f"（但由于您已在 config.yaml 强制硬编码指定了显卡分配流水上限为 {override_max_gpu} 路，"
                f"系统将听从配置，抛弃自动算力评估，以此硬设定为准！）"
            )
            self.max_gpu = override_max_gpu

    def _get_best_gpu(self) -> str | None:
        """优先选择当前任务数最少的GPU（负载均衡），任务数相同时选剩余显存最大的。
        返回 'cuda:N' 或 None（全部显存不足时）。"""
        candidates = []  # [(device_str, current_tasks, free_gb)]
        
        for info in self.gpu_info:
            idx = info["index"]
            device_str = f"cuda:{idx}"
            current_tasks = self.gpu_task_count.get(device_str, 0)
            
            # 检查该卡是否已达到最大流数上限
            if current_tasks >= info["max_streams"]:
                continue
            
            try:
                # torch.cuda.mem_get_info 返回 (free, total)，单位 bytes
                free_bytes, total_bytes = torch.cuda.mem_get_info(idx)
                free_gb = free_bytes / (1024**3)
                
                # 至少要有 vram_per_stream GB 的剩余显存才认为可以接收新任务
                if free_gb >= self.vram_per_stream:
                    candidates.append((device_str, current_tasks, free_gb))
            except Exception as e:
                logger.warning(f"查询 GPU #{idx} 实时显存失败: {e}，跳过。")
                continue
        
        if not candidates:
            return None
        
        # 排序策略：优先任务数最少（负载均衡），任务数相同时选剩余显存最大的
        candidates.sort(key=lambda x: (x[1], -x[2]))
        best_device, best_tasks, best_free_vram = candidates[0]
        
        logger.info(
            f"动态显存调度：选中 {best_device}（当前 {best_tasks} 路任务，剩余显存 {best_free_vram:.1f} GB），"
            f"候选池共 {len(candidates)} 张卡可用。"
        )
        
        return best_device

    def allocate_device(self) -> str:
        """接口调用发生时的核心动态流调度室：优先塞给显存最多的GPU，全部满了塞CPU。"""
        with self.lock:
            if self.has_gpu:
                best_gpu = self._get_best_gpu()
                if best_gpu:
                    self.gpu_task_count[best_gpu] = self.gpu_task_count.get(best_gpu, 0) + 1
                    return best_gpu
                else:
                    logger.warning(
                        "所有 GPU 显存余量不足或已达最大流数上限，本次任务将降级至 CPU 运行。"
                    )
            
            # GPU 不可用或已满载，降级到 CPU
            self.current_cpu += 1
            if self.current_cpu > self.max_cpu:
                logger.error(
                    f"严重警告：您源源不断地送入流的数量 [{self.current_cpu} 路] "
                    f"已经远远超出了系统所能够舒适负载的估算承受极限 [{self.max_cpu} 路]！"
                    f"系统极可能会发生雪崩式的全页面发热极慢卡死或崩溃！"
                )
            return "cpu"

    def release_device(self, device: str):
        """当一个任务结束时，系统会唤回被借走的算力编制配额放回总池子中。"""
        with self.lock:
            if device.startswith("cuda"):
                self.gpu_task_count[device] = max(0, self.gpu_task_count.get(device, 1) - 1)
                logger.info(f"释放设备 {device}，该卡剩余任务数: {self.gpu_task_count[device]}")
            else:
                self.current_cpu = max(0, self.current_cpu - 1)

    def get_gpu_usage(self) -> dict:
        """返回当前所有GPU的使用状态信息。"""
        with self.lock:
            status = {}
            for info in self.gpu_info:
                device_str = f"cuda:{info['index']}"
                current = self.gpu_task_count.get(device_str, 0)
                try:
                    free_bytes, total_bytes = torch.cuda.mem_get_info(info["index"])
                    free_gb = round(free_bytes / (1024**3), 1)
                except Exception:
                    free_gb = -1
                status[device_str] = {
                    "name": info["name"],
                    "total_vram_gb": info["total_vram_gb"],
                    "free_vram_gb": free_gb,
                    "current_tasks": current,
                    "max_streams": info["max_streams"],
                }
            return status
