import queue
import threading
import time
from collections import defaultdict


class SingleModelBatchWorker:
    """
    单模型批量推理 Worker。
    每个 Worker 只托管一个模型，但会跨多路视频请求攒批后统一推理。
    """

    def __init__(self, model_id, model, model_path, device, batch_size=4, max_wait_ms=5, imgsz=640, half=True):
        self.model_id = model_id
        self.model = model
        self.model_path = model_path
        self.device = device
        self.batch_size = batch_size
        self.max_wait_sec = max_wait_ms / 1000.0
        self.imgsz = imgsz
        self.half = half and str(device) != "cpu"
        self.supported_labels = set(model.names.values())
        self._request_queue = queue.Queue()
        self._running = True
        self._log_counter = 0
        self._batch_counter = 0

        if "cuda" in str(device):
            import torch
            self.stream = torch.cuda.Stream(device=device)
        else:
            self.stream = None
        self._thread = threading.Thread(
            target=self._batch_loop,
            daemon=True,
            name=f"BatchInfer-{device}-{model_id}",
        )
        self._thread.start()

    def submit(self, frame, target_set, conf_thres=0.5):
        event = threading.Event()
        request = {
            "frame": frame,
            "target_set": set(target_set or ()),
            "conf_thres": conf_thres,
            "event": event,
            "result": {},
            "submit_ts": time.time(),
        }
        self._request_queue.put(request)
        return request

    def wait_result(self, request, deadline):
        remaining = deadline - time.time()
        if remaining <= 0 or not request["event"].wait(timeout=remaining):
            raise TimeoutError(f"{self.model_id} 推理超时")
        return dict(request["result"])

    def wait(self, request, deadline):
        return self.wait_result(request, deadline).get("boxes", [])

    def infer_result(self, frame, target_set, conf_thres=0.5, timeout=10.0):
        request = self.submit(frame, target_set, conf_thres)
        deadline = time.time() + timeout
        return self.wait_result(request, deadline)

    def infer(self, frame, target_set, conf_thres=0.5, timeout=10.0):
        return self.infer_result(frame, target_set, conf_thres, timeout).get("boxes", [])

    def describe(self):
        return {
            "model_id": self.model_id,
            "path": self.model_path,
            "device": self.device,
            "batch_size": self.batch_size,
            "labels": sorted(self.supported_labels),
        }

    def _gather_requests(self):
        requests = []
        try:
            requests.append(self._request_queue.get(timeout=1.0))
        except queue.Empty:
            return requests

        # 立即收割队列中已有的请求
        while len(requests) < self.batch_size:
            try:
                requests.append(self._request_queue.get_nowait())
            except queue.Empty:
                break

        # ★ 自适应凑批策略：
        # 若只拿到 1 个请求且队列已空 → 说明当前只有单路流在工作 → 立即处理，0ms 等待！
        # 若已拿到 2+ 请求 → 说明有多路流并发 → 值得再等一小段时间凑更满的 batch
        if len(requests) >= 2 and len(requests) < self.batch_size:
            oldest_submit = min(req["submit_ts"] for req in requests)
            deadline = oldest_submit + self.max_wait_sec
            while len(requests) < self.batch_size:
                remaining = deadline - time.time()
                if remaining <= 0:
                    break
                try:
                    requests.append(self._request_queue.get(timeout=max(0.001, remaining)))
                except queue.Empty:
                    break
        return requests

    def _run_model(self, frames, min_conf):
        if self.stream is not None:
            import torch
            with torch.cuda.stream(self.stream):
                return self.model.predict(
                    source=frames,
                    device=self.device,
                    conf=min_conf,
                    half=self.half,
                    imgsz=self.imgsz,
                    verbose=False,
                )
        else:
            return self.model.predict(
                source=frames,
                device=self.device,
                conf=min_conf,
                half=self.half,
                imgsz=self.imgsz,
                verbose=False,
            )

    def _batch_loop(self):
        print(
            f"[BatchInfer] {self.device}/{self.model_id} Worker 已启动 "
            f"(batch_size={self.batch_size}, labels={sorted(self.supported_labels)})"
        )

        while self._running:
            requests = self._gather_requests()
            if not requests:
                continue

            self._batch_counter += 1
            batch_id = self._batch_counter
            batch_start_ts = time.time()
            
            original_len = len(requests)
            frames = [req["frame"] for req in requests]
            
            # [HOTFIX] 针对 TensorRT 10.x 的 CuTensor 通病：
            # 动态 batch 时底层重排算子会崩溃，我们强制对齐到最大 batch_size
            while len(frames) < self.batch_size:
                frames.append(frames[-1])
                
            min_conf = min(req["conf_thres"] for req in requests)
            per_request_boxes = [[] for _ in requests]
            model_exec_ms = 0.0

            try:
                exec_start_ts = time.time()
                results = self._run_model(frames, min_conf)
                
                # 剔除因为补齐凑 batch 产生的幻觉结果
                results = results[:original_len]
                
                model_exec_ms = (time.time() - exec_start_ts) * 1000.0
                for idx, res in enumerate(results):
                    target_set = requests[idx]["target_set"]
                    filtered_boxes = []
                    for box in res.boxes:
                        cls_id = int(box.cls[0].item())
                        conf_val = float(box.conf[0].item())
                        label_en = self.model.names[cls_id]
                        if label_en in target_set:
                            coords = tuple(box.xyxy[0].cpu().tolist())
                            filtered_boxes.append((coords, label_en, conf_val))
                    per_request_boxes[idx] = filtered_boxes
            except Exception as exc:
                print(f"[BatchInfer] {self.device}/{self.model_id} batch推理异常: {exc}")
            batch_end_ts = time.time()

            for idx, req in enumerate(requests):
                req["result"] = {
                    "boxes": per_request_boxes[idx],
                    "queue_wait_ms": max(0.0, (batch_start_ts - req["submit_ts"]) * 1000.0),
                    "model_exec_ms": max(0.0, model_exec_ms),
                    "model_roundtrip_ms": max(0.0, (batch_end_ts - req["submit_ts"]) * 1000.0),
                    "model_id": self.model_id,
                    "batch_id": batch_id,
                    "batch_size": len(requests),
                }
                req["event"].set()

            self._log_counter += 1
            if self._log_counter % 200 == 0:
                print(
                    f"[BatchInfer] {self.device}/{self.model_id} | "
                    f"本次 batch={len(requests)}/{self.batch_size} | 队列积压={self._request_queue.qsize()}"
                )

    def stop(self):
        self._running = False
        if self._thread.is_alive():
            self._thread.join(timeout=5)


class MultiModelBatchService:
    """
    单设备多模型推理服务。
    负责根据目标标签动态选择需要的模型 Worker，并汇总多模型输出。
    """

    def __init__(self, workers, device):
        self.device = device
        self.workers = {worker.model_id: worker for worker in workers}
        self.label_to_model_ids = defaultdict(list)

        for worker in workers:
            for label in worker.supported_labels:
                self.label_to_model_ids[label].append(worker.model_id)

    def supported_labels(self):
        return set(self.label_to_model_ids.keys())

    def get_model_labels(self, model_id):
        worker = self.workers.get(model_id)
        return set(worker.supported_labels) if worker else set()

    def has_model(self, model_id):
        return model_id in self.workers

    def describe_models(self):
        return [worker.describe() for worker in self.workers.values()]

    def resolve_model_ids(self, target_set):
        model_ids = []
        seen = set()
        for label in sorted(set(target_set or ())):
            for model_id in self.label_to_model_ids.get(label, []):
                if model_id not in seen:
                    seen.add(model_id)
                    model_ids.append(model_id)
        return model_ids

    @staticmethod
    def _empty_perf():
        return {
            "queue_wait_ms": 0.0,
            "model_exec_ms": 0.0,
            "overall_model_stage_ms": 0.0,
        }

    @staticmethod
    def summarize_result_metrics(start_ts, request_results):
        if not request_results:
            return MultiModelBatchService._empty_perf()

        max_queue_wait_ms = 0.0
        max_model_exec_ms = 0.0
        for item in request_results:
            q_ms = float(item.get("queue_wait_ms", 0.0) or 0.0)
            e_ms = float(item.get("model_exec_ms", 0.0) or 0.0)
            if q_ms > max_queue_wait_ms:
                max_queue_wait_ms = q_ms
            if e_ms > max_model_exec_ms:
                max_model_exec_ms = e_ms

        overall_ms = max(0.0, (time.time() - start_ts) * 1000.0)
        return {
            "queue_wait_ms": max_queue_wait_ms,
            "model_exec_ms": max_model_exec_ms,
            "overall_model_stage_ms": overall_ms,
        }

    def infer(self, frame, target_set, conf_thres=0.5, timeout=10.0):
        model_ids = self.resolve_model_ids(target_set)
        if not model_ids:
            return {"boxes": [], "perf": self._empty_perf()}

        start_ts = time.time()
        deadline = time.time() + timeout
        pending_requests = []
        for model_id in model_ids:
            worker = self.workers[model_id]
            pending_requests.append((worker, worker.submit(frame, target_set, conf_thres)))

        merged_boxes = []
        request_results = []
        for worker, request in pending_requests:
            try:
                result = worker.wait_result(request, deadline)
                request_results.append(result)
                merged_boxes.extend(result.get("boxes", []))
            except TimeoutError:
                print(f"[BatchInfer] WARNING: {self.device}/{worker.model_id} 推理超时")
            except Exception as exc:
                print(f"[BatchInfer] WARNING: {self.device}/{worker.model_id} 等待结果失败: {exc}")
        return {
            "boxes": merged_boxes,
            "perf": self.summarize_result_metrics(start_ts, request_results),
        }

    def submit_model(self, model_id, frame, target_labels, conf_thres=0.5):
        worker = self.workers.get(model_id)
        if worker is None:
            raise KeyError(f"未知模型: {model_id}")
        labels = set(target_labels or worker.supported_labels)
        return worker.submit(frame, labels, conf_thres)

    def wait_model_result(self, model_id, request, deadline):
        worker = self.workers.get(model_id)
        if worker is None:
            raise KeyError(f"未知模型: {model_id}")
        return worker.wait_result(request, deadline)

    def wait_model(self, model_id, request, deadline):
        return self.wait_model_result(model_id, request, deadline).get("boxes", [])

    def stop(self):
        for worker in self.workers.values():
            worker.stop()

class ClusterBatchService:
    """
    跨多 GPU 切分复合推理任务，将单流的多模型堆叠延迟减半。
    """
    def __init__(self, services):
        self.services = services
        self.device = "cluster"
        self._pool_version = services[0]._pool_version if hasattr(services[0], "_pool_version") else 0

    def get(self, key, default=None):
        if key == "_pool_version":
            return getattr(self, "_pool_version", 0)
        return default

    def describe_models(self):
        return self.services[0].describe_models()

    def supported_labels(self):
        return self.services[0].supported_labels()

    def resolve_model_ids(self, target_set):
        return self.services[0].resolve_model_ids(target_set)
        
    def infer(self, frame, target_set, conf_thres=0.5, timeout=10.0):
        model_ids = self.resolve_model_ids(target_set)
        if not model_ids:
            return {"boxes": [], "perf": MultiModelBatchService._empty_perf()}

        start_ts = time.time()
        deadline = time.time() + timeout
        
        # 将本次需要的 model 均匀打散分配给多个 GPU service
        pending_requests = []
        for idx, model_id in enumerate(model_ids):
            target_svc = self.services[idx % len(self.services)]
            worker = target_svc.workers[model_id]
            pending_requests.append((target_svc.device, worker, worker.submit(frame, target_set, conf_thres)))

        merged_boxes = []
        max_queue_wait_ms = 0.0
        max_model_exec_ms = 0.0
        
        for dev, worker, request in pending_requests:
            try:
                result = worker.wait_result(request, deadline)
                merged_boxes.extend(result.get("boxes", []))
                
                q_ms = float(result.get("queue_wait_ms", 0.0) or 0.0)
                e_ms = float(result.get("model_exec_ms", 0.0) or 0.0)
                if q_ms > max_queue_wait_ms: max_queue_wait_ms = q_ms
                if e_ms > max_model_exec_ms: max_model_exec_ms = e_ms
            except Exception as exc:
                print(f"[ClusterInfer] WARNING: {dev}/{worker.model_id} 等待结果失败: {exc}")

        overall_ms = max(0.0, (time.time() - start_ts) * 1000.0)
        return {
            "boxes": merged_boxes,
            "perf": {
                "queue_wait_ms": max_queue_wait_ms,
                "model_exec_ms": max_model_exec_ms,
                "overall_model_stage_ms": overall_ms,
            }
        }

    def _get_target_svc(self, model_id):
        all_models = sorted(self.services[0].workers.keys())
        try:
            idx = all_models.index(model_id)
            return self.services[idx % len(self.services)]
        except ValueError:
            return self.services[0]

    def submit_model(self, model_id, frame, target_labels, conf_thres=0.5):
        svc = self._get_target_svc(model_id)
        return svc.submit_model(model_id, frame, target_labels, conf_thres)

    def wait_model_result(self, model_id, request, deadline):
        svc = self._get_target_svc(model_id)
        return svc.wait_model_result(model_id, request, deadline)

    def wait_model(self, model_id, request, deadline):
        svc = self._get_target_svc(model_id)
        return svc.wait_model(model_id, request, deadline)

    @staticmethod
    def summarize_result_metrics(start_ts, request_results):
        return MultiModelBatchService.summarize_result_metrics(start_ts, request_results)

    def stop(self):
        for svc in self.services:
            svc.stop()
