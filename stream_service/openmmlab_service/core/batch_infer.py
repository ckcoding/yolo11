import queue
import threading
import time
from collections import defaultdict

import numpy as np

_REGISTER_LOCK = threading.Lock()
_MODULES_REGISTERED = False


def _register_openmmlab_modules():
    global _MODULES_REGISTERED
    if _MODULES_REGISTERED:
        return

    with _REGISTER_LOCK:
        if _MODULES_REGISTERED:
            return

        try:
            from mmyolo.utils import register_all_modules as register_mmyolo_modules

            register_mmyolo_modules()
        except Exception:
            pass

        try:
            from mmdet.utils import register_all_modules as register_mmdet_modules

            register_mmdet_modules()
        except Exception:
            pass

        _MODULES_REGISTERED = True


class OpenMMLabModelAdapter:
    def __init__(self, spec, device):
        _register_openmmlab_modules()

        from mmdet.apis import inference_detector, init_detector

        self.model_id = spec["id"]
        self.config_path = spec["config"]
        self.checkpoint_path = spec["checkpoint"]
        self.device = device
        self.default_score_thr = float(spec.get("score_thr", 0.25) or 0.25)
        self.inference_detector = inference_detector
        self.model = init_detector(self.config_path, self.checkpoint_path, device=device)
        self.class_names = self._resolve_class_names(spec)
        self.label_to_index = {label: idx for idx, label in enumerate(self.class_names)}

    def _resolve_class_names(self, spec):
        configured_classes = [str(item).strip() for item in spec.get("classes", []) if str(item).strip()]
        if configured_classes:
            return tuple(configured_classes)

        dataset_meta = getattr(self.model, "dataset_meta", None) or {}
        meta_classes = dataset_meta.get("classes")
        if meta_classes:
            return tuple(str(item) for item in meta_classes)

        model_classes = getattr(self.model, "CLASSES", None)
        if model_classes:
            return tuple(str(item) for item in model_classes)

        cfg = getattr(self.model, "cfg", None)
        if cfg is not None:
            for attr_name in ("metainfo", "dataset_meta"):
                meta = getattr(cfg, attr_name, None)
                if isinstance(meta, dict) and meta.get("classes"):
                    return tuple(str(item) for item in meta["classes"])

        raise RuntimeError(
            f"模型 {self.model_id} 未提供类别名称，请在 openmmlab_service/config.yml 的 models.items[].classes 中显式配置"
        )

    def warmup(self):
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        self.predict_batch([dummy], self.default_score_thr)

    def predict_batch(self, frames, score_thr=None):
        threshold = float(self.default_score_thr if score_thr is None else score_thr)
        try:
            raw_results = self.inference_detector(self.model, frames)
            if not isinstance(raw_results, list):
                raw_results = [raw_results]
        except Exception:
            raw_results = [self.inference_detector(self.model, frame) for frame in frames]
        return [self._parse_result(result, threshold) for result in raw_results]

    def _parse_result(self, result, score_thr):
        if hasattr(result, "pred_instances"):
            pred_instances = result.pred_instances
            bboxes = self._to_numpy(getattr(pred_instances, "bboxes", []))
            scores = self._to_numpy(getattr(pred_instances, "scores", []))
            labels = self._to_numpy(getattr(pred_instances, "labels", []))
            return self._merge_predictions(bboxes, scores, labels, score_thr)

        if isinstance(result, tuple):
            result = result[0]

        if isinstance(result, list):
            merged = []
            for cls_idx, cls_boxes in enumerate(result):
                if cls_boxes is None:
                    continue
                arr = self._to_numpy(cls_boxes)
                if arr.size == 0:
                    continue
                if arr.ndim == 1:
                    arr = arr.reshape(1, -1)
                label_name = self.class_names[cls_idx] if cls_idx < len(self.class_names) else str(cls_idx)
                for row in arr:
                    if len(row) < 5:
                        continue
                    conf = float(row[4])
                    if conf < score_thr:
                        continue
                    merged.append(((float(row[0]), float(row[1]), float(row[2]), float(row[3])), label_name, conf))
            return merged

        return []

    @staticmethod
    def _to_numpy(value):
        if value is None:
            return np.array([])
        if hasattr(value, "detach"):
            value = value.detach()
        if hasattr(value, "cpu"):
            value = value.cpu()
        if hasattr(value, "numpy"):
            return value.numpy()
        return np.asarray(value)

    def _merge_predictions(self, bboxes, scores, labels, score_thr):
        merged = []
        if len(bboxes) == 0 or len(scores) == 0 or len(labels) == 0:
            return merged

        for box_xyxy, score, label_idx in zip(bboxes, scores, labels):
            conf = float(score)
            if conf < score_thr:
                continue
            cls_idx = int(label_idx)
            if cls_idx < 0 or cls_idx >= len(self.class_names):
                continue
            merged.append(
                (
                    (float(box_xyxy[0]), float(box_xyxy[1]), float(box_xyxy[2]), float(box_xyxy[3])),
                    self.class_names[cls_idx],
                    conf,
                )
            )
        return merged


class SingleModelBatchWorker:
    """
    单模型批量推理 Worker。
    对 OpenMMLab 来说，这里优先尝试 list 批量推理；若底层模型不支持，则自动回退逐张执行。
    """

    def __init__(self, model_spec, device, batch_size=1, max_wait_ms=5):
        self.model_id = model_spec["id"]
        self.model_path = model_spec["checkpoint"]
        self.model_config = model_spec["config"]
        self.device = device
        self.batch_size = batch_size
        self.max_wait_sec = max_wait_ms / 1000.0
        self.adapter = OpenMMLabModelAdapter(model_spec, device)
        self.supported_labels = set(self.adapter.class_names)
        self._request_queue = queue.Queue()
        self._running = True
        self._log_counter = 0
        self._batch_counter = 0
        self._thread = threading.Thread(
            target=self._batch_loop,
            daemon=True,
            name=f"BatchInfer-{device}-{self.model_id}",
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
            "config": self.model_config,
            "path": self.model_path,
            "device": self.device,
            "batch_size": self.batch_size,
            "labels": sorted(self.supported_labels),
        }

    def warmup(self):
        self.adapter.warmup()

    def _gather_requests(self):
        requests = []
        try:
            requests.append(self._request_queue.get(timeout=1.0))
        except queue.Empty:
            return requests

        while len(requests) < self.batch_size:
            try:
                requests.append(self._request_queue.get_nowait())
            except queue.Empty:
                break

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
            frames = [req["frame"] for req in requests]
            min_conf = min(req["conf_thres"] for req in requests)
            per_request_boxes = [[] for _ in requests]
            model_exec_ms = 0.0

            try:
                exec_start_ts = time.time()
                results = self.adapter.predict_batch(frames, min_conf)
                model_exec_ms = (time.time() - exec_start_ts) * 1000.0
                for idx, boxes in enumerate(results):
                    target_set = requests[idx]["target_set"]
                    if target_set:
                        boxes = [item for item in boxes if item[1] in target_set]
                    per_request_boxes[idx] = boxes
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
    跨多 GPU 切分复合推理任务。
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
                if q_ms > max_queue_wait_ms:
                    max_queue_wait_ms = q_ms
                if e_ms > max_model_exec_ms:
                    max_model_exec_ms = e_ms
            except Exception as exc:
                print(f"[ClusterInfer] WARNING: {dev}/{worker.model_id} 等待结果失败: {exc}")

        overall_ms = max(0.0, (time.time() - start_ts) * 1000.0)
        return {
            "boxes": merged_boxes,
            "perf": {
                "queue_wait_ms": max_queue_wait_ms,
                "model_exec_ms": max_model_exec_ms,
                "overall_model_stage_ms": overall_ms,
            },
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
