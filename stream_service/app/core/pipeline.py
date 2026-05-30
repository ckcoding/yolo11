import time

from app.config import get_label_display_name


def _catalog_maps(model_pool):
    models = model_pool.describe_models()
    labels_by_model = {}
    model_order = []
    for item in models:
        model_id = item["model_id"]
        model_order.append(model_id)
        labels_by_model[model_id] = set(item["labels"])
    return labels_by_model, model_order


def normalize_pipeline(model_pool, target_classes, raw_pipeline, default_conf=0.25, allow_empty=False):
    labels_by_model, model_order = _catalog_maps(model_pool)
    target_classes = dict(target_classes or {})
    normalized_nodes = []
    selected_models = []
    selected_labels = []
    seen_node_ids = set()

    if raw_pipeline:
        for idx, raw_node in enumerate(raw_pipeline):
            node = raw_node.model_dump() if hasattr(raw_node, "model_dump") else dict(raw_node)
            model_id = str(node.get("model", "")).strip()
            if model_id not in labels_by_model:
                raise ValueError(f"未加载的模型: {model_id}")

            node_id = str(node.get("id") or f"{model_id}_{idx + 1}").strip()
            if not node_id or node_id in seen_node_ids:
                raise ValueError(f"流水线节点 ID 无效或重复: {node_id or model_id}")

            labels = node.get("labels") or sorted(labels_by_model[model_id])
            labels = [str(label).strip() for label in labels if str(label).strip()]
            if not labels:
                raise ValueError(f"节点 {node_id} 未指定有效 labels")

            unsupported = sorted(set(labels) - labels_by_model[model_id])
            if unsupported:
                raise ValueError(f"节点 {node_id} 指定了模型 {model_id} 不支持的标签: {unsupported}")

            depends_on = str(node.get("depends_on") or "").strip() or None
            if depends_on and depends_on not in seen_node_ids:
                raise ValueError(f"节点 {node_id} 依赖的上游节点不存在或顺序错误: {depends_on}")

            fps = node.get("fps")
            if fps is not None:
                try:
                    fps = float(fps)
                except (TypeError, ValueError):
                    raise ValueError(f"节点 {node_id} 的 fps 非法")
                if fps <= 0:
                    raise ValueError(f"节点 {node_id} 的 fps 必须大于 0")

            conf = node.get("conf")
            if conf is None:
                conf = default_conf
            try:
                conf = float(conf)
            except (TypeError, ValueError):
                raise ValueError(f"节点 {node_id} 的 conf 非法")

            roi_labels = node.get("roi_labels") or []
            roi_labels = [str(label).strip() for label in roi_labels if str(label).strip()]
            max_rois = node.get("max_rois")
            if max_rois is not None:
                try:
                    max_rois = int(max_rois)
                except (TypeError, ValueError):
                    raise ValueError(f"节点 {node_id} 的 max_rois 非法")
                if max_rois <= 0:
                    raise ValueError(f"节点 {node_id} 的 max_rois 必须大于 0")
            elif depends_on:
                max_rois = 8

            normalized_nodes.append(
                {
                    "id": node_id,
                    "model": model_id,
                    "labels": labels,
                    "fps": fps,
                    "conf": conf,
                    "depends_on": depends_on,
                    "roi_labels": roi_labels,
                    "max_rois": max_rois,
                }
            )
            seen_node_ids.add(node_id)
    else:
        if not target_classes:
            if allow_empty:
                return {
                    "pipeline": [],
                    "target_classes": {},
                    "selected_models": [],
                }
            raise ValueError("请至少指定一个检测目标")

        labels_by_node = {}
        for label in target_classes.keys():
            matched_model = None
            for model_id in model_order:
                if label in labels_by_model[model_id]:
                    matched_model = model_id
                    break
            if matched_model is None:
                raise ValueError(f"存在未加载的检测目标: {label}")
            labels_by_node.setdefault(matched_model, []).append(label)

        for model_id in model_order:
            labels = labels_by_node.get(model_id)
            if not labels:
                continue
            normalized_nodes.append(
                {
                    "id": model_id,
                    "model": model_id,
                    "labels": labels,
                    "fps": None,
                    "conf": float(default_conf),
                    "depends_on": None,
                    "roi_labels": [],
                    "max_rois": None,
                }
            )

    used_labels = []
    seen_labels = set()
    for node in normalized_nodes:
        model_id = node["model"]
        if model_id not in selected_models:
            selected_models.append(model_id)
        for label in node["labels"]:
            if label not in seen_labels:
                seen_labels.add(label)
                used_labels.append(label)
                target_classes.setdefault(label, get_label_display_name(label))

    return {
        "pipeline": normalized_nodes,
        "target_classes": {label: target_classes[label] for label in used_labels},
        "selected_models": selected_models,
    }


def _clip_box(box_xyxy, width, height):
    x1, y1, x2, y2 = box_xyxy
    x1 = max(0, min(int(x1), width - 1))
    y1 = max(0, min(int(y1), height - 1))
    x2 = max(0, min(int(x2), width))
    y2 = max(0, min(int(y2), height))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def _empty_perf():
    return {
        "queue_wait_ms": 0.0,
        "model_exec_ms": 0.0,
        "overall_model_stage_ms": 0.0,
    }


def _run_single_model(service, frame, node, deadline):
    request = service.submit_model(node["model"], frame, node["labels"], node["conf"])
    result = service.wait_model_result(node["model"], request, deadline)
    return result.get("boxes", []), [result]


def _run_dependency_model(service, frame, node, dependency_boxes, deadline):
    height, width = frame.shape[:2]
    allowed_labels = set(node["roi_labels"] or [])
    source_boxes = []
    for box_xyxy, label_en, conf in dependency_boxes:
        if allowed_labels and label_en not in allowed_labels:
            continue
        clipped = _clip_box(box_xyxy, width, height)
        if clipped is not None:
            source_boxes.append((clipped, label_en, conf))

    if node["max_rois"]:
        source_boxes = source_boxes[: node["max_rois"]]
    if not source_boxes:
        return [], []

    pending = []
    for clipped_box, _, _ in source_boxes:
        x1, y1, x2, y2 = clipped_box
        roi = frame[y1:y2, x1:x2].copy()
        if roi.size == 0:
            continue
        request = service.submit_model(node["model"], roi, node["labels"], node["conf"])
        pending.append((clipped_box, request))

    merged = []
    request_results = []
    for clipped_box, request in pending:
        x1, y1, _, _ = clipped_box
        result = service.wait_model_result(node["model"], request, deadline)
        request_results.append(result)
        boxes = result.get("boxes", [])
        for roi_box_xyxy, label_en, conf in boxes:
            rx1, ry1, rx2, ry2 = roi_box_xyxy
            merged.append(((rx1 + x1, ry1 + y1, rx2 + x1, ry2 + y1), label_en, conf))
    return merged, request_results


def execute_pipeline(service, frame, pipeline_nodes, timeout=10.0, base_results=None, run_node_ids=None):
    if not pipeline_nodes:
        return {"boxes": [], "node_results": {}, "perf": _empty_perf()}

    start_ts = time.time()
    deadline = time.time() + timeout
    node_results = {node_id: list(boxes) for node_id, boxes in (base_results or {}).items()}
    request_results = []

    # 将节点按依赖关系分层：无依赖的独立节点可并行提交
    independent_nodes = []
    dependent_nodes = []
    for node in pipeline_nodes:
        node_id = node["id"]
        should_run = run_node_ids is None or node_id in run_node_ids
        if not should_run and node_id in node_results:
            continue
        if node.get("depends_on"):
            dependent_nodes.append(node)
        else:
            independent_nodes.append(node)

    # 阶段一：所有独立节点同时提交，实现 GPU 并行推理
    if independent_nodes:
        pending = []
        for node in independent_nodes:
            request = service.submit_model(node["model"], frame, node["labels"], node["conf"])
            pending.append((node, request))

        # 统一收割所有独立节点结果
        for node, request in pending:
            try:
                result = service.wait_model_result(node["model"], request, deadline)
                request_results.append(result)
                node_results[node["id"]] = result.get("boxes", [])
            except TimeoutError:
                node_results[node["id"]] = []
            except Exception as exc:
                print(f"[Pipeline] 独立节点 {node['id']} 推理失败: {exc}")
                node_results[node["id"]] = []

    # 阶段二：依赖节点按原有顺序串行执行（需要上游结果才能运行）
    for node in dependent_nodes:
        depends_on = node["depends_on"]
        boxes, node_request_results = _run_dependency_model(
            service,
            frame,
            node,
            node_results.get(depends_on, []),
            deadline,
        )
        node_results[node["id"]] = boxes
        request_results.extend(node_request_results)

    merged_boxes = []
    for node in pipeline_nodes:
        merged_boxes.extend(node_results.get(node["id"], []))

    return {
        "boxes": merged_boxes,
        "node_results": node_results,
        "perf": service.summarize_result_metrics(start_ts, request_results),
    }

