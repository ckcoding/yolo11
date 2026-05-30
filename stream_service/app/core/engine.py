import math
import os
import subprocess
import time

import cv2

from app.config import OUTPUT_FPS
from app.core.draw import cv2_draw_chinese_batch
from app.core.events import dispatch_preview_frame, dispatch_snapshot_event
from app.core.pipeline import execute_pipeline
from app.core.pool import model_pool
from app.core.state import active_tasks


def _detect_encoder():
    """启动时探测 FFmpeg 支持哪些编码器"""
    ffmpeg_bin = os.environ.get("FFMPEG_BIN", "ffmpeg")
    try:
        result = subprocess.run(
            [ffmpeg_bin, "-hide_banner", "-encoders"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        encoders = result.stdout
        if "h264_nvenc" in encoders:
            return "h264_nvenc", ["-c:v", "h264_nvenc", "-pix_fmt", "yuv420p", "-preset", "fast"]
        if "libx264" in encoders:
            return "libx264", ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "ultrafast", "-tune", "zerolatency"]
        return "mpeg4", ["-c:v", "mpeg4", "-pix_fmt", "yuv420p", "-q:v", "5"]
    except Exception:
        return "mpeg4", ["-c:v", "mpeg4", "-pix_fmt", "yuv420p", "-q:v", "5"]


FFMPEG_BIN = os.environ.get("FFMPEG_BIN", "ffmpeg")
ENCODER_NAME, ENCODER_OPTS = _detect_encoder()
print(f"[Engine] FFmpeg 视频编码器: {ENCODER_NAME}")


def _resolve_output_fps(source_fps: float, configured_output_fps: float) -> float:
    source = float(source_fps) if math.isfinite(source_fps) and source_fps > 0 else float(configured_output_fps)
    return max(1.0, min(source, float(configured_output_fps)))


def push_stream_worker(
    task_id: str,
    input_url: str,
    push_url: str,
    output_fps: float,
    model_obj: dict,
    target_classes: dict,
    conf_thres: float,
    mode: str = "stream",
    pipeline_nodes: list | None = None,
    confidence_display: bool = False,
):
    """分离出的多层组合推理线（融合双擎流分发）"""
    service = model_obj["service"]
    cap = cv2.VideoCapture(input_url)

    if not cap.isOpened():
        print(f"[{task_id}] 无法打开视频流 {input_url}")
        active_tasks.update_fields(task_id, status="error")
        active_tasks.pop(task_id, None)
        model_pool.release(model_obj)
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    if not (0 < fps < 120):
        fps = 25
    output_fps = _resolve_output_fps(float(fps), float(output_fps or OUTPUT_FPS))

    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    is_local_file = str(input_url).endswith(".mp4") or str(input_url).endswith(".avi") or str(input_url).startswith("/workspace/")
    print(f"[{task_id}] 源帧率={fps:.1f} FPS, 目标输出={float(OUTPUT_FPS):.1f} FPS, 实际输出={output_fps:.1f} FPS")

    process = None
    if mode in ["stream", "webrtc", "flv"]:
        print(f"[{task_id}] 使用流媒体管线: {ENCODER_NAME}")
        command = [
            FFMPEG_BIN,
            "-y",
            "-f",
            "rawvideo",
            "-vcodec",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-s",
            f"{width}x{height}",
            "-r",
            str(output_fps),
            "-i",
            "-",
        ] + ENCODER_OPTS + [
            "-loglevel",
            "error",
            "-f",
            "flv",
            push_url,
        ]

        process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)
        time.sleep(0.5)
        if process.poll() is not None:
            print(f"[{task_id}] FFmpeg 启动失败! 退出码: {process.returncode}")
            active_tasks.update_fields(task_id, status="error")
            active_tasks.pop(task_id, None)
            model_pool.release(model_obj)
            return

    active_tasks.update_fields(
        task_id,
        process=process,
        status="processing",
        source_fps=round(float(fps), 1),
        output_fps=round(float(output_fps), 1),
        current_fps=0.0,
        frame_count=0,
        queue_wait_ms=0.0,
        model_exec_ms=0.0,
        draw_ms=0.0,
        write_ms=0.0,
    )

    current_target_set = {item.strip() for item in target_classes.keys()}
    current_pipeline = [dict(node) for node in (pipeline_nodes or [])]
    cooldown_dict = {}
    last_ws_push = 0.0

    cached_boxes = []
    cached_node_results = {}
    latest_queue_wait_ms = 0.0
    latest_model_exec_ms = 0.0
    latest_draw_ms = 0.0
    latest_write_ms = 0.0
    node_last_submit_ts = {}
    frame_gate_accumulator = 0.0

    # 同步推理：每帧检测、每帧绘制、框与画面完美同步
    # 凑批已从 35ms 压到 5ms，单路延迟约 26ms，15fps(66ms) 预算内绰绰有余
    # 多路流各自运行在独立线程中，互不阻塞

    try:
        frame_interval = 1.0 / output_fps
        frame_count = 0
        fps_window_start = time.time()
        fps_window_frames = 0
        while active_tasks.get_field(task_id, "status") == "processing":
            start_total = time.time()

            latest_target_classes = active_tasks.get_field(task_id, "target_classes", target_classes)
            latest_pipeline = active_tasks.get_field(task_id, "pipeline", current_pipeline)
            if latest_target_classes != target_classes or latest_pipeline != current_pipeline:
                target_classes = latest_target_classes
                current_target_set = {item.strip() for item in target_classes.keys()}
                current_pipeline = [dict(node) for node in (latest_pipeline or [])]
                cached_boxes = []
                cached_node_results = {}
                latest_queue_wait_ms = 0.0
                latest_model_exec_ms = 0.0
                node_last_submit_ts = {}

            start_infer = time.time()
            frame_gate_accumulator += output_fps
            should_process_frame = frame_gate_accumulator + 1e-6 >= fps
            if should_process_frame:
                frame_gate_accumulator = max(0.0, frame_gate_accumulator - fps)

            start_read = time.time()
            if not should_process_frame:
                # 不需要处理的帧只做 grab() 跳过解码拷贝，节省 CPU
                cap.grab()
                continue
            ret, frame = cap.read()
            read_time = time.time() - start_read

            if not ret:
                if is_local_file:
                    print(f"[{task_id}] 测试视频到达末尾，重新从头播放...")
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    frame_gate_accumulator = 0.0
                    start_read = time.time()
                    ret, frame = cap.read()
                    read_time += time.time() - start_read
                    if not ret:
                        print(f"[{task_id}] 视频循环重置失败，安全退出")
                        break
                else:
                    print(f"[{task_id}] 视频流读取结束或断开 (已处理 {frame_count} 帧)")
                    break

            def _due_node_ids(now_ts):
                due_ids = set()
                for node in current_pipeline:
                    node_id = node["id"]
                    node_fps = node.get("fps")
                    if not node_fps:
                        due_ids.add(node_id)
                        continue
                    last_ts = node_last_submit_ts.get(node_id)
                    if last_ts is None or now_ts - last_ts >= 1.0 / node_fps:
                        due_ids.add(node_id)
                return due_ids

            due_ids = _due_node_ids(time.time()) if current_pipeline else set()
            has_detect_work = bool(current_target_set) or bool(current_pipeline)

            if has_detect_work:
                current_svc = model_pool.get_main_service() or service
                
                try:
                    if current_pipeline and due_ids:
                        sub_ts = time.time()
                        for node_id in due_ids:
                            node_last_submit_ts[node_id] = sub_ts
                        infer_result = execute_pipeline(current_svc, frame, current_pipeline, 10.0, cached_node_results.copy(), due_ids)
                    else:
                        infer_result = current_svc.infer(frame, current_target_set, conf_thres, 10.0)
                    
                    if infer_result is not None:
                        if isinstance(infer_result, dict):
                            cached_boxes = infer_result.get("boxes", [])
                            if "node_results" in infer_result:
                                cached_node_results = infer_result.get("node_results", {})
                            perf_info = infer_result.get("perf", {})
                            latest_queue_wait_ms = float(perf_info.get("queue_wait_ms", 0.0) or 0.0)
                            latest_model_exec_ms = float(perf_info.get("overall_model_stage_ms", perf_info.get("model_exec_ms", 0.0)) or 0.0)
                        else:
                            cached_boxes = infer_result
                            latest_queue_wait_ms = 0.0
                            latest_model_exec_ms = 0.0
                            
                        active_tasks.update_fields(
                            task_id,
                            queue_wait_ms=round(latest_queue_wait_ms, 1),
                            model_exec_ms=round(latest_model_exec_ms, 1),
                        )
                except Exception as exc:
                    print(f"[{task_id}] 推理异常抛出: {exc}")
                    active_tasks.update_fields(task_id, queue_wait_ms=0.0, model_exec_ms=0.0)
            else:
                cached_boxes = []
                cached_node_results = {}
                latest_queue_wait_ms = 0.0
                latest_model_exec_ms = 0.0
                active_tasks.update_fields(task_id, queue_wait_ms=0.0, model_exec_ms=0.0)

            all_boxes = cached_boxes
            infer_time = time.time() - start_infer

            start_draw = time.time()
            frame, labels_detected = cv2_draw_chinese_batch(
                frame,
                all_boxes,
                target_classes,
                show_confidence=confidence_display,
            )
            draw_time = time.time() - start_draw
            latest_draw_ms = draw_time * 1000.0

            if labels_detected:
                current_time = time.time()
                triggered_labels = set()
                for label in labels_detected:
                    if current_time - cooldown_dict.get(label, 0) > 2.0:
                        triggered_labels.add(label)
                        cooldown_dict[label] = current_time

                if triggered_labels:
                    ret_jpg, img_encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                    if ret_jpg:
                        dispatch_snapshot_event(task_id, triggered_labels, img_encoded.tobytes())

            try:
                start_write = time.time()
                if mode in ["stream", "webrtc", "flv"] and process:
                    process.stdin.write(frame.tobytes())
                elif mode == "websocket":
                    curr = time.time()
                    if curr - last_ws_push > 0.15:
                        last_ws_push = curr
                        scale_ratio = min(1.0, 1920 / width)
                        if scale_ratio < 1.0:
                            preview_h = int(height * scale_ratio)
                            preview_frame = cv2.resize(
                                frame,
                                (int(width * scale_ratio), preview_h),
                                interpolation=cv2.INTER_LINEAR,
                            )
                        else:
                            preview_frame = frame

                        ret_jpg, img_encoded = cv2.imencode(".jpg", preview_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                        if ret_jpg:
                            dispatch_preview_frame(task_id, img_encoded.tobytes())

                write_time = time.time() - start_write
                latest_write_ms = write_time * 1000.0
                total_time = time.time() - start_total

                frame_count += 1
                fps_window_frames += 1
                now_ts = time.time()
                fps_elapsed = now_ts - fps_window_start
                if fps_elapsed >= 1.0:
                    current_fps = fps_window_frames / fps_elapsed
                    active_tasks.update_fields(
                        task_id,
                        current_fps=round(current_fps, 1),
                        frame_count=frame_count,
                        queue_wait_ms=round(latest_queue_wait_ms, 1),
                        model_exec_ms=round(latest_model_exec_ms, 1),
                        draw_ms=round(latest_draw_ms, 1),
                        write_ms=round(latest_write_ms, 1),
                    )
                    fps_window_start = now_ts
                    fps_window_frames = 0

                if frame_count % 100 == 0:
                    pure_model_ms = max(0.0, latest_model_exec_ms - latest_queue_wait_ms)
                    print(
                        f"[{task_id}] 帧耗时 ({frame_count}): "
                        f"读取={read_time * 1000:.1f}ms, 凑批={latest_queue_wait_ms:.1f}ms, "
                        f"推理={pure_model_ms:.1f}ms, 绘制={latest_draw_ms:.1f}ms, "
                        f"推流={latest_write_ms:.1f}ms | "
                        f"总延迟={total_time * 1000:.1f}ms | "
                        f"检测框={len(all_boxes)}"
                    )
            except BrokenPipeError:
                print(f"[{task_id}] FFmpeg 管道断裂!")
                break
            except Exception as exc:
                print(f"[{task_id}] 写入错误: {exc}")
                break

            elapsed = time.time() - start_total
            if is_local_file and elapsed < frame_interval:
                time.sleep(frame_interval - elapsed)

    except Exception as exc:
        print(f"[{task_id}] 推流处理崩溃: {exc}")
    finally:
        cap.release()
        if process:
            try:
                process.stdin.close()
            except Exception:
                pass
            try:
                process.terminate()
                process.wait(timeout=3)
            except Exception:
                try:
                    process.kill()
                    process.wait(timeout=1)
                except Exception:
                    pass

        model_pool.release(model_obj)
        active_tasks.pop(task_id, None)
        print(f"[{task_id}] 任务线程安全退出，流控令牌已归还停车场。")
