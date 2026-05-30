import argparse
import importlib.util
import logging
import queue
import random
import subprocess
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from threading import Thread
from typing import Dict, List, Optional

import cv2
import numpy as np
import psutil
import uvicorn
from PIL import Image, ImageDraw, ImageFont
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
from pathlib import Path
import torch
import multiprocessing
import gc  # 添加垃圾回收模块
import contextlib
import sys

# 导入默认配置
from config import *
from ultralytics import YOLO

app = FastAPI(title="视频流检测服务", description="YOLO多路视频流实时检测服务")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许所有来源
    allow_credentials=True,
    allow_methods=["*"],  # 允许所有方法
    allow_headers=["*"],  # 允许所有头部
)


# 命令行参数处理
def parse_args():
    parser = argparse.ArgumentParser(description="YOLO视频流检测服务")
    parser.add_argument('config_file', nargs='?', default='config.py',
                        help='配置文件路径 (默认: config.py)')
    parser.add_argument('--gpu', type=int, default=None,
                        help='指定默认使用的GPU卡号，优先级高于配置文件 (例如: --gpu 0)')
    return parser.parse_args()


# 动态导入配置
def load_config(config_path):
    if not os.path.exists(config_path):
        print(f"配置文件不存在: {config_path}")
        print("使用默认配置...")
        return

    try:
        # 动态加载配置模块
        spec = importlib.util.spec_from_file_location("config", config_path)
        config_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config_module)

        # 更新全局变量
        globals().update({k: v for k, v in config_module.__dict__.items()
                          if not k.startswith('__')})
        print(f"成功加载配置: {config_path}")
    except Exception as e:
        print(f"加载配置文件出错: {e}")
        print("使用默认配置...")


# 初始化配置和日志
def init_app():
    # 设置日志
    log_level = getattr(logging, LOGGING["level"])
    logging.basicConfig(
        level=log_level,
        format=LOGGING["format"]
    )
    logger = logging.getLogger(__name__)
    logger.info("初始化应用...")

    # 初始化全局变量
    global gpu_usage, cn_font

    # 初始化GPU使用状态
    gpu_usage = {i: 0 for i in range(GPU_COUNT)} if CUDA_AVAILABLE else {}

    # 加载中文字体
    cn_font = None
    try:
        cn_font = ImageFont.truetype(FONT["path"], FONT["size"])
        logger.info(f"成功加载中文字体: {FONT['path']}")
    except Exception as e:
        logger.error(f"加载中文字体失败: {e}")

    # 记录GPU信息
    if CUDA_AVAILABLE:
        logger.info(f"找到 {GPU_COUNT} 个GPU设备: {[torch.cuda.get_device_name(i) for i in range(GPU_COUNT)]}")
    else:
        logger.warning("未找到GPU设备，将使用CPU进行检测")

    return logger


class StreamRequest(BaseModel):
    stream_url: str
    names_dict: str
    confidence_threshold: float = 0.25
    confidence_display: bool = False  # 添加控制标签显示的参数，默认不显示
    gpu_id: Optional[int] = 0  # 添加指定GPU卡号的参数，默认为None，表示自动选择


class StreamIDRequest(BaseModel):
    stream_id: str


class StreamInfo(BaseModel):
    stream_id: str
    stream_url: str
    detecting_classes: str
    status: str
    start_time: float
    frames_processed: int = 0
    fps: float = 0
    playback_url: str
    web_rtc: str
    gpu_id: Optional[int] = None
    frame_skip_count: int = 0  # 当前跳过的帧数
    reconnect_count: int = 0  # 重连次数计数


# 辅助函数 - OpenCV到PIL图像转换
def cv2_to_pil(cv2_img):
    cv2_img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(cv2_img)


# 辅助函数 - PIL到OpenCV图像转换
def pil_to_cv2(pil_img):
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


def select_gpu(stream_id):
    if not CUDA_AVAILABLE or GPU_COUNT == 0:
        return None

    # 随机选择GPU策略
    available_gpus = [gpu for gpu, load in gpu_usage.items() if load < GPU["max_nvenc_per_gpu"]]
    if not available_gpus:
        logger.warning(f"Stream {stream_id} - 所有GPU NVENC会话已满，切换到CPU编码")
        return None
    selected_gpu = random.choice(available_gpus)

    gpu_usage[selected_gpu] += 1
    logger.info(f"Stream {stream_id} - 分配到 GPU {selected_gpu}, 当前使用计数: {gpu_usage}")
    return selected_gpu


def release_gpu(gpu_id):
    if gpu_id is not None and gpu_id in gpu_usage:
        gpu_usage[gpu_id] = max(0, gpu_usage[gpu_id] - 1)
        logger.info(f"释放GPU {gpu_id}, 当前使用计数: {gpu_usage}")


# 用于释放模型资源的辅助函数
def free_gpu_memory(model=None, device=None, model_type=None, model_path=None):
    """彻底释放GPU内存 - 针对不同类型模型的处理方法"""
    try:
        # 1. 释放模型 - 直接使用模型类型信息
        if model is not None:
            try:
                # 直接使用传入的模型类型信息
                is_onnx_model = model_type == 'onnx' or (hasattr(model, 'model_path') and str(getattr(model, 'model_path', '')).endswith('.onnx'))
                
                logger.info(f"释放模型资源: {'ONNX 模型' if is_onnx_model else 'PyTorch 模型'}{f' [{model_path}]' if model_path else ''}")
                
                # ONNX模型专用处理
                if is_onnx_model:
                    # 不要尝试调用任何方法，只清除引用
                    if hasattr(model, 'predictor'):
                        model.predictor = None
                    if hasattr(model, 'model'):
                        model.model = None
                    # 强制GC
                    gc.collect()
                    # 模型引用置空
                    model = None
                else:
                    # PyTorch模型处理 - 只有确定是PyTorch模型才尝试调用cpu()
                    try:
                        if hasattr(model, 'cpu'):
                            model.cpu()
                    except Exception as e:
                        logger.warning(f"将模型移至CPU时出错 (忽略此错误): {str(e)}")
                    
                    # 安全删除引用
                    if hasattr(model, 'model'):
                        model.model = None
                    if hasattr(model, 'predictor'):
                        model.predictor = None
                    
                    # 最后删除整个模型引用
                    model = None
                    
                # 强制垃圾回收
                gc.collect()
            except Exception as e:
                logger.error(f"释放模型资源时出错: {str(e)}")
                # 确保模型引用被清除
                try:
                    model = None
                    gc.collect()
                except:
                    pass
            
        # 2. 处理ONNX运行时会话
        if model_type == 'onnx':
            try:
                # 尝试直接手动卸载ONNX Runtime的CUDA资源
                import onnxruntime as ort
                gc.collect()  # 触发对未引用ONNX会话的回收
            except (ImportError, Exception):
                pass
        
        # 3. 强制垃圾回收多次
        for _ in range(5):  # 增加回收次数
            gc.collect()
        
        # 4. 清理CUDA缓存和重置设备
        if CUDA_AVAILABLE:
            try:
                # 如果指定了设备，对特定设备执行操作
                if device is not None:
                    # 提取设备索引，支持"cuda:0"或0格式
                    if isinstance(device, str) and ':' in device:
                        device_idx = int(device.split(':')[-1])
                    else:
                        device_idx = device if isinstance(device, int) else 0
                        
                    # 对特定设备执行清理
                    with torch.cuda.device(device_idx):
                        # 重置设备状态
                        torch.cuda.empty_cache()
                        torch.cuda.ipc_collect()
                        torch.cuda.reset_peak_memory_stats(device_idx)
                        torch.cuda.synchronize()
                else:
                    # 对所有设备执行清理
                    for i in range(GPU_COUNT):
                        with torch.cuda.device(i):
                            torch.cuda.empty_cache()
                            torch.cuda.ipc_collect()
                            torch.cuda.reset_peak_memory_stats(i)
                            torch.cuda.synchronize()
                
                # 最终同步
                torch.cuda.synchronize()
                
            except Exception as e:
                logger.error(f"清理CUDA缓存时出错 (忽略继续): {str(e)}")
                
        # 5. 最终垃圾回收
        gc.collect()
                
        logger.info(f"已执行GPU内存彻底清理" + (f" 对设备 {device}" if device is not None else ""))
    except Exception as e:
        logger.error(f"清理GPU内存时出错: {str(e)}")


# 线程池和活动流管理
stream_executor = ThreadPoolExecutor(max_workers=VIDEO_PROCESSING["worker_threads"]["stream"])
detect_executor = ThreadPoolExecutor(max_workers=VIDEO_PROCESSING["worker_threads"]["detection"])
active_streams = {}


def process_frame(model, frame, conf_threshold=0.25, target_classes=None, show_label=False, device='cpu'):
    if frame is None:
        return None
    try:
        # 英文到中文的标签映射
        en_to_cn = CLASS_MAPPING

        # 提高图像质量
        # 1. 保持原始分辨率，不做降采样
        frame_high_quality = frame.copy()

        # 2. 使用配置中的锐化设置
        if VIDEO_PROCESSING.get("sharpening", {}).get("enabled", False):
            # 获取配置的锐化内核和强度
            kernel_values = VIDEO_PROCESSING.get("sharpening", {}).get("kernel",
                                                                       [-0.5, -0.5, -0.5,
                                                                        -0.5, 5.0, -0.5,
                                                                        -0.5, -0.5, -0.5])
            kernel = np.array(kernel_values).reshape(3, 3)
            strength = VIDEO_PROCESSING.get("sharpening", {}).get("strength", 0.8)

            # 应用锐化滤镜
            sharpened = cv2.filter2D(frame_high_quality, -1, kernel)
            # 按强度混合原图和锐化图
            frame_high_quality = cv2.addWeighted(frame_high_quality, 1.0 - strength, sharpened, strength, 0)
        else:
            # 默认锐化处理
            kernel = np.array([[-0.5, -0.5, -0.5],
                               [-0.5, 5.0, -0.5],
                               [-0.5, -0.5, -0.5]])
            frame_high_quality = cv2.filter2D(frame_high_quality, -1, kernel)

        # 3. 可选的轻度降噪
        frame_high_quality = cv2.GaussianBlur(frame_high_quality, (3, 3), 0.5)

        # 使用高质量图像进行检测
        classes_indices = [i for i, name in model.names.items() if name in target_classes] if target_classes else None

        try:
            results = model.predict(
                source=frame_high_quality,  # 使用高质量图像检测
                conf=conf_threshold,
                imgsz=MODEL["img_size"],  # 使用配置的图像尺寸
                classes=classes_indices,
                show=False,
                stream=True,
                verbose=False,
                device=device  # 指定运行设备
            )

            for result in results:
                # 在高质量图像上绘制结果
                annotated_frame = frame_high_quality.copy()
                boxes = result.boxes

                # 获取渲染质量设置
                render_settings = VIDEO_PROCESSING.get("render_quality", {})
                line_thickness = render_settings.get("line_thickness", 3)  # 获取线宽设置
                text_size = render_settings.get("text_size", 1.0)  # 获取文本大小设置
                font_thickness = render_settings.get("font_thickness", 2)  # 获取文本厚度设置

                # 如果有检测结果
                if len(boxes) > 0:
                    # 使用PIL绘制边界框和可选的标签
                    if cn_font is not None:
                        # 转换为PIL图像
                        pil_img = cv2_to_pil(annotated_frame)
                        draw = ImageDraw.Draw(pil_img)

                        for box in boxes:
                            # 获取边界框坐标
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)

                            # 获取类别和置信度
                            cls = int(box.cls[0].item())
                            conf = float(box.conf[0].item())

                            # 获取英文标签
                            en_label = model.names[cls]

                            # 转换为中文标签（如果有映射）
                            cn_label = en_to_cn.get(en_label, en_label)

                            # 在标签中添加置信度
                            if show_label and conf_threshold > 0:
                                cn_label = f"{cn_label} {conf:.2f}"

                            # 绘制更醒目的边界框 (RGB格式) - 使用配置的线宽
                            draw.rectangle([x1, y1, x2, y2], outline=(0, 255, 0), width=line_thickness)

                            # 如果需要显示标签
                            if show_label:
                                # 使用配置的文本大小
                                adjusted_font = cn_font
                                if text_size != 1.0:
                                    try:
                                        adjusted_font = ImageFont.truetype(
                                            FONT["path"],
                                            int(FONT["size"] * text_size)
                                        )
                                    except:
                                        pass  # 使用默认字体

                                # 绘制标签背景
                                text_bbox = draw.textbbox((0, 0), cn_label, font=adjusted_font)
                                text_width = text_bbox[2] - text_bbox[0]
                                text_height = text_bbox[3] - text_bbox[1]

                                draw.rectangle([x1, y1 - text_height - 5, x1 + text_width, y1],
                                               fill=(0, 255, 0))

                                # 绘制中文标签
                                draw.text((x1, y1 - text_height - 5), cn_label, fill=(0, 0, 0), font=adjusted_font)

                        # 转回OpenCV格式
                        annotated_frame = pil_to_cv2(pil_img)
                    else:
                        # 使用OpenCV绘制
                        for box in boxes:
                            # 获取边界框坐标
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)

                            # 获取类别和置信度
                            cls = int(box.cls[0].item())
                            conf = float(box.conf[0].item())

                            # 获取英文标签
                            en_label = model.names[cls]
                            cn_label = en_to_cn.get(en_label, en_label)

                            # 在标签中添加置信度
                            if show_label and conf_threshold > 0:
                                cn_label = f"{cn_label} {conf:.2f}"

                            # 绘制边界框 - BGR颜色（绿色） - 使用配置的线宽
                            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), line_thickness)

                            # 如果需要显示标签
                            if show_label:
                                # 使用配置的文本大小和厚度
                                font_scale = 0.5 * text_size
                                # 绘制标签背景
                                text_size_px = \
                                cv2.getTextSize(cn_label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)[0]
                                cv2.rectangle(annotated_frame, (x1, y1 - text_size_px[1] - 5),
                                              (x1 + text_size_px[0], y1), (0, 255, 0), -1)

                                # 绘制标签文本
                                cv2.putText(annotated_frame, cn_label, (x1, y1 - 5),
                                            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), font_thickness)

                if CUDA_AVAILABLE and isinstance(annotated_frame, torch.Tensor):
                    annotated_frame = annotated_frame.cpu().numpy()
                if not isinstance(annotated_frame, np.ndarray):
                    annotated_frame = np.array(annotated_frame)

                return cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

            return cv2.cvtColor(frame_high_quality, cv2.COLOR_BGR2RGB)

        except Exception as e:
            if "Got invalid dimensions" in str(e):
                logger.error(f"ONNX模型维度错误: {str(e)}")
                logger.error("请确保模型输入尺寸设置为640x640")
            elif "CUDA" in str(e) or "GPU" in str(e):
                logger.error(f"GPU推理错误: {str(e)}")
                logger.error("尝试切换到CPU进行推理")
                # 如果GPU推理失败，尝试使用CPU
                return process_frame(model, frame, conf_threshold, target_classes, show_label, 'cpu')
            else:
                logger.error(f"模型预测失败: {str(e)}")
            # 返回原始帧，不做任何处理
            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    except Exception as e:
        logger.error(f"检测处理失败: {str(e)}")
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def get_ffmpeg_command(output_url: str, width: int, height: int, fps: int) -> dict:
    base_cmd = [FFMPEG["path"], "-loglevel", "info", "-stats"]
    input_options = [
        '-f', 'rawvideo',
        '-pix_fmt', 'rgb24',
        '-s', f'{width}x{height}',
        '-r', str(fps),  # 使用实际检测到的帧率
        '-thread_queue_size', '512',  # 降低队列大小以减少缓冲
        '-i', '-'
    ]

    # 通用低延迟选项
    latency_options = [
        '-fflags', 'nobuffer',  # 关闭输入缓冲
        '-flags', 'low_delay',  # 低延迟标志
        '-vsync', 'passthrough',  # 保持视频同步
        '-movflags', '+faststart',  # 快速启动
        '-g', str(FFMPEG["encoding"]["gop_size"]),  # GOP间隔
        '-keyint_min', str(FFMPEG["encoding"]["keyint_min"]),  # 最小关键帧间隔
        '-flush_packets', '1'  # 立即刷新包
    ]

    # 质量和清晰度参数 - 减少参数避免冲突
    quality_options = [
        '-crf', str(FFMPEG["encoding"]["crf"]),  # 质量控制
    ]

    # 使用CPU编码
    output_options = [
        '-pix_fmt', 'yuv420p',
        '-c:v', 'libx264',
        # 为libx264添加预设
        '-preset', FFMPEG["encoding"]["preset"],  # 使用配置的预设，对于libx264是有效的
        '-b:v', FFMPEG["encoding"]["bitrate"],  # 基础比特率
        '-bufsize', FFMPEG["encoding"]["bufsize"],  # 缓冲区大小
        '-maxrate', FFMPEG["encoding"]["maxrate"],  # 最大比特率
        '-profile:v', FFMPEG["x264"]["profile"],  # 编码配置文件
        '-level', FFMPEG["x264"]["level"],  # 兼容级别
    ]

    # 为libx264添加zerolatency调优
    if FFMPEG["x264"].get("tune") == "zerolatency":
        output_options.extend(['-tune', 'zerolatency'])

    # x264特定参数
    x264_params = []
    if FFMPEG["x264"]["scenecut"] == 0:
        x264_params.append(f'scenecut=0')

    # 添加VBV参数
    if FFMPEG["encoding"]["bufsize"]:
        bufsize_value = FFMPEG["encoding"]["bufsize"].replace('M', '000')
        maxrate_value = FFMPEG["encoding"]["maxrate"].replace('M', '000')
        x264_params.append(f'vbv-bufsize={bufsize_value}:vbv-maxrate={maxrate_value}')

    # 添加I帧和PB帧压缩率
    if FFMPEG["x264"].get("ipratio"):
        x264_params.append(f'ipratio={FFMPEG["x264"]["ipratio"]}')

    if FFMPEG["x264"].get("pbratio"):
        x264_params.append(f'pbratio={FFMPEG["x264"]["pbratio"]}')

    if x264_params:
        output_options.extend(['-x264-params', ':'.join(x264_params)])

    output_options.extend([
        '-threads', str(min(8, max(4, CPU_COUNT // 4))),  # 线程数
        '-f', 'flv',
        output_url
    ])

    common_options = input_options + latency_options + quality_options
    return {"cmd": base_cmd + common_options + output_options, "env": None}


def detect_stream(stream_url: str, output_url: str, stream_id: str, target_classes: List[str], conf_threshold: float,
                  gpu_id=None, show_label=False):
    model = None  # 初始化模型变量以便最后能清理它
    model_type = None  # 记录模型类型
    model_path = None  # 记录模型路径
    try:
        # 加载模型
        model_path = MODEL["path"]
        logger.info(f"Stream {stream_id} - 开始加载YOLO模型: {model_path}")
        
        # 根据文件扩展名识别模型类型
        if model_path.endswith('.onnx'):
            model_type = 'onnx'
            logger.info(f"Stream {stream_id} - 检测到ONNX模型")
        elif model_path.endswith('.pt'):
            model_type = 'pytorch'
            logger.info(f"Stream {stream_id} - 检测到PyTorch模型")
        else:
            model_type = 'unknown'
            logger.info(f"Stream {stream_id} - 未知模型类型: {model_path}")
        
        model = YOLO(model_path, task=MODEL["task"])
        logger.info(f"Stream {stream_id} - YOLO模型加载完成")

        # 检查模型预期的输入尺寸
        model_imgsz = MODEL["img_size"]  # 从配置获取输入尺寸
        logger.info(f"Stream {stream_id} - 模型输入尺寸设置为: {model_imgsz}x{model_imgsz}")

        # 确定设备参数 - ONNX模型不使用to()方法，而是在predict时指定device
        device = f'cuda:{gpu_id}' if gpu_id is not None and CUDA_AVAILABLE else 'cpu'
        logger.info(f"Stream {stream_id} - 将在 {device} 上运行推理")

        logger.info(f"Stream {stream_id} - 模型名称映射: {model.names}")
        classes_indices = [i for i, name in model.names.items() if name in target_classes]
        if not classes_indices:
            logger.error(f"Stream {stream_id} - 未找到任何有效的检测类别: {target_classes}")
            return

        logger.info(
            f"Stream {stream_id} - 初始化模型，检测类别: {target_classes}, 索引: {classes_indices}, 置信度: {conf_threshold}")

        cap = None
        process = None
        active_threads = []  # 跟踪创建的所有线程

        # 用于安全终止的事件标志
        stop_event = threading.Event()

        # 最大重连次数限制
        MAX_RECONNECT_COUNT = 3

        # 主处理循环，允许重连
        while stream_id in active_streams and active_streams[stream_id].reconnect_count < MAX_RECONNECT_COUNT:
            try:
                stream_info = active_streams[stream_id]
                
                # 检查是否请求停止
                if stream_info.status in ["stopped", "stopping"]:
                    logger.info(f"Stream {stream_id} - 收到停止请求，正在安全终止")
                    break
                
                stream_info.status = "connecting"

                logger.info(f"Stream {stream_id} - 正在连接视频流 (重连次数: {stream_info.reconnect_count})")

                cap = cv2.VideoCapture(stream_url)

                # 设置VideoCapture缓冲区大小
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 100)  # 最小缓冲区大小

                if not cap.isOpened():
                    stream_info.status = "error"
                    logger.error(f"Stream {stream_id} - 无法打开视频流: {stream_url}")
                    stream_info.reconnect_count += 1
                    if stream_info.reconnect_count >= MAX_RECONNECT_COUNT:
                        logger.error(f"Stream {stream_id} - 达到最大重连次数({MAX_RECONNECT_COUNT})，停止任务")
                        stream_info.status = "stopped"
                        break
                    time.sleep(1)  # 等待一段时间再重连
                    continue

                width, height, fps = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(
                    cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), max(15,
                                                             min(int(
                                                                 cap.get(
                                                                     cv2.CAP_PROP_FPS)),
                                                                 30))
                logger.info(f"Stream {stream_id} - 视频信息: {width}x{height} @ {fps}fps")
                stream_info.status = "streaming"

                ffmpeg_cmd = get_ffmpeg_command(output_url, width, height, fps)["cmd"]
                process = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE,
                                           stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                           bufsize=10 ** 6)  # 保持较小缓冲区

                # 使用配置中的队列大小
                frame_queue_size = VIDEO_PROCESSING["queue_size"]["frame"]
                processed_queue_size = VIDEO_PROCESSING["queue_size"]["processed"]
                frame_queue = queue.Queue(maxsize=frame_queue_size)
                processed_frame_queue = queue.Queue(maxsize=processed_queue_size)

                logger.info(
                    f"Stream {stream_id} - 队列大小设置: 输入帧={frame_queue_size}, 处理帧={processed_queue_size}")

                start_time, frame_count = time.time(), 0
                # 创建一个事件标志，用于指示流是否中断
                stream_interrupted = threading.Event()

                def read_frames():
                    nonlocal frame_count, cap
                    last_time = time.time()
                    frames_this_second = 0
                    last_capture_time = time.time()
                    consecutive_failures = 0

                    while (cap and cap.isOpened() and 
                           stream_id in active_streams and 
                           active_streams[stream_id].status == "streaming" and
                           not stop_event.is_set()):
                        # 检查是否应该停止
                        if stream_id in active_streams and active_streams[stream_id].status in ["stopping", "stopped"]:
                            logger.info(f"Stream {stream_id} - 读取线程检测到停止请求")
                            break
                            
                        # 废弃缓冲区中可能积累的旧帧
                        if time.time() - last_capture_time > 0.1:  # 如果超过100ms没有读取
                            for _ in range(3):  # 尝试清除积累的缓冲帧
                                if cap and cap.isOpened():
                                    cap.grab()

                        # 安全读取帧
                        if cap and cap.isOpened():
                            ret, frame = cap.read()
                            last_capture_time = time.time()

                            if not ret:
                                consecutive_failures += 1
                                logger.warning(
                                    f"Stream {stream_id} - 输入流读取失败 ({consecutive_failures}): {stream_url}")

                                # 如果连续3次读取失败，认为流已断开
                                if consecutive_failures >= 3:
                                    logger.error(f"Stream {stream_id} - 视频流断开连接")
                                    stream_interrupted.set()  # 设置流中断标志
                                    break

                                time.sleep(0.1)  # 短暂等待后重试
                                continue

                            # 成功读取，重置连续失败计数
                            consecutive_failures = 0

                            frame_count += 1
                            frames_this_second += 1
                            current_time = time.time()

                            # 计算当前秒的帧率
                            if current_time - last_time >= 1.0:
                                # 安全更新流信息
                                if stream_id in active_streams:
                                    active_streams[stream_id].fps = frames_this_second
                                frames_this_second = 0
                                last_time = current_time

                            # 动态跳帧逻辑
                            if stream_id in active_streams:
                                active_streams[stream_id].frame_skip_count += 1
                                input_fps = min(fps, 30)  # 限制输入帧率上限为30fps

                                # 使用配置中的动态跳帧设置
                                dynamic_dropping = VIDEO_PROCESSING.get("dynamic_frame_dropping", True)
                                skip_frame_ratio = VIDEO_PROCESSING.get("skip_frame_ratio", 4)
                                max_detection_fps = VIDEO_PROCESSING.get("max_detection_fps", 15)

                                # 计算基于输入帧率的跳帧比例
                                if dynamic_dropping and input_fps > max_detection_fps:
                                    # 更激进的跳帧，使检测帧率更低一些，但同时保持检测质量
                                    required_ratio = max(1, round(input_fps / (max_detection_fps - 5)))
                                    # 随着fps下降少跳一些帧
                                    dynamic_ratio = max(1, required_ratio - (30 - input_fps) // skip_frame_ratio)
                                    skip_ratio = dynamic_ratio
                                else:
                                    # 如果禁用动态跳帧或输入帧率较低，使用最小跳帧
                                    skip_ratio = 1

                                # 检查是否需要处理这一帧
                                process_this_frame = (active_streams[stream_id].frame_skip_count >= skip_ratio)

                                if process_this_frame:
                                    active_streams[stream_id].frame_skip_count = 0  # 重置跳帧计数
                                    active_streams[stream_id].frames_processed += 1

                                    # 如果队列即将满，清空队列并放入新帧
                                    if frame_queue.qsize() >= frame_queue.maxsize - 1:
                                        try:
                                            while not frame_queue.empty():
                                                frame_queue.get_nowait()
                                        except queue.Empty:
                                            pass

                                    try:
                                        frame_queue.put(frame, block=False)
                                    except queue.Full:
                                        # 如果队列满了，尝试丢弃一帧然后再放入
                                        try:
                                            frame_queue.get_nowait()
                                            frame_queue.put(frame, block=False)
                                        except:
                                            pass
                        else:
                            # cap不可用
                            logger.warning(f"Stream {stream_id} - 视频捕获对象不可用")
                            stream_interrupted.set()
                            break
                        
                        # 防止CPU过度占用
                        time.sleep(0.001)

                    logger.info(f"Stream {stream_id} - 读取线程结束")

                def process_frames():
                    while (stream_id in active_streams and 
                           active_streams[stream_id].status == "streaming" and 
                           not stop_event.is_set()):
                        # 检查是否应该停止
                        if stream_id in active_streams and active_streams[stream_id].status in ["stopping", "stopped"]:
                            logger.info(f"Stream {stream_id} - 处理线程检测到停止请求")
                            break
                            
                        try:
                            # 使用更短的超时时间减少等待
                            frame = frame_queue.get(timeout=0.05)

                            # 如果处理队列已经很大，则不再处理新帧
                            if processed_frame_queue.qsize() >= processed_frame_queue.maxsize - 1:
                                continue

                            # 使用当前设备处理帧
                            processed = process_frame(model, frame, conf_threshold, target_classes, show_label, device)
                            if processed is not None:
                                # 如果输出队列即将满，清空队列
                                if processed_frame_queue.qsize() >= processed_frame_queue.maxsize - 1:
                                    try:
                                        while not processed_frame_queue.empty():
                                            processed_frame_queue.get_nowait()
                                    except queue.Empty:
                                        pass

                                try:
                                    processed_frame_queue.put(processed, block=False)
                                except queue.Full:
                                    # 如果队列满了，尝试丢弃一帧然后再放入
                                    try:
                                        processed_frame_queue.get_nowait()
                                        processed_frame_queue.put(processed, block=False)
                                    except:
                                        pass
                        except queue.Empty:
                            continue
                        except queue.Full:
                            continue
                        except Exception as e:
                            logger.error(f"Stream {stream_id} - 处理帧错误: {str(e)}")
                        
                        # 防止CPU过度占用
                        time.sleep(0.001)
                        
                    logger.info(f"Stream {stream_id} - 处理线程结束")

                def write_frames():
                    while (stream_id in active_streams and 
                           active_streams[stream_id].status == "streaming" and 
                           not stop_event.is_set()):
                        # 检查是否应该停止
                        if stream_id in active_streams and active_streams[stream_id].status in ["stopping", "stopped"]:
                            logger.info(f"Stream {stream_id} - 写入线程检测到停止请求")
                            break
                        
                        try:
                            if process is None or process.poll() is not None:
                                logger.error(f"Stream {stream_id} - FFmpeg进程已终止")
                                break

                            # 使用更短的超时时间
                            frame = processed_frame_queue.get(timeout=0.05)
                            if process and process.stdin:
                                try:
                                    process.stdin.write(frame.tobytes())
                                    process.stdin.flush()
                                except (BrokenPipeError, IOError) as e:
                                    logger.error(f"Stream {stream_id} - 写入帧错误: {str(e)}")
                                    break
                        except queue.Empty:
                            continue
                        except Exception as e:
                            logger.error(f"Stream {stream_id} - 推流错误: {str(e)}")
                        
                        # 防止CPU过度占用
                        time.sleep(0.001)
                        
                    logger.info(f"Stream {stream_id} - 写入线程结束")

                def log_ffmpeg_output():
                    ffmpeg_log = []
                    while (process and process.poll() is None and 
                           stream_id in active_streams and 
                           active_streams[stream_id].status == "streaming" and
                           not stop_event.is_set()):
                        # 检查是否应该停止
                        if stream_id in active_streams and active_streams[stream_id].status in ["stopping", "stopped"]:
                            logger.info(f"Stream {stream_id} - 日志线程检测到停止请求")
                            break
                            
                        try:
                            if process and process.stderr:
                                line = process.stderr.readline().decode().strip()
                                if line:
                                    ffmpeg_log.append(line)
                                    logger.info(f"Stream {stream_id} - FFmpeg: {line}")
                                if "No capable devices found" in line or "OpenEncodeSessionEx failed" in line:
                                    logger.error(f"Stream {stream_id} - NVENC 失败，尝试重启流")
                                    process.terminate()
                                    stream_interrupted.set()  # 设置流中断标志
                                    break
                            else:
                                # FFmpeg进程不可用
                                break
                        except Exception as e:
                            logger.error(f"Stream {stream_id} - 读取FFmpeg日志错误: {str(e)}")
                            break
                            
                        # 防止CPU过度占用
                        time.sleep(0.001)
                        
                    logger.info(f"Stream {stream_id} - 日志线程结束")

                # 创建并启动所有线程，确保它们设置为守护线程
                read_thread = Thread(target=read_frames, daemon=True)
                process_thread = Thread(target=process_frames, daemon=True)
                write_thread = Thread(target=write_frames, daemon=True)
                log_thread = Thread(target=log_ffmpeg_output, daemon=True)
                
                active_threads.extend([read_thread, process_thread, write_thread, log_thread])
                read_thread.start()
                process_thread.start() 
                write_thread.start()
                log_thread.start()

                # 等待流中断或者被主动停止
                while not stream_interrupted.is_set() and stream_id in active_streams:
                    # 检查是否需要停止
                    if stream_id in active_streams and active_streams[stream_id].status in ["stopping", "stopped"]:
                        logger.info(f"Stream {stream_id} - 主循环检测到停止请求")
                        break
                    time.sleep(0.5)

                # 如果流被中断且未达到最大重连次数，尝试重连
                if stream_interrupted.is_set() and stream_id in active_streams and active_streams[stream_id].status != "stopping":
                    active_streams[stream_id].reconnect_count += 1
                    logger.warning(f"Stream {stream_id} - 视频流中断，准备重连 (第{active_streams[stream_id].reconnect_count}次重连)")

                # 安全清理当前会话的资源
                stop_event.set()  # 通知所有线程停止
                
                # 等待所有线程安全结束 - 设置超时避免无限等待
                for thread in active_threads:
                    if thread.is_alive():
                        thread.join(timeout=2.0)
                active_threads.clear()
                
                # 释放资源
                if cap:
                    try:
                        cap.release()
                        cap = None
                    except Exception as e:
                        logger.error(f"Stream {stream_id} - 释放视频捕获资源错误: {str(e)}")
                
                if process:
                    try:
                        # 安全关闭FFmpeg进程
                        if process.stdin:
                            try:
                                process.stdin.close()
                            except:
                                pass
                        
                        if process.poll() is None:  # 如果进程仍在运行
                            try:
                                process.terminate()
                                process.wait(timeout=3)
                            except:
                                # 如果进程无法正常终止，强制杀死
                                try:
                                    process.kill()
                                except:
                                    pass
                        process = None
                    except Exception as e:
                        logger.error(f"Stream {stream_id} - 释放FFmpeg进程错误: {str(e)}")

                # 如果达到最大重连次数或被请求停止，停止流
                if (stream_id not in active_streams or 
                    (stream_id in active_streams and 
                     (active_streams[stream_id].reconnect_count >= MAX_RECONNECT_COUNT or 
                      active_streams[stream_id].status in ["stopping", "stopped"]))):
                    logger.info(f"Stream {stream_id} - 停止任务")
                    if stream_id in active_streams:
                        active_streams[stream_id].status = "stopped"
                    break

                # 如果流被主动停止，退出循环
                if stream_id not in active_streams:
                    break

                # 等待一段时间再重连
                time.sleep(2)

            except Exception as e:
                logger.error(f"Stream {stream_id} - 处理失败: {str(e)}")
                if stream_id in active_streams:
                    active_streams[stream_id].reconnect_count += 1
                    if active_streams[stream_id].reconnect_count >= MAX_RECONNECT_COUNT:
                        logger.error(f"Stream {stream_id} - 达到最大重连次数({MAX_RECONNECT_COUNT})，停止任务")
                        active_streams[stream_id].status = "stopped"
                        break
                    logger.warning(f"Stream {stream_id} - 将尝试重连 (第{active_streams[stream_id].reconnect_count}次)")
                    time.sleep(2)  # 等待一段时间再重连
                else:
                    break

        # 最终清理
        stop_event.set()  # 确保所有线程收到停止信号
        
        # 等待所有线程结束
        for thread in active_threads:
            if thread.is_alive():
                thread.join(timeout=2.0)
                
        if stream_id in active_streams:
            if stream_info.gpu_id is not None:
                device = f'cuda:{stream_info.gpu_id}' if CUDA_AVAILABLE else None
                # 清理GPU资源
                free_gpu_memory(model, device, model_type, model_path)
                release_gpu(stream_info.gpu_id)
                
            # 如果流仍然处于stopping状态，将其设置为finished
            if stream_id in active_streams and active_streams[stream_id].status == "stopping":
                active_streams[stream_id].status = "finished"
                logger.info(f"Stream {stream_id} - 任务完成并释放资源")

    except Exception as e:
        logger.error(f"Stream {stream_id} - 加载模型失败: {str(e)}")
        if stream_id in active_streams:
            active_streams[stream_id].status = "error"
    
    finally:
        # 最终清理 - 确保无论如何都会释放GPU资源
        if stream_id in active_streams:
            if stream_info.gpu_id is not None:
                device = f'cuda:{stream_info.gpu_id}' if CUDA_AVAILABLE else None
                # 清理GPU资源
                free_gpu_memory(model, device, model_type, model_path)
                release_gpu(stream_info.gpu_id)
                
            # 如果流仍然处于stopping状态，将其设置为finished
            if stream_id in active_streams and active_streams[stream_id].status == "stopping":
                active_streams[stream_id].status = "finished"
                logger.info(f"Stream {stream_id} - 任务完成并释放资源")


@app.post("/process_stream", response_model=Dict[str, str])
async def process_stream_endpoint(request: StreamRequest, background_tasks: BackgroundTasks):
    if len(active_streams) >= VIDEO_PROCESSING["max_streams"]:
        raise HTTPException(status_code=429, detail="Too many streams")

    stream_id = f"{uuid.uuid4().hex[:12]}"
    target_classes = [cls.strip() for cls in request.names_dict.split(',')]
    conf_threshold = request.confidence_threshold
    show_label = request.confidence_display  # 获取是否显示标签的参数

    # 选择GPU卡号
    selected_gpu = None
    if CUDA_AVAILABLE and GPU_COUNT > 0:
        # 1. 优先使用用户指定的GPU卡号
        if request.gpu_id is not None:
            if request.gpu_id >= 0 and request.gpu_id < GPU_COUNT:
                selected_gpu = request.gpu_id
                # 更新GPU使用计数
                gpu_usage[selected_gpu] += 1
                logger.info(f"Stream {stream_id} - 使用用户指定的GPU {selected_gpu}, 当前使用计数: {gpu_usage}")
            else:
                logger.warning(f"Stream {stream_id} - 指定的GPU {request.gpu_id} 不存在，将使用配置的默认GPU")
                selected_gpu = MODEL.get("default_gpu")
                if selected_gpu is not None:
                    gpu_usage[selected_gpu] += 1
                    logger.info(f"Stream {stream_id} - 使用配置的默认GPU {selected_gpu}, 当前使用计数: {gpu_usage}")
        # 2. 其次使用配置文件中的默认GPU
        elif MODEL.get("default_gpu") is not None:
            selected_gpu = MODEL["default_gpu"]
            gpu_usage[selected_gpu] += 1
            logger.info(f"Stream {stream_id} - 使用配置的默认GPU {selected_gpu}, 当前使用计数: {gpu_usage}")
        # 3. 最后使用自动选择策略
        else:
            selected_gpu = select_gpu(stream_id)
    
    output_url = f"rtmp://{STREAM_SERVER['srs_server']}:{STREAM_SERVER['rtmp_port']}/live/{stream_id}"
    hls_url = f"http://{STREAM_SERVER['srs_server']}:{STREAM_SERVER['hls_port']}/live/{stream_id}.m3u8"
    web_rtc = f"webrtc://{STREAM_SERVER['srs_server']}/live/{stream_id}"

    active_streams[stream_id] = StreamInfo(
        stream_id=stream_id,
        stream_url=request.stream_url,
        detecting_classes=",".join(target_classes),
        status="starting",
        start_time=time.time(),
        playback_url=hls_url,
        web_rtc=web_rtc,
        gpu_id=selected_gpu,
        reconnect_count=0  # 初始化重连计数
    )

    background_tasks.add_task(
        stream_executor.submit,
        detect_stream,
        request.stream_url,
        output_url,
        stream_id,
        target_classes,
        conf_threshold,
        selected_gpu,
        show_label  # 传递是否显示标签的参数
    )

    logger.info(f"Stream {stream_id} - 请求启动，检测类别: {target_classes}, 置信度阈值: {conf_threshold}")
    logger.info(f"Active streams: {len(active_streams)}")
    return {
        "status": "success",
        "message": "流处理已启动",
        "playback_url": hls_url,
        "web_rtc": web_rtc,
        "stream_id": stream_id,
        "detecting_classes": ",".join(target_classes),
        "confidence_threshold": str(conf_threshold),
        "encoder": "CPU (libx264)",  # 统一使用CPU编码
        "detector": f"YOLO on {'GPU ' + str(selected_gpu) if selected_gpu is not None else 'CPU'}"
    }


@app.post("/stop_stream", response_model=Dict[str, str])
async def stop_stream_endpoint(request: StreamIDRequest):
    """停止指定ID的视频流检测任务，并确保释放GPU资源"""
    stream_id = request.stream_id
    
    if stream_id not in active_streams:
        raise HTTPException(status_code=404, detail=f"Stream ID {stream_id} not found")
    
    stream_info = active_streams[stream_id]
    
    # 如果流已经处于停止或错误状态，直接返回
    if stream_info.status in ["stopped", "finished", "error"]:
        return {
            "status": "success",
            "message": f"流 {stream_id} 已经处于 {stream_info.status} 状态"
        }
    
    # 将流状态设置为停止，主循环会检测到这个状态并终止流处理
    try:
        # 使用防护策略来设置状态，防止并发问题
        logger.info(f"Stream {stream_id} - 请求停止")
        stream_info.status = "stopping"  # 首先设置为正在停止状态
        
        # 如果流使用了GPU，立即释放GPU资源
        if stream_info.gpu_id is not None:
            gpu_id = stream_info.gpu_id
            logger.info(f"Stream {stream_id} - 正在强制释放GPU {gpu_id} 资源")
            
            # 同步立即清理CUDA缓存
            if CUDA_AVAILABLE:
                try:
                    with torch.cuda.device(gpu_id):
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                except Exception as e:
                    logger.error(f"初步清理GPU缓存时出错: {str(e)}")
            
            # 更新GPU使用计数
            release_gpu(gpu_id)
            
        # 返回成功信息
        return {
            "status": "success",
            "message": f"流 {stream_id} 正在停止并已清理GPU资源",
            "stream_id": stream_id
        }
    except Exception as e:
        logger.error(f"Stream {stream_id} - 停止过程中发生错误: {str(e)}")
        return {
            "status": "error",
            "message": f"停止流时发生错误: {str(e)}",
            "stream_id": stream_id
        }


@app.get("/list_streams", response_model=Dict[str, List[Dict[str, str]]])
async def list_streams_endpoint():
    """列出所有当前活动的视频流"""
    streams_list = []
    
    for stream_id, stream_info in active_streams.items():
        # 计算运行时间
        runtime = time.time() - stream_info.start_time
        
        streams_list.append({
            "stream_id": stream_id,
            "status": stream_info.status,
            "classes": stream_info.detecting_classes,
            "runtime": f"{runtime:.2f}秒",
            "fps": f"{stream_info.fps:.2f}",
            "frames_processed": str(stream_info.frames_processed),
            "playback_url": stream_info.playback_url,
            "web_rtc": stream_info.web_rtc,
            "gpu": str(stream_info.gpu_id) if stream_info.gpu_id is not None else "CPU"
        })
    
    return {
        "streams": streams_list
    }




# 主程序入口
if __name__ == "__main__":
    # 解析命令行参数
    args = parse_args()

    # 加载配置
    if args.config_file != 'config.py':
        load_config(args.config_file)
    
    # 如果命令行指定了GPU卡号，则更新MODEL配置
    if args.gpu is not None and CUDA_AVAILABLE:
        if args.gpu >= 0 and args.gpu < GPU_COUNT:
            MODEL["default_gpu"] = args.gpu
            print(f"使用命令行指定的默认GPU: {args.gpu}")
        else:
            print(f"警告: 指定的GPU {args.gpu} 不存在，将使用配置文件中的设置")

    # 初始化应用
    logger = init_app()

    # 如果启用了ONNX Runtime，配置执行提供程序
    if MODEL.get("onnx_runtime", {}).get("enabled", False):
        try:
            import onnxruntime as ort

            # 配置ONNX Runtime会话选项
            ort_options = ort.SessionOptions()
            ort_options.inter_op_num_threads = MODEL["onnx_runtime"].get("inter_op_num_threads", CPU_COUNT)
            ort_options.intra_op_num_threads = MODEL["onnx_runtime"].get("intra_op_num_threads", CPU_COUNT)

            # 设置图优化级别
            if MODEL["onnx_runtime"].get("graph_optimization_level"):
                opt_level = getattr(ort.GraphOptimizationLevel, MODEL["onnx_runtime"]["graph_optimization_level"])
                ort_options.graph_optimization_level = opt_level

            # 配置内存设置以降低延迟
            ort_options.enable_mem_pattern = MODEL["onnx_runtime"].get("enable_mem_pattern", True)
            ort_options.enable_cpu_mem_arena = MODEL["onnx_runtime"].get("enable_cpu_mem_arena", True)

            # 选择执行提供程序
            use_cuda = MODEL["onnx_runtime"].get("use_cuda", False) and CUDA_AVAILABLE
            use_tensorrt = MODEL["onnx_runtime"].get("use_tensorrt", False) and CUDA_AVAILABLE

            if use_tensorrt:
                providers = MODEL["onnx_runtime"]["providers"].get("tensorrt",
                                                                   ["TensorrtExecutionProvider",
                                                                    "CUDAExecutionProvider", "CPUExecutionProvider"])
            elif use_cuda:
                providers = MODEL["onnx_runtime"]["providers"].get("cuda",
                                                                   ["CUDAExecutionProvider", "CPUExecutionProvider"])
            else:
                providers = MODEL["onnx_runtime"]["providers"].get("cpu", ["CPUExecutionProvider"])

            # 设置为环境变量，让ONNX Runtime使用
            os.environ["OMP_NUM_THREADS"] = str(CPU_COUNT)
            os.environ["OMP_WAIT_POLICY"] = "ACTIVE"

            logger.info(
                f"ONNX Runtime已配置: 提供程序={providers}, 优化级别={MODEL['onnx_runtime']['graph_optimization_level']}")
        except ImportError:
            logger.warning("无法导入onnxruntime，将使用默认推理设置")

    # 使用配置中的服务器设置
    uvicorn.run(app,
                host=SERVER["host"],
                port=SERVER["port"],
                workers=SERVER["workers"])
