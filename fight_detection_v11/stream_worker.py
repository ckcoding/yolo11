import cv2
import time
import base64
import queue
import threading
import logging
import uuid
import os

# 必须在任何 cv2.VideoCapture 调用之前就设定好，否则不生效
os.environ["OPENCV_FFMPEG_READ_ATTEMPTS"] = "65536"
from detector import Detector
from hardware_manager import HardwareManager
from minio_client import minio_uploader
from config import config

logger = logging.getLogger(__name__)

# 全局硬件管理器实例
hw_manager = HardwareManager(override_max_gpu=config.max_gpu_streams, vram_per_stream=config.vram_per_stream)

class StreamWorker:
    def __init__(self, task_id, params):
        self.task_id = task_id
        self.stream_url = params.stream_url
        
        # 支持多模型：将逗号隔开的字符串转为列表
        self.model_names = [m.strip() for m in params.model_name.split(",")] if "," in params.model_name else [params.model_name]
        self.names_dicts = [n.strip() for n in params.names_dict.split(",")] if "," in params.names_dict else [params.names_dict]
        
        # 如果模型数和过滤词组数不匹配，进行补齐（通常建议一一对应）
        if len(self.names_dicts) < len(self.model_names):
            self.names_dicts.extend([self.names_dicts[-1]] * (len(self.model_names) - len(self.names_dicts)))
            
        self.alias = params.alias
        self.conf_thres = params.conf_thres
        self.confidence_display = params.confidence_display
        self.inference_fps = getattr(params, 'inference_fps', 10) 
        self.fight_detection = getattr(params, 'fight_detection', False) # 新增
        
        self.q = queue.Queue(maxsize=30)  # 缓冲队列，最多保留 30 帧避免堆积
        self.stop_event = threading.Event()
        self.inference_thread = None
        self.reader_thread = None
        
        self.device = None
        self.detectors = [] # 改为列表存储多个检测器
        self.is_running = False
        
        # 用于多线程剥离读取，永远存放热乎的最新帧，避免 OpenCV 内部排队引发延迟
        self.latest_frame = None
        self.frame_ready = threading.Event()
        self.cap = None

    def start(self):
        """分配设备并启动推流检测线程"""
        # allocate_device() 现在直接返回具体设备名如 'cuda:0', 'cuda:1', 'cpu'
        self.device = hw_manager.allocate_device()
        logger.info(f"开启融合流调度 {self.task_id}，目标模型组 {self.model_names} -> 派送往: {self.device}")
        
        # 为每个模型初始化独立的检测引擎
        self.detectors = []
        for m_name in self.model_names:
            self.detectors.append(Detector(model_name=m_name, device=self.device))
        
        # 启动视频抽取子线程和推理解算主线程
        self.is_running = True
        self.reader_thread = threading.Thread(target=self._frame_reader, daemon=True)
        self.inference_thread = threading.Thread(target=self._run_inference, daemon=True)
        
        self.reader_thread.start()
        self.inference_thread.start()

    def stop(self):
        """停止检测任务，释放相关硬件资源占用"""
        self.stop_event.set()
        self.is_running = False
        
        # 等待线程安全退出
        if self.reader_thread and self.reader_thread.is_alive():
            self.reader_thread.join(timeout=2.0)
        if self.inference_thread and self.inference_thread.is_alive():
            self.inference_thread.join(timeout=2.0)
        
        # 安全释放 OpenCV 资源
        if self.cap:
            try:
                self.cap.release()
            except Exception as e:
                logger.warning(f"释放视频捕获器时出现异常: {e}")
            finally:
                self.cap = None
        
        # 释放硬件资源
        if self.device:
            hw_manager.release_device(self.device)
            self.device = None

    def _frame_reader(self):
        """独立网络拉流抓取器"""
        logger.info(f"开启抽取视频流后台线程: {self.stream_url}")
        
        try:
            self.cap = cv2.VideoCapture(self.stream_url)
            # 尝试设定缓存区大小（并不是所有解码栈都支持，但聊胜于无）
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            if not self.cap.isOpened():
                logger.error(f"无法首发打开视频流地址: {self.stream_url}")
                self.stop()
                return
                
            retry_count = 0
                
            while not self.stop_event.is_set():
                ret, frame = self.cap.read()
                if not ret:
                    retry_count += 1
                    # 遭遇了暂时性的网络掉帧或短暂脱机，不要立刻终止任务而是进入重试心跳侦测
                    if retry_count > 60:
                        # 如果持续超过大概 6 秒还没抢通信号，判定流源确认死亡，再执行终止大发
                        logger.warning(f"流媒体 {self.task_id} 视频连接由于长时间网络断线信号消亡而主动挂断。")
                        self.latest_frame = None
                        self.stop()
                        break
                    else:
                        if retry_count % 15 == 0:
                            logger.warning(f"监控到 OpenCV 流获取受阻({retry_count}/60)，尝试暴力重启视频捕获拉流器...")
                            # 强行重置通道对象句柄，往往能瞬间解决花屏或者卡死
                            if self.cap:
                                try:
                                    self.cap.release()
                                except:
                                    pass
                            self.cap = cv2.VideoCapture(self.stream_url)
                            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                            
                        time.sleep(0.1) # 休眠等待网络信道缓和，然后继续重试
                        continue
                
                # 抓取帧成功，网络依然健康通畅，清空警报探针计数
                retry_count = 0
                
                # 持续拿到最新一帧，覆盖老帧
                self.latest_frame = frame
                self.frame_ready.set()
                # 稍微给出一个极其微弱的休眠避免此原生轮询线程完全霸占单核算力
                time.sleep(0.005)
        except Exception as e:
            logger.error(f"视频流读取线程异常退出 {self.task_id}: {e}")
        finally:
            # 确保资源被释放
            if self.cap:
                try:
                    self.cap.release()
                except:
                    pass

    def _run_inference(self):
        """独立守护线程：负责耗时的推理解算，推入缓冲队列并异步定时上传 MinIO"""
        
        # 【算力控制节阀】提升到 20 FPS 以获得更流畅的实时体验（每 50ms 一帧）
        # 如果需要节省算力，可以降低到 10 FPS (frame_interval = 1.0 / 10.0)
        last_minio_upload = time.time()
        while not self.stop_event.is_set():
            # 动态计算频率 (V2.2 性能优化版)
            frame_interval = 1.0 / max(1, self.inference_fps)
            start_time = time.time()
            
            # 等待上一级抓帧线程输送新的一批解算任务
            if not self.frame_ready.wait(timeout=0.5):
                continue
                
            frame = self.latest_frame
            self.frame_ready.clear() # 处理完毕清空旗标
            
            if frame is None:
                continue
                
            # 核心升级：画面流水线。依次穿过多个模型进行识别与加框
            annotated_frame = frame
            detected_fight = False
            for idx, detector in enumerate(self.detectors):
                # 接收打架识别结果标志
                annotated_frame, is_fight = detector.process_frame(
                    annotated_frame, 
                    names_dict=self.names_dicts[idx], 
                    conf_thres=self.conf_thres, 
                    confidence_display=self.confidence_display,
                    fight_detection=self.fight_detection
                )
                if is_fight:
                    detected_fight = True
            
            # 将处理后的截图数据转换为 base64 格式
            _, buffer = cv2.imencode('.jpg', annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            frame_bytes = buffer.tobytes()
            b64_frame = base64.b64encode(frame_bytes).decode('utf-8')
            
            # 推入队列
            if self.q.full():
                try:
                    self.q.get_nowait()
                except queue.Empty:
                    pass
            self.q.put_nowait(b64_frame)
            
            # --- MinIO 存储策略 ---
            # 逻辑：如果开启了打架检测，则只有检测到打架才存。否则按 2 秒一次存。
            should_upload = False
            if self.fight_detection:
                if detected_fight:
                    # 降低打架抓拍频率，防止瞬间存太多，比如 0.5 秒存一张
                    if time.time() - last_minio_upload > 0.5:
                        should_upload = True
            else:
                # 普通模式，2 秒一张
                if time.time() - last_minio_upload > 2.0:
                    should_upload = True

            if should_upload:
                filename = f"{self.task_id}/{int(time.time())}.jpg"
                threading.Thread(target=minio_uploader.upload_image, args=(frame_bytes, filename), daemon=True).start()
                last_minio_upload = time.time()
                
            # 限制帧率，给系统的其他网络线程预留切片
            elapsed = time.time() - start_time
            sleep_time = max(0, frame_interval - elapsed)
            time.sleep(sleep_time)
            
        logger.info(f"检测任务推理解算线程 {self.task_id} 已安全退出收尾。")
