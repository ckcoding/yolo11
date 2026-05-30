import asyncio
import json
import time
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict

from app.core.storage import persist_bytes

# 建立 5 个线程规模的后台上传线程池，防止 MinIO 抖动锁死画帧主进程
executor = ThreadPoolExecutor(max_workers=5)

# Websocket 连接池：专门托管正在连接到对应 task_id 进行监听的前端客端
active_websockets = defaultdict(list)
# 每个 task 最大允许的 WebSocket 连接数，防止恶意连接导致资源耗尽
MAX_WS_PER_TASK = 50

# 全局异步事件循环对象（用于跨越物理核心从线程池逆向传参给 WebSocket 的协程）
main_loop = None

def _store_snapshot_worker(task_id, labels, img_bytes, timestamp):
    """完全隔离在次级工线程中打工的底层函数，阻塞也不会影响推流"""
    url = ""
    try:
        stored = persist_bytes("snapshots", img_bytes, f"{task_id}_{timestamp}", ".jpg", "image/jpeg")
        url = stored["url"]
    except Exception as exc:
        print(f"[Events] 抓拍图片落盘失败: {exc}")

    # 不管落盘成没成功，都要告知所有关注该视角直播流的前端：“有活靶子出现了”
    if main_loop and active_websockets.get(task_id):
        msg = {
            "type": "snapshot",
            "task_id": task_id,
            "time": timestamp,
            "labels": list(labels),
            "snapshot_url": url,
            "message": "AI发现指定侦测目标"
        }
        msg_str = json.dumps(msg)
        
        # 将发送信号强行抛入 fastapi 老家的真·原生事件循环体内排队（由于处于独立 OS 线程，这是唯一且最安全的办法）
        main_loop.call_soon_threadsafe(_notify_websockets_sync, task_id, msg_str)


def _notify_websockets_sync(task_id, msg_str):
    """由 call_soon_threadsafe 从主循环拉起的桥接发射器，它会立刻无缝创建一个极轻量原生协程来遍历广播"""
    async def _send():
        dead_clients = []
        for ws in active_websockets.get(task_id, []):
            try:
                await ws.send_text(msg_str)
            except Exception:
                dead_clients.append(ws)
        # 挥刀杀掉早已断网挂起的僵尸前端客源（避免堆积泄露）
        for ws in dead_clients:
            active_websockets[task_id].remove(ws)
            
    asyncio.create_task(_send())


def dispatch_snapshot_event(task_id: str, labels: set, img_bytes: bytes):
    """
    暴露给 AI Core 引擎的高层 API
    内部直接交给 Pool 处理，0 等待秒返回不带走一片云彩
    """
    timestamp = int(time.time())
    executor.submit(_store_snapshot_worker, task_id, labels, img_bytes, timestamp)

def _notify_websockets_binary(task_id, img_bytes):
    """二进制帧发送桥接器：用于 WebSocket 直传 JPEG 字节，零文本编解码"""
    async def _send():
        dead_clients = []
        for ws in active_websockets.get(task_id, []):
            try:
                await ws.send_bytes(img_bytes)
            except Exception:
                dead_clients.append(ws)
        for ws in dead_clients:
            active_websockets[task_id].remove(ws)
    asyncio.create_task(_send())


def dispatch_preview_frame(task_id: str, img_bytes: bytes):
    """
    供 websocket 模式下直接发送二进制 JPEG 帧数据。
    相比 Base64 文本传输，带宽占用降低约 40-50%，同时省去编解码 CPU 开销。
    """
    if main_loop and active_websockets.get(task_id):
        main_loop.call_soon_threadsafe(_notify_websockets_binary, task_id, img_bytes)
