import os
import cv2
import numpy as np
import logging
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO
from model_manager import model_manager
from config import config, CLASS_CHINESE

logger = logging.getLogger(__name__)

class Detector:
    def __init__(self, model_name: str, device: str):
        # 将字符串如 'cuda:1' 转化为具体的 torch.device 对象
        import torch
        self.device = torch.device(device) if device.startswith("cuda") else torch.device("cpu")
        self.model = model_manager.get_model(model_name, device)
        
        self._last_names_dict = None
        self._cached_filters = []
        
        # 打斗识别相关的缓存逻辑
        self.track_history = {} # {id: {"keypoints": [], "box": []}}
        self.fight_status = {}  # {id: bool}
        self.max_history = 10   # 记忆过去10帧的动作，计算瞬时加速度
        
        # 预载字体
        os.makedirs(config.fonts_dir, exist_ok=True)
        self.font = None
        for font_candidate in os.listdir(config.fonts_dir):
            if font_candidate.lower().endswith(".ttf"):
                self.font_path = os.path.join(config.fonts_dir, font_candidate)
                try:
                    self.font = ImageFont.truetype(self.font_path, 20)
                    break 
                except:
                    continue

    def draw_text(self, img_np, text, position, color=(0, 255, 0)):
        if self.font:
            img_pil = Image.fromarray(cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(img_pil)
            rgb_color = tuple(reversed(color)) if isinstance(color, (list, tuple)) else color
            draw.text(position, text, font=self.font, fill=rgb_color)
            return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        else:
            cv2.putText(img_np, text, position, cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            return img_np

    def process_frame(self, frame, names_dict=None, conf_thres=0.5, confidence_display=False, fight_detection=False):
        """核心推理解算：支持按需开启打架行为逻辑"""
        
        is_pose_model = hasattr(self.model, 'task') and self.model.task == 'pose'
        current_model = self.model
        if fight_detection and not is_pose_model:
            try:
                current_model = model_manager.get_model("yolo11n-pose.pt", str(self.device))
                is_pose_model = True
            except:
                pass

        results = current_model.track(frame, conf=conf_thres, device=self.device, verbose=False, persist=True)
        result = results[0]
        boxes = result.boxes
        keypoints = getattr(result, 'keypoints', None) if is_pose_model else None
        
        if names_dict != self._last_names_dict:
            self._last_names_dict = names_dict
            self._cached_filters = [f.strip().lower() for f in names_dict.split(',') if f.strip()] if names_dict and names_dict.lower() not in ["all", ""] else []

        annotated_frame = frame.copy()
        
        current_persons = []
        if boxes is not None and boxes.id is not None:
            for i, box in enumerate(boxes):
                obj_id = int(box.id[0])
                cls = int(box.cls[0])
                name = result.names[cls]
                
                if name.lower() != 'person':
                    xyxy = box.xyxy[0].cpu().numpy().astype(int)
                    color_rgb = config.class_colors.get(name.lower(), [0, 255, 0])
                    color_tuple = tuple(color_rgb[::-1])
                    cv2.rectangle(annotated_frame, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), color_tuple, 2)
                    continue
                
                xyxy = box.xyxy[0].cpu().numpy().astype(int)
                conf = float(box.conf[0])
                kp = keypoints[i].cpu().numpy() if keypoints is not None else None
                current_persons.append({"id": obj_id, "xyxy": xyxy, "conf": conf, "keypoints": kp})

        if not fight_detection:
            for p in current_persons:
                x1, y1, x2, y2 = p["xyxy"]
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            return annotated_frame, False # 返回帧和识别结果

        # --- 精英版打架判定逻辑 ---
        active_ids = [p["id"] for p in current_persons]
        for p in current_persons:
            pid = p["id"]
            if pid not in self.track_history: self.track_history[pid] = []
            center = [(p["xyxy"][0] + p["xyxy"][2])/2, (p["xyxy"][1] + p["xyxy"][3])/2]
            feature = p["keypoints"].data[0] if p["keypoints"] is not None else np.array([center])
            self.track_history[pid].append(feature)
            if len(self.track_history[pid]) > self.max_history: self.track_history[pid].pop(0)

        # 清理旧 ID
        expired_ids = [pid for pid in self.track_history if pid not in active_ids]
        for pid in expired_ids:
            self.track_history.pop(pid, None)
            self.fight_status.pop(pid, None)

        any_fight = False
        for i, p1 in enumerate(current_persons):
            is_fighting = False
            for j, p2 in enumerate(current_persons):
                if i == j: continue
                
                # 1. 距离压缩判定 (更严格的距离)
                c1 = np.array([(p1["xyxy"][0]+p1["xyxy"][2])/2, (p1["xyxy"][1]+p1["xyxy"][3])/2])
                c2 = np.array([(p2["xyxy"][0]+p2["xyxy"][2])/2, (p2["xyxy"][1]+p2["xyxy"][3])/2])
                dist = np.linalg.norm(c1 - c2)
                box_w = (p1["xyxy"][2] - p1["xyxy"][0])
                
                # 在无人机视角下，打架几乎总是重叠的
                if dist < box_w * 0.9: 
                    h1 = self.track_history.get(p1["id"], [])
                    h2 = self.track_history.get(p2["id"], [])
                    if len(h1) >= 5 and len(h2) >= 5:
                        # 2. 计算相对速度差 (分析是否同步移动)
                        v1 = np.diff(np.array(h1[-5:]), axis=0).mean(axis=0)
                        v2 = np.diff(np.array(h2[-5:]), axis=0).mean(axis=0)
                        
                        # 如果是平行走路，v1 和 v2 高度一致，其范数差异很小
                        # 打架时，两人的移动方向和速度往往是剧烈对抗/混乱的
                        relative_v = np.linalg.norm(v1 - v2)
                        
                        # 3. 统计爆发性能量 (Std 代表混乱度)
                        m1 = np.std(np.diff(np.array(h1[-5:]), axis=0))
                        m2 = np.std(np.diff(np.array(h2[-5:]), axis=0))
                        
                        # 高强度混乱判定
                        if (m1 + m2 > 25.0) and (relative_v > 5.0):
                            is_fighting = True
                            any_fight = True
                            break
            self.fight_status[p1["id"]] = is_fighting

        # --- 渲染阶段 ---
        for p in current_persons:
            pid = p["id"]
            x1, y1, x2, y2 = p["xyxy"]
            is_fight = self.fight_status.get(pid, False)
            color = (0, 0, 255) if is_fight else (0, 255, 0)
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 4 if is_fight else 2)
            if confidence_display or is_fight:
                label = f"ID:{pid}" + (" [FIGHTING!]" if is_fight else "")
                annotated_frame = self.draw_text(annotated_frame, label, (x1, y1 - 35), color)
            if p["keypoints"] is not None:
                for kx, ky, kconf in p["keypoints"].data[0]:
                    if kconf > 0.5: cv2.circle(annotated_frame, (int(kx), int(ky)), 2, color, -1)

        return annotated_frame, any_fight



