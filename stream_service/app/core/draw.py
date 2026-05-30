import random
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os
from app.config import FONT_PATH, get_label_color

class_colors = {}

def get_color_for_class(class_name: str):
    """自动给类目分配颜色缓存映射"""
    configured_color = get_label_color(class_name)
    if configured_color is not None:
        red, green, blue = configured_color
        return (blue, green, red)

    if class_name not in class_colors:
        import colorsys
        h = random.random()
        s = 0.8 + random.random() * 0.2
        v = 0.8 + random.random() * 0.2
        r, g, b = colorsys.hsv_to_rgb(h, s, v)
        color = (int(r * 255), int(g * 255), int(b * 255))
        class_colors[class_name] = color
    return class_colors[class_name]

font_cache = {}

def get_cached_font(size: int):
    global font_cache
    if size not in font_cache:
        try:
            font_cache[size] = ImageFont.truetype(FONT_PATH, size) if os.path.exists(FONT_PATH) else ImageFont.load_default()
        except Exception:
            font_cache[size] = ImageFont.load_default()
    return font_cache[size]

text_render_cache = {}

def _measure_text(font, display_text):
    try:
        left, top, right, bottom = font.getbbox(display_text)
        return max(1, right - left), max(1, bottom - top)
    except Exception:
        dummy_draw = ImageDraw.Draw(Image.new("RGB", (1, 1)))
        text_bbox = dummy_draw.textbbox((0, 0), display_text, font=font)
        return max(1, text_bbox[2] - text_bbox[0]), max(1, text_bbox[3] - text_bbox[1])


def get_cached_text_bgr(display_text, color, size):
    """全局文本渲染缓存器：基于(文字, 颜色, 字号)做三元联合缓存，极速直拷"""
    key = (display_text, color, size)
    patch = text_render_cache.get(key)
    if patch is not None:
        return patch

    # RGB/BGR 互换，因为 PIL 使用 RGB
    rgb_color = (color[2], color[1], color[0])

    font = get_cached_font(size)
    text_w, text_h = _measure_text(font, display_text)

    # 新建对应大小的画布，底色直接铺死
    pil_img = Image.new('RGB', (text_w, text_h + 4), color=rgb_color)
    draw = ImageDraw.Draw(pil_img)
    draw.text((0, 2), display_text, font=font, fill=(255, 255, 255))

    # 转换为 OpenCV 天然能接受的矩阵并压入缓存池
    bgr_patch = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    # 精准淘汰：仅弹出一个最老的值
    if len(text_render_cache) > 8192:
        text_render_cache.pop(next(iter(text_render_cache)))

    text_render_cache[key] = bgr_patch
    return bgr_patch


def cv2_draw_chinese_batch(img, all_boxes, target_classes, show_confidence=True):
    """极速绘制版：直接内存拷贝贴图 <1ms"""
    labels_detected = set()
    if not all_boxes:
        return img, labels_detected

    img_h = img.shape[0]
    img_w = img.shape[1]
    
    # 固定精致小字号
    dynamic_size = 12
    line_thickness = 1

    get_patch = get_cached_text_bgr
    get_color_fallback = get_color_for_class

    try:
        for box_xyxy, label_en, conf in all_boxes:
            b0, b1, b2, b3 = box_xyxy[0], box_xyxy[1], box_xyxy[2], box_xyxy[3]
            
            x1 = int(b0)
            x1 = x1 if x1 >= 0 else 0
            y1 = int(b1)
            y1 = y1 if y1 >= 0 else 0
            x2 = int(b2)
            x2 = x2 if x2 < img_w else img_w - 1
            y2 = int(b3)
            y2 = y2 if y2 < img_h else img_h - 1

            color = get_color_fallback(label_en)

            cn_label = target_classes.get(label_en)
            if cn_label is None:
                cn_label = label_en
            labels_detected.add(cn_label)

            cv2.rectangle(img, (x1, y1), (x2, y2), color, line_thickness, lineType=cv2.LINE_4)

            if show_confidence:
                display_text = f"{cn_label} {conf:.2f}"
            else:
                display_text = cn_label

            bgr_patch = get_patch(display_text, color, dynamic_size)
            patch_h = bgr_patch.shape[0]
            patch_w = bgr_patch.shape[1]

            x_start = x1
            y_start = y1 - patch_h

            if x_start >= 0 and y_start >= 0 and x_start + patch_w < img_w and y_start + patch_h < img_h:
                # 极速直拷：纯内存覆写，<1ms
                img[y_start:y_start+patch_h, x_start:x_start+patch_w] = bgr_patch

        return img, labels_detected

    except Exception as e:
        print("[Draw] 绘制异常:", e)
        for box_xyxy, label_en, conf in all_boxes:
            x1, y1, x2, y2 = map(int, box_xyxy)
            cn_label = target_classes.get(label_en, label_en)
            display_text = f"{cn_label} {conf:.2f}" if show_confidence else cn_label
            color = get_color_for_class(label_en)
            cv2.putText(img, display_text, (max(0, x1), max(0, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            labels_detected.add(cn_label)
        return img, labels_detected
