# 工具函数模块

import os
import re
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def create_directory(path):
    """创建目录（如果不存在）"""
    os.makedirs(path, exist_ok=True)


def validate_file_path(file_path):
    """验证文件是否存在"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"无法找到文件: {file_path}")
    return True


def get_absolute_path(path):
    """获取绝对路径，处理中文路径"""
    return os.path.abspath(path)


def preprocess_image_for_ocr(img_np):
    """图像预处理，增强OCR识别效果"""
    # 自适应阈值处理
    thresh = cv2.adaptiveThreshold(
        img_np, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    
    # 形态学操作，增强字符连接
    kernel = np.ones((1, 1), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # 图像锐化
    sharpen_kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    sharpened = cv2.filter2D(opening, -1, sharpen_kernel)
    
    return sharpened


def create_category_header(category_name, config):
    """创建类别标题图像"""
    img = Image.new('RGB', 
                    (config.HEADER_IMAGE_WIDTH, config.HEADER_IMAGE_HEIGHT), 
                    color=config.HEADER_COLOR)
    draw = ImageDraw.Draw(img)
    try:
        # 尝试使用指定字体
        font = ImageFont.truetype(config.DEFAULT_FONT_PATH, config.HEADER_FONT_SIZE)
    except:
        # 后备方案：使用默认字体
        try:
            font = ImageFont.truetype("simhei.ttf", config.HEADER_FONT_SIZE)
        except:
            font = ImageFont.load_default()
    
    draw.text((20, 20), f"【{category_name}】", font=font, fill=config.HEADER_TEXT_COLOR)
    return img


def is_question_number(text):
    """判断文本是否为题号格式"""
    text_stripped = text.strip()
    # 匹配纯数字或数字+点格式，且数字长度在1-3位之间
    return bool((re.match(r'^\d{1,3}$', text_stripped) or 
                re.match(r'^\d{1,3}\.$', text_stripped)) and 
                len(text_stripped) <= 4)


def calculate_image_dimensions(img_size, max_width):
    """计算图像大小以适应页面宽度"""
    img_width, img_height = img_size
    if img_width > max_width:
        ratio = max_width / img_width
        img_width = max_width
        img_height = img_height * ratio
    return img_width, img_height


def get_pdf_page_layout(config):
    """获取PDF页面布局信息"""
    from reportlab.lib.pagesizes import letter
    width, height = letter
    
    return {
        'page_width': width,
        'page_height': height,
        'content_width': width - config.PDF_MARGIN_LEFT - config.PDF_MARGIN_RIGHT,
        'content_height': height - config.PDF_MARGIN_TOP - config.PDF_MARGIN_BOTTOM,
        'start_x': config.PDF_MARGIN_LEFT,
        'start_y': height - config.PDF_MARGIN_TOP
    }