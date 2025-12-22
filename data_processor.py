# 数据处理模块 - 负责PDF转换和题目分割

import cv2
import numpy as np
import pytesseract
from pdf2image import convert_from_path
from PIL import Image

# 将相对导入改为绝对导入
from config import *
from utils import is_question_number, preprocess_image_for_ocr


class PDFConverter:
    """PDF转图像转换器"""
    
    def __init__(self, config):
        self.config = config
    
    def convert(self, pdf_path, dpi=300):
        """将PDF转换为图像列表"""
        print("正在转换PDF为图像...")
        return convert_from_path(
            pdf_path, 
            dpi=dpi, 
            poppler_path=self.config.POPPLER_PATH
        )


class QuestionDetector:
    """题目检测器 - 负责检测和分割题目"""
    
    def __init__(self, config):
        self.config = config
        # 设置Tesseract路径
        pytesseract.pytesseract.tesseract_cmd = self.config.TESSERACT_PATH

    def page_has_question(self, img):
        """快速检测页面是否包含题号"""
        height, width = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        config = r'--oem 3 --psm 6 outputbase digits'
        data = pytesseract.image_to_data(thresh, config=config, output_type=pytesseract.Output.DICT)

        left_margin = int(width * 0.1)
        for i, text in enumerate(data['text']):
            if is_question_number(text):
                x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
                if x < left_margin and y > height * 0.1 and y < height * 0.9 and h < height * 0.05:
                    return True
        return False
    def detect_questions(self, img):
        """检测并分割题目"""
        # 获取图像尺寸
        height, width = img.shape[:2]
        
        # 预处理：灰度化+二值化
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # 检测题号
        config = r'--oem 3 --psm 6 outputbase digits'
        data = pytesseract.image_to_data(thresh, config=config, output_type=pytesseract.Output.DICT)

        # 找出可能的题号位置
        question_starts = []
        left_margin = int(width * 0.2)  # 左侧30%宽度作为题号区域
        for i, text in enumerate(data['text']):
            if is_question_number(text):
                x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
                # 检查是否在左侧区域，排除页面顶部和底部的区域
                if x < left_margin and y > height * 0.1 and y < height * 0.9 and h < height * 0.05:
                    question_starts.append((x, y, w, h))

        # 按y坐标排序
        question_starts.sort(key=lambda pos: pos[1])

        # 分割题目区域
        questions = []
        for i, (x, y, w, h) in enumerate(question_starts):
            # 确定题目区域：当前题号到下一题号（或页面底部）
            start_y = y
            end_y = question_starts[i + 1][1] if i < len(question_starts) - 1 else height

            # 扩展题号区域（包含题目内容）
            margin = int(h * 1.5)
            roi = img[max(0, start_y - margin):end_y, 0:width]
            questions.append((roi, (0, start_y - margin, width, end_y)))

        return questions


class OCREngine:
    """OCR引擎 - 负责识别题目文本"""
    
    def __init__(self, config):
        self.config = config
        # 设置Tesseract路径
        pytesseract.pytesseract.tesseract_cmd = self.config.TESSERACT_PATH
    
    def extract_text(self, img):
        """OCR识别题目文本，优化数学符号和公式识别"""
        # 1. 图像预处理 - 增强对比度和锐度
        img = img.convert('L')  # 灰度
        img_np = np.array(img)
        
        # 预处理图像
        processed_img_np = preprocess_image_for_ocr(img_np)
        
        # 转回PIL图像
        processed_img = Image.fromarray(processed_img_np)
        
        # 2. 基本OCR识别
        basic_text = pytesseract.image_to_string(processed_img, config=self.config.CUSTOM_CONFIG)
        
        # 3. 针对数学公式的专门识别
        math_text = pytesseract.image_to_string(processed_img, config=self.config.MATH_CONFIG)
        
        # 4. 尝试识别LaTeX格式公式（实验性）
        latex_text = pytesseract.image_to_string(processed_img, config=self.config.LATEX_CONFIG)
        
        # 5. 合并结果（优先选择包含更多数学符号的结果）
        # 简单的启发式规则：选择长度最长的结果
        results = [basic_text, math_text, latex_text]
        final_text = max(results, key=len).strip()
        
        return final_text