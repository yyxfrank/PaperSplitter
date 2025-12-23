# 数据处理模块 - 负责PDF转换和题目分割

import cv2
import numpy as np
import pytesseract
import fitz  # PyMuPDF
from pdf2image import convert_from_path
from PIL import Image

# 将相对导入改为绝对导入
from config import *
from utils import is_question_number, preprocess_image_for_ocr
import re

# 可扩展的字体"粗体"关键字集合（可根据语言/字体补充）
BOLD_KEYWORDS = ["bold", "bd", "black", "heavy", "semibold", "demibold", "hei", "黑体", "粗体"]

# 判断一个字体名是否看起来是加粗（启发式）
def font_name_is_bold(fontname: str) -> bool:
    if not fontname:
        return False
    name = fontname.lower()
    for kw in BOLD_KEYWORDS:
        if kw in name:
            return True
    return False

# 判断 token 是否为题号候选（数字或带括号的数字）
def token_is_number(token: str) -> bool:
    token = token.strip()
    # 匹配：1 1. 1） 1) （1） (1) 第1题  等常见格式（可自行扩展）
    patterns = [
        r"^\d+$",
        r"^\d+\.$",
        r"^\d+[）\)]$",
        r"^[（(]\d+[)）]$",
        r"^第\d+题$"
    ]
    for p in patterns:
        if re.fullmatch(p, token):
            return True
    return False


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


def find_leftmost_bold_numbers_on_page(page, left_ratio=0.3, y_margin=(0.05, 0.95), size_min=None):
    """
    在单页中寻找满足条件的题号（数字 & 字体名显示为 bold & x0 是最小/接近最小）。
    left_ratio: 只在页面左侧 left_ratio 的宽度内寻找候选（防止页眉/页脚干扰）
    y_margin: 忽略页面顶部/底部的 y 区域（比例）
    size_min: 若提供，排除字号小于该值的文本（避免页码/注释）
    返回：列表 of dict: { 'token': str, 'x0': float, 'y0': float, 'bbox': (x0,y0,x1,y1), 'font': str, 'size': float, 'bold': bool }
    """
    page_dict = page.get_text("dict")
    page_width = page.mediabox_size[0]
    page_height = page.mediabox_size[1]

    # 收集所有 word-level 信息（来自 spans -> words）
    words = []  # 每项: (x0,y0,x1,y1, text, font, size)
    for block in page_dict.get("blocks", []):
        if block.get("type") != 0:  # 0 == text
            continue
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                font = span.get("font", "")
                size = span.get("size", 0)
                text = span.get("text", "")
                # spans 的 text 可能包含多个词，需要按空格拆分并估算位置
                # 更精确的方法是使用 page.get_text("words"), 但这里我们也可以从 spans 中拆词
                # 使用 page.get_text("words") 得到更准确的 x0,x1

    # 下面使用 words 输出（更精确）
    raw_words = page.get_text("words")  # (x0, y0, x1, y1, "text", block_no, line_no, word_no)
    # 要把每个 word 映射到对应的 span 的 font/size：使用 rawdict 的 spans 字符信息更精确（含 chars）
    # 为简洁起见，我们 will try to map by spatial overlap: find span that overlaps the word bbox
    raw_dict = page.get_text("rawdict")
    spans_index = []
    for block in raw_dict.get("blocks", []):
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                spans_index.append({
                    "bbox": span.get("bbox"),  # [x0,y0,x1,y1]
                    "font": span.get("font"),
                    "size": span.get("size"),
                    "text": span.get("text")
                })

    # helper: test overlap of boxes
    def bbox_overlap(a, b):
        ax0, ay0, ax1, ay1 = a
        bx0, by0, bx1, by1 = b
        return not (ax1 < bx0 or bx1 < ax0 or ay1 < by0 or by1 < ay0)

    for w in raw_words:
        x0, y0, x1, y1, txt, *_ = w
        # filter by y margin
        if not (page_height * y_margin[0] <= y0 <= page_height * y_margin[1]):
            continue
        # filter by left area
        if x0 > page_width * left_ratio:
            continue
        # optional size filter: will be filled from matched span
        matched_font = ""
        matched_size = None
        for s in spans_index:
            if bbox_overlap((x0, y0, x1, y1), tuple(s["bbox"])):
                matched_font = s.get("font", "")
                matched_size = s.get("size", None)
                break
        if size_min is not None and matched_size is not None and matched_size < size_min:
            continue
        words.append({
            "x0": x0, "y0": y0, "x1": x1, "y1": y1,
            "text": txt.strip(),
            "font": matched_font,
            "size": matched_size
        })

    if not words:
        return []

    # 计算页面的最小 x0（在我们筛选后的正文候选词中）
    min_x0 = min(w["x0"] for w in words)

    # 现在筛选：文本为数字/题号样式 & font 看起来是 bold & x0 接近 min_x0
    candidates = []
    for w in words:
        txt = w["text"]
        if not txt:
            continue
        # 把类似 "1." 或 "1" 或 "(1)" 等作为 token 前缀匹配
        token = txt
        # strip trailing punctuation often attached like "1." or "1)"
        token_norm = token.strip().strip(".、)）")
        # 判断是否数字
        if not re.fullmatch(r"\d+", token_norm):
            continue
        # 判断bold（启发式）
        bold = font_name_is_bold(w.get("font", ""))
        # 判断是否最左：我们允许小容差，如 2-3 个像素或 1% 页宽
        tol = max(3.0, page_width * 0.01)
        is_leftmost = abs(w["x0"] - min_x0) <= tol
        if bold and is_leftmost:
            candidates.append({**w, "token": token, "bold": bold})
    return candidates


class PyMuPDFQuestionDetector:
    """基于PyMuPDF的题目检测器"""
    
    def __init__(self, config):
        self.config = config
    
    def detect_questions(self, pdf_path, images):
        """使用PyMuPDF检测并分割题目"""
        all_questions = []
        
        # 打开PDF文档
        doc = fitz.open(pdf_path)
        
        # 第一步：检测从哪一页开始有题目
        start_index = 0
        for i in range(len(doc)):
            page = doc[i]
            candidates = find_leftmost_bold_numbers_on_page(page)
            if candidates:
                start_index = i
                print(f"✅ 检测到第 {i + 1} 页开始出现题目，将从这里开始分析。")
                break
        else:
            print("⚠️ 未检测到题目页，终止题目检测阶段。")
            doc.close()
            return {'questions': []}

        # 第二步：只处理从 start_index 开始的页
        for page_num in range(start_index, len(doc)):
            page = doc[page_num]
            # 获取该页的候选题号
            candidates = find_leftmost_bold_numbers_on_page(page)
            
            # 如果没有找到候选，尝试放宽条件
            if not candidates:
                candidates = find_leftmost_bold_numbers_on_page(page, left_ratio=0.4)
            
            if candidates:
                print(f"页面 {page_num + 1} 检测到 {len(candidates)} 道题目")
                
                # 按y坐标排序候选
                candidates_sorted = sorted(candidates, key=lambda c: c['y0'])
                
                # 获取页面图像
                page_image = images[page_num]
                page_img_np = np.array(page_image)
                page_height, page_width = page_img_np.shape[:2]
                
                # 基于候选题号分割题目图像
                for i, candidate in enumerate(candidates_sorted):
                    # 确定题目的边界
                    start_y = candidate['y0']
                    # 下一题号的y坐标或页面底部
                    if i < len(candidates_sorted) - 1:
                        end_y = candidates_sorted[i + 1]['y0']
                    else:
                        end_y = page.rect.height
                    
                    # 计算图像上的实际坐标（考虑dpi缩放）
                    # 假设PDF的默认分辨率是72dpi，而我们的图像是300dpi
                    scale_factor = 300 / 72
                    start_y_img = int(start_y * scale_factor)
                    end_y_img = int(end_y * scale_factor)
                    
                    # 确保坐标在图像范围内
                    start_y_img = max(0, start_y_img)
                    end_y_img = min(page_height, end_y_img)
                    
                    # 提取题目图像区域
                    q_img = page_img_np[start_y_img:end_y_img, 0:page_width]
                    
                    bbox = (0, start_y_img, page_width, end_y_img)
                    
                    # 创建题目数据
                    q_id = f"p{page_num + 1}_q{i + 1}"
                    question_data = {
                        "id": q_id,
                        "image": q_img,
                        "page": page_num + 1,
                        "position": start_y_img,
                        "bbox": bbox,
                        "token": candidate['token'],  # 添加题号文本
                        "text": ""  # 先留空，后面OCR阶段会填充
                    }
                    all_questions.append(question_data)
        
        doc.close()
        return {'questions': all_questions}


class TextExtractor:
    """文本提取器 - 使用PyMuPDF实现题目文本提取"""
    
    def __init__(self, config):
        self.config = config
    
    def extract_text_from_pdf(self, pdf_path, questions):
        """使用PyMuPDF从PDF中提取题目文本"""
        print("正在使用PyMuPDF提取题目文本...")
        
        # 打开PDF文档
        doc = fitz.open(pdf_path)
        
        # 按页面分组题目
        questions_by_page = {}
        for q in questions:
            page_num = q['page'] - 1  # 转换为0索引
            if page_num not in questions_by_page:
                questions_by_page[page_num] = []
            questions_by_page[page_num].append(q)
        
        # 对每个页面的题目提取文本
        for page_num, page_questions in questions_by_page.items():
            if page_num >= len(doc):
                continue
                
            page = doc[page_num]
            # 获取页面的所有行文本及其坐标
            words = page.get_text("words")  # 返回(x0, y0, x1, y1, text, block_no, line_no, word_no)
            
            # 按题目分割文本
            for q in page_questions:
                # 获取题目的y坐标范围（PDF坐标，72dpi）
                # 需要从图像坐标转换回PDF坐标
                scale_factor = 72 / 300
                q_start_y = q['position'] * scale_factor
                
                # 找到该题目的结束y坐标
                if 'bbox' in q and len(q['bbox']) >= 4:
                    q_end_y = q['bbox'][3] * scale_factor
                else:
                    # 如果没有bbox信息，尝试从相邻题目获取
                    q_end_y = page.rect.height
                    for other_q in page_questions:
                        if other_q['position'] > q['position']:
                            other_start_y = other_q['position'] * scale_factor
                            q_end_y = min(q_end_y, other_start_y)
                
                # 收集该题目范围内的文本
                q_words = []
                for word in words:
                    x0, y0, x1, y1, text = word[:5]
                    # 如果单词的y坐标在题目范围内
                    if q_start_y <= y0 <= q_end_y:
                        q_words.append((x0, y0, text))
                
                # 按y和x坐标排序，然后按行合并文本
                q_words.sort(key=lambda w: (w[1], w[0]))
                
                # 简单地按行聚合文本
                current_y = None
                current_line = []
                q_text_lines = []
                
                for x0, y0, text in q_words:
                    # 如果是新行（y坐标变化超过2个单位）
                    if current_y is None or abs(y0 - current_y) > 2:
                        if current_line:
                            q_text_lines.append(' '.join(current_line))
                            current_line = []
                        current_y = y0
                    current_line.append(text)
                
                if current_line:
                    q_text_lines.append(' '.join(current_line))
                
                # 将多行文本合并
                q_text = '\n'.join(q_text_lines)
                
                # 如果PyMuPDF提取的文本为空，仍然可以使用OCR作为后备方案
                if not q_text.strip() and 'image' in q:
                    try:
                        # 这里可以添加简单的OCR作为后备，但我们先注释掉
                        # import pytesseract
                        # q_text = pytesseract.image_to_string(Image.fromarray(q['image']))
                        q_text = f"[题目文本提取失败 - 题号: {q.get('token', '未知')}]"
                    except:
                        q_text = "[题目文本提取失败]"
                
                q['text'] = q_text
        
        doc.close()
        return questions