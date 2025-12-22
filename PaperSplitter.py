import os
import re
import cv2
import numpy as np
import time
from pdf2image import convert_from_path
from PIL import Image, ImageDraw, ImageFont
import pytesseract
from openai import OpenAI
# 添加PDF生成库
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# 设置Tesseract路径（根据你的安装位置修改）
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# 配置Tesseract使用中文、英文和数学公式语言包
# 注意：确保tessdata目录中包含chi_sim.traineddata、eng.traineddata和equ.traineddata文件
custom_config = r'--oem 3 --psm 6 -l chi_sim+eng+equ'

# 额外的数学符号识别配置
math_config = r'--oem 3 --psm 4 -l chi_sim+eng+equ'

# 配置Tesseract识别LaTeX公式的参数（实验性功能）
latex_config = r'--oem 3 --psm 11 -l chi_sim+eng+equ'


class PaperSplitter:
    def __init__(self, pdf_path, output_dir, deepseek_api_key, custom_categories=None):
        self.pdf_path = pdf_path
        self.output_dir = output_dir
        self.api_key = deepseek_api_key
        # 初始化OpenAI客户端，使用NVIDIA API
        try:
            # 尝试不带proxies参数的初始化方式
            self.client = OpenAI(
                base_url="https://integrate.api.nvidia.com/v1",
                api_key=self.api_key
            )
        except TypeError as e:
            # 如果遇到类型错误，可能是库版本问题，尝试更简单的初始化方式
            print(f"初始化客户端时出错，尝试替代方案: {e}")
            self.client = OpenAI(
                api_key=self.api_key
            )
            # 手动设置base_url
            self.client.base_url = "https://integrate.api.nvidia.com/v1"
        os.makedirs(output_dir, exist_ok=True)
        
        # 设置默认候选类别
        self.default_categories = ["代数", "几何", "概率统计", "函数", "解析几何", "三角函数"]
        self.candidate_labels = custom_categories if custom_categories else self.default_categories
        
        # 初始化关键词映射
        self.keyword_map = {
            "代数": ["方程", "不等式", "代数", "多项式", "因式分解", "有理数", "无理数"],
            "几何": ["图形", "几何", "三角形", "四边形", "圆", "面积", "体积", "相似", "全等"],
            "概率统计": ["概率", "统计", "期望", "方差", "频率", "分布", "抽样"],
            "函数": ["函数", "定义域", "值域", "单调性", "奇偶性", "周期性", "反函数"],
            "解析几何": ["坐标", "直线", "圆", "椭圆", "双曲线", "抛物线", "向量"],
            "三角函数": ["正弦", "余弦", "正切", "三角函数", "周期", "振幅", "相位"]
        }
        
        # 如果使用自定义类别，过滤关键词映射
        if custom_categories:
            self.keyword_map = {k: v for k, v in self.keyword_map.items() if k in custom_categories}
        


    def process_pdf(self):
        """主处理流程"""
        # 1. PDF转图像
        print("正在转换PDF为图像...")
        images = convert_from_path(self.pdf_path, dpi=300,poppler_path=r'C:\Program Files\Release-25.07.0-0\Library\bin')

        all_questions = []
        for page_num, img in enumerate(images):
            # 2. 检测并分割题目
            page_questions = self.detect_and_crop_questions(np.array(img))
            print(f"页面 {page_num + 1} 检测到 {len(page_questions)} 道题目")

            for i, (q_img, bbox) in enumerate(page_questions):
                # 3. OCR识别题目文本
                text = self.extract_question_text(Image.fromarray(q_img))

                # 4. 使用DeepSeek API进行分类
                category = self.classify_with_deepseek(text)

                # 保存题目切片
                q_id = f"p{page_num + 1}_q{i + 1}"
                filename = f"{q_id}_{category[:10]}.png"
                Image.fromarray(q_img).save(os.path.join(self.output_dir, filename))

                all_questions.append({
                    "id": q_id,
                    "image": q_img,
                    "text": text,
                    "category": category,
                    "page": page_num + 1,
                    "position": bbox[1]  # 使用y坐标排序
                })

                # 添加延迟避免API速率限制
                time.sleep(0.5)

        # 5. 按类别整理结果
        self.organize_by_category(all_questions)

    def detect_and_crop_questions(self, img):
        """检测题号位置并分割题目"""
        # 获取图像尺寸
        height, width = img.shape[:2]
        
        # 预处理：灰度化+二值化
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # 检测题号（匹配纯数字或数字+点格式，如"1"、"2."）
        config = r'--oem 3 --psm 6 outputbase digits'
        data = pytesseract.image_to_data(thresh, config=config, output_type=pytesseract.Output.DICT)

        # 找出可能的题号位置
        question_starts = []
        left_margin = int(width * 0.1)  # 左侧10%宽度作为题号区域
        for i, text in enumerate(data['text']):
            text_stripped = text.strip()
            # 匹配纯数字或数字+点格式，且数字长度在1-3位之间
            if (re.match(r'^\d{1,3}$', text_stripped) or re.match(r'^\d{1,3}\.$', text_stripped)) and len(text_stripped) <= 4:
                x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
                # 检查是否在左侧区域（自成一列的题号通常在左侧）
                # 排除页面顶部和底部的区域，避免条形码干扰
                if x < left_margin and y > height * 0.1 and y < height * 0.9 and h < height * 0.05:
                    question_starts.append((x, y, w, h))

        # 按y坐标排序
        question_starts.sort(key=lambda pos: pos[1])

        # 分割题目区域
        questions = []
        height, width = img.shape[:2]
        for i, (x, y, w, h) in enumerate(question_starts):
            # 确定题目区域：当前题号到下一题号（或页面底部）
            start_y = y
            end_y = question_starts[i + 1][1] if i < len(question_starts) - 1 else height

            # 扩展题号区域（包含题目内容）
            margin = int(h * 1.5)
            roi = img[max(0, start_y - margin):end_y, 0:width]
            questions.append((roi, (0, start_y - margin, width, end_y)))

        return questions

    def extract_question_text(self, img):
        """OCR识别题目文本，优化数学符号和公式识别"""
        # 1. 图像预处理 - 增强对比度和锐度
        img = img.convert('L')  # 灰度
        img_np = np.array(img)
        
        # 自适应阈值处理，提高文字与背景对比度
        thresh = cv2.adaptiveThreshold(
            img_np, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        
        # 形态学操作，增强字符连接
        kernel = np.ones((1, 1), np.uint8)
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
        
        # 图像锐化
        sharpen_kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        sharpened = cv2.filter2D(opening, -1, sharpen_kernel)
        
        # 转回PIL图像
        processed_img = Image.fromarray(sharpened)
        
        # 2. 基本OCR识别
        basic_text = pytesseract.image_to_string(processed_img, config=custom_config)
        
        # 3. 针对数学公式的专门识别
        math_text = pytesseract.image_to_string(processed_img, config=math_config)
        
        # 4. 尝试识别LaTeX格式公式（实验性）
        latex_text = pytesseract.image_to_string(processed_img, config=latex_config)
        
        # 5. 合并结果（优先选择包含更多数学符号的结果）
        # 简单的启发式规则：选择长度最长的结果
        results = [basic_text, math_text, latex_text]
        final_text = max(results, key=len).strip()
        
        return final_text

    def classify_with_deepseek(self, text):
        """使用DeepSeek API进行分类"""
        # 定义候选类别（可根据需要修改）
        # candidate_labels = ["代数", "几何", "概率统计", "函数", "解析几何", "三角函数"]

        try:
            # 使用OpenAI客户端调用API
            response = self.client.chat.completions.create(
                model="deepseek-ai/deepseek-r1-0528",
                messages=[
                    {"role": "system", "content": "你是一名题目分类专家，需要将题目分类到指定类别中。"},
                    {"role": "user", "content": f"请将以下试卷中的题目分类到以下类别之一：{', '.join(self.candidate_labels)}\n\n题目内容：{text}\n\n只返回类别名称，不要添加任何解释。如果提供的文本中没有可以被认定为题目的内容，则不返回任何内容。"}


                ],
                temperature=0.6,
                top_p=0.7,
                max_tokens=4096,
                stream=True
            )
            # 初始化类别变量
            category = ""

            for chunk in response:
                if chunk.choices and len(chunk.choices) > 0:
                    delta = chunk.choices[0].delta
                    reasoning = getattr(delta, "reasoning_content", None)
                    if reasoning:
                        print(reasoning, end="")
                    if delta.content is not None:
                        category += delta.content
                        print(delta.content, end="")
            
            category = category.strip()

            # 验证返回的类别是否在候选列表中
            if category in self.candidate_labels:
                return category
            else:
                # 如果返回不在列表中，尝试匹配最相似的类别
                for label in self.candidate_labels:
                    if label in category:
                        return label
                # 默认返回第一个类别
                return self.candidate_labels[0]

        except Exception as e:
            print(f"DeepSeek API调用失败: {str(e)}")
            # API失败时使用简单的关键词匹配作为后备方案
            return self.fallback_classification(text)

    def fallback_classification(self, text):
        """当API失败时的后备分类方法"""
        # 简单的关键词匹配分类

        # 统计关键词出现次数
        scores = {label: 0 for label in self.candidate_labels}
        for label, keywords in self.keyword_map.items():
            for keyword in keywords:
                if keyword in text:
                    scores[label] += 1

        # 找到得分最高的类别
        max_score = max(scores.values())
        if max_score > 0:
            # 如果有多个类别得分相同，返回第一个
            for label in self.candidate_labels:
                if scores[label] == max_score:
                    return label

        # 如果没有匹配的关键词，返回第一个类别
        return self.candidate_labels[0]

    def set_categories(self, categories, keyword_map=None):
        """设置新的分类类别和关键词映射"""
        self.candidate_labels = categories
        if keyword_map:
            self.keyword_map = keyword_map
        else:
            # 如果没有提供关键词映射，保留现有映射但过滤掉不在新类别中的项
            self.keyword_map = {k: v for k, v in self.keyword_map.items() if k in categories}

    def get_categories(self):
        """获取当前的分类类别"""
        return self.candidate_labels

    def get_keyword_map(self):
        """获取当前的关键词映射"""
        return self.keyword_map


    def organize_by_category(self, questions):
        """按类别整理题目并添加类别标签"""
        # 按类别分组
        categories = {}  # 用于存储类别顺序
        category_order = []  # 类别顺序列表
        for q in questions:
            cat = q['category']
            if cat not in categories:
                categories[cat] = []
                category_order.append(cat)  # 记录类别顺序
            categories[cat].append(q)

        # 按类别内题目位置排序
        for cat, items in categories.items():
            items.sort(key=lambda x: (x['page'], x['position']))

        # 为每个类别生成单独的PDF
        for cat in category_order:
            # 生成当前类别的结果图像列表
            result_images = []
            # 添加类别标题
            title = self.create_category_heading(cat)
            result_images.append((title, "header"))
            # 添加当前类别的题目
            for q in categories[cat]:
                result_images.append((Image.fromarray(q['image']), q['id']))
            # 为当前类别保存结果
            self.save_results(result_images, [cat])

        print(f"所有类别PDF生成完成，保存在 {self.output_dir} 目录")

    def create_category_heading(self, category_name):
        """创建类别标题图片"""
        img = Image.new('RGB', (1200, 80), color=(73, 109, 137))
        draw = ImageDraw.Draw(img)
        try:
            # 尝试使用系统字体
            font = ImageFont.truetype("simhei.ttf", 40)
        except:
            # 后备方案：使用默认字体
            font = ImageFont.load_default()
        draw.text((20, 20), f"【{category_name}】", font=font, fill=(255, 255, 0))
        return img

    def save_results(self, result_images, category_order):
        """保存整理后的结果并为每个类别生成单独的PDF"""
        output_dir = os.path.join(self.output_dir, "organized")
        os.makedirs(output_dir, exist_ok=True)

        # 保存为PNG图像
        for i, (img, name) in enumerate(result_images):
            img.save(os.path.join(output_dir, f"{i + 1:03d}_{name}.png"))

        # 生成PDF文件 - 使用类别名称作为文件名
        if category_order and len(category_order) > 0:
            category_name = category_order[0]
        else:
            category_name = "未知类别"
        pdf_path = os.path.join(self.output_dir, f"{category_name}_题目.pdf")
        c = canvas.Canvas(pdf_path, pagesize=letter)
        width, height = letter

        # 设置中文字体
        try:
            # 尝试使用系统中的中文字体
            font_path = r"C:\Windows\Fonts\simhei.ttf"
            pdfmetrics.registerFont(TTFont('SimHei', font_path))
            c.setFont("SimHei", 12)
            font_available = True
        except Exception as e:
            print(f"无法加载中文字体: {str(e)}")
            # 如果找不到指定字体，使用默认字体
            c.setFont("Helvetica", 12)
            font_available = False

        y_position = height - 50  # 起始y坐标
        header_index = 0  # 用于跟踪当前处理的类别标题
        for i, (img, name) in enumerate(result_images):
            if name == "header":
                # 添加类别标题
                if header_index < len(category_order):
                    category_name = category_order[header_index]
                    header_index += 1
                else:
                    category_name = "未知类别"
                      
                c.setFont("SimHei" if font_available else "Helvetica", 16)
                c.drawString(50, y_position, f"【{category_name}】")
                y_position -= 30
                c.setFont("SimHei" if font_available else "Helvetica", 12)
            else:
                # 添加题目图像
                img_path = os.path.join(output_dir, f"{i + 1:03d}_{name}.png")
                img_reader = ImageReader(img_path)
                img_width, img_height = img.size
                
                # 计算图像大小以适应页面宽度
                max_width = width - 100
                if img_width > max_width:
                    ratio = max_width / img_width
                    img_width = max_width
                    img_height = img_height * ratio
                
                # 检查是否有足够空间放置图像，如果没有则新建页面
                if y_position - img_height < 50:
                    c.showPage()
                    y_position = height - 50
                    c.setFont("SimHei" if font_available else "Helvetica", 12)
                
                c.drawImage(img_reader, 50, y_position - img_height, width=img_width, height=img_height)
                y_position -= img_height + 20

        c.save()
        print(f"{category_name} PDF文件已生成: {pdf_path}")


import argparse

if __name__ == "__main__":
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='试卷拆分工具')
    parser.add_argument('pdf_path', help='PDF文件路径')
    parser.add_argument('--output_dir', default='output_questions', help='输出目录')
    parser.add_argument('--api_key', help='DeepSeek API密钥')
    parser.add_argument('--categories', help='自定义分类类别，用逗号分隔（例如：代数,几何,概率统计）')
    args = parser.parse_args()

    # 从命令行参数或环境变量获取API密钥
    DEEPSEEK_API_KEY = args.api_key or "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"  # 替换为你的实际API密钥

    # 使用绝对路径确保中文路径正确编码
    pdf_file_path = os.path.abspath(args.pdf_path)
    
    # 检查文件是否存在
    if not os.path.exists(pdf_file_path):
        raise FileNotFoundError(f"无法找到PDF文件: {pdf_file_path}")
    
    # 处理自定义类别
    custom_categories = None
    if args.categories:
        custom_categories = [cat.strip() for cat in args.categories.split(',')]
        print(f"使用自定义分类类别: {custom_categories}")
    else:
        print("使用默认分类类别")

    processor = PaperSplitter(
        pdf_path=pdf_file_path,
        output_dir=args.output_dir,
        deepseek_api_key=DEEPSEEK_API_KEY,
        custom_categories=custom_categories
    )
    processor.process_pdf()
    print(f"处理完成！结果保存在 {args.output_dir}/organized 目录")

# 程序输入输出说明
# 输入:
# - pdf_path: 数学试卷PDF文件路径
# - output_dir: 结果输出目录
# - deepseek_api_key: DeepSeek API密钥
# - custom_categories (可选): 自定义分类类别列表
# 输出:
# - 原始题目图像: 保存在output_dir目录
# - 分类整理后的图像: 保存在output_dir/organized目录
# - 分类整理题目.pdf: 保存在output_dir目录