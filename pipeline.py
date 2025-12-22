# 管道模块 - 核心流程控制

import time
import numpy as np
from PIL import Image

# 将相对导入改为绝对导入
from config import *
from utils import create_directory, validate_file_path, get_absolute_path
# 导入PyMuPDF处理模块，使用下划线替代空格
import fitz
from import_fitz import find_leftmost_bold_numbers_on_page, extract_questions_using_candidates
from classifier import OpenAIClassifier
from result_processor import ResultOrganizer, ResultSaver


class PipelineStage:
    """管道阶段基类"""
    
    def __init__(self, name):
        self.name = name
    
    def execute(self, data):
        """执行阶段处理，子类必须实现"""
        raise NotImplementedError("子类必须实现execute方法")


class PDFToImageStage(PipelineStage):
    """PDF转图像阶段 - 使用PyMuPDF实现"""
    
    def __init__(self, config):
        super().__init__("PDF转图像")
        self.config = config
    
    def execute(self, data):
        """使用PyMuPDF执行PDF转图像处理"""
        pdf_path = data['pdf_path']
        print("正在使用PyMuPDF转换PDF为图像...")
        
        images = []
        doc = fitz.open(pdf_path)
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            # 设置dpi为300，与原实现保持一致
            pix = page.get_pixmap(dpi=300)
            # 将pixmap转换为PIL Image
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            images.append(img)
        
        doc.close()
        return {'images': images}


class QuestionDetectionStage(PipelineStage):
    """题目检测阶段 - 使用PyMuPDF实现"""

    def __init__(self, config):
        super().__init__("题目检测与分割")
        self.config = config

    def execute(self, data):
        """使用PyMuPDF执行题目检测和分割"""
        pdf_path = data['pdf_path']
        images = data['images']
        all_questions = []

        print("正在使用PyMuPDF检测题目...")
        
        # 打开PDF文档
        doc = fitz.open(pdf_path)
        
        # 🔹 第一步：检测从哪一页开始有题目
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

        # 🔹 第二步：只处理从 start_index 开始的页
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


class TextExtractionStage(PipelineStage):
    """文本提取阶段 - 使用PyMuPDF实现"""
    
    def __init__(self, config):
        super().__init__("文本提取")
        self.config = config
    
    def execute(self, data):
        """使用PyMuPDF执行文本提取"""
        pdf_path = data['pdf_path']
        questions = data['questions']
        
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
        return {'questions': questions}


class ClassificationStage(PipelineStage):
    """分类阶段"""
    
    def __init__(self, config,custom_categories=None):
        super().__init__("题目分类")
        self.classifier = OpenAIClassifier(config,custom_categories)
        self.rate_limit_delay = config.API_RATE_LIMIT_DELAY
    
    def execute(self, data):
        """执行题目分类"""
        questions = data['questions']

        i=0
        for q in questions:
            # 使用分类器进行分类
            category = self.classifier.classify(q['text'])
            q['category'] = category
            i+=1
            print(f"正在分类第{i}题")
            # 添加延迟避免API速率限制
            time.sleep(self.rate_limit_delay)
        
        return {'questions': questions}


class ResultSavingStage(PipelineStage):
    """结果保存阶段"""
    
    def __init__(self, config, output_dir):
        super().__init__("结果保存")
        self.result_saver = ResultSaver(config, output_dir)
    
    def execute(self, data):
        """执行结果保存"""
        questions = data['questions']
        
        # 保存原始题目图像
        for q in questions:
            filename = f"{q['id']}_{q['category'][:10]}.png"
            self.result_saver.save_question(q['image'], filename)
        
        return {'questions': questions}


class OrganizationStage(PipelineStage):
    """结果组织阶段"""
    
    def __init__(self, config, output_dir):
        super().__init__("结果组织")
        self.organizer = ResultOrganizer(config, output_dir)
        self.result_saver = ResultSaver(config, output_dir)
        self.config = config
    
    def execute(self, data):
        """执行结果组织和PDF生成"""
        questions = data['questions']
        
        # 按类别整理题目
        categories, category_order = self.organizer.organize_by_category(questions)
        
        # 为每个类别生成PDF
        pdf_paths = []
        for cat in category_order:
            # 准备类别结果
            result_images = self.result_saver.prepare_category_results(
                cat, categories[cat], self.config
            )
            # 保存并生成PDF
            pdf_path = self.result_saver.save_organized_results(
                result_images, [cat]
            )
            pdf_paths.append(pdf_path)
        
        print(f"所有类别PDF生成完成，保存在 {self.result_saver.output_dir} 目录")
        return {
            'questions': questions,
            'categories': categories,
            'category_order': category_order,
            'pdf_paths': pdf_paths
        }


class PaperProcessingPipeline:
    """试卷处理管道 - 组织和执行整个处理流程"""
    
    def __init__(self, config, pdf_path, output_dir, custom_categories=None):
        # 验证和准备路径
        self.pdf_path = get_absolute_path(pdf_path)
        validate_file_path(self.pdf_path)
        self.output_dir = output_dir
        create_directory(self.output_dir)
        
        self.config = config
        self.custom_categories = custom_categories
        
        # 初始化各个阶段
        self.stages = [
            PDFToImageStage(config),
            QuestionDetectionStage(config),
            TextExtractionStage(config),
            ClassificationStage(config, custom_categories),
            ResultSavingStage(config, output_dir),
            OrganizationStage(config, output_dir)
        ]
    
    def execute(self):
        """执行整个处理管道"""
        # 初始数据
        data = {'pdf_path': self.pdf_path}
        
        # 按顺序执行每个阶段
        for stage in self.stages:
            print(f"执行阶段: {stage.name}")
            result = stage.execute(data)
            # 合并结果到数据字典
            data.update(result)
        
        return data
    
    def get_classifier(self):
        """获取分类器实例"""
        # 从分类阶段获取分类器
        for stage in self.stages:
            if isinstance(stage, ClassificationStage):
                return stage.classifier
        return None
    
    def set_categories(self, categories, keyword_map=None):
        """设置分类类别"""
        classifier = self.get_classifier()
        if classifier:
            classifier.set_categories(categories, keyword_map)
            self.custom_categories = categories