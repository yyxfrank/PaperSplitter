# 管道模块 - 核心流程控制

import time
import numpy as np
from PIL import Image

# 将相对导入改为绝对导入
from config import *
from utils import create_directory, validate_file_path, get_absolute_path
# 导入必要的模块
import fitz  # PyMuPDF
from classifier import OpenAIClassifier
from result_processor import ResultOrganizer, ResultSaver
from data_processor import TextExtractor, PyMuPDFQuestionDetector


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
        # 初始化题目检测器
        self.detector = PyMuPDFQuestionDetector(config)

    def execute(self, data):
        """调用data_processor中的PyMuPDFQuestionDetector执行题目检测和分割"""
        pdf_path = data['pdf_path']
        images = data['images']
        
        print("正在使用PyMuPDF检测题目...")
        
        # 调用data_processor中的检测方法
        result = self.detector.detect_questions(pdf_path, images)
        return result


class TextExtractionStage(PipelineStage):
    """文本提取阶段 - 调用data_processor中的TextExtractor实现"""
    
    def __init__(self, config):
        super().__init__("文本提取")
        self.config = config
        self.text_extractor = TextExtractor(config)
    
    def execute(self, data):
        """调用data_processor中的TextExtractor执行文本提取"""
        pdf_path = data['pdf_path']
        questions = data['questions']
        
        # 调用TextExtractor进行文本提取
        questions_with_text = self.text_extractor.extract_text_from_pdf(pdf_path, questions)
        
        return {'questions': questions_with_text}


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