# 结果处理模块 - 负责组织和保存处理后的题目结果

import os
from PIL import Image
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# 将相对导入改为绝对导入
from utils import create_directory, calculate_image_dimensions, get_pdf_page_layout, create_category_header


class ResultOrganizer:
    """结果组织器 - 按类别整理题目"""
    
    def __init__(self, config, output_dir):
        self.config = config
        self.output_dir = output_dir
        self.organized_dir = os.path.join(output_dir, "organized")
        create_directory(self.organized_dir)
    
    def organize_by_category(self, questions):
        """按类别整理题目并返回类别分组"""
        # 按类别分组
        categories = {}
        category_order = []  # 类别顺序列表
        
        for q in questions:
            cat = q['category']
            if cat not in categories:
                categories[cat] = []
                category_order.append(cat)
            categories[cat].append(q)

        # 按类别内题目位置排序
        for cat, items in categories.items():
            items.sort(key=lambda x: (x['page'], x['position']))
        
        return categories, category_order


class PDFGenerator:
    """PDF生成器 - 生成分类后的PDF文件"""
    
    def __init__(self, config, output_dir):
        self.config = config
        self.output_dir = output_dir
        self.organized_dir = os.path.join(output_dir, "organized")
        create_directory(self.organized_dir)
    
    def generate_pdf(self, result_images, category_order):
        """生成PDF文件"""
        # 使用类别名称作为文件名
        if category_order and len(category_order) > 0:
            category_name = category_order[0]
        else:
            category_name = "未知类别"
        
        pdf_path = os.path.join(self.output_dir, f"{category_name}_题目.pdf")
        
        # 创建PDF画布
        c = canvas.Canvas(pdf_path, pagesize=letter)
        width, height = letter

        # 设置中文字体
        font_available = self._setup_font(c)

        # 获取页面布局信息
        layout = get_pdf_page_layout(self.config)
        y_position = layout['start_y']
        header_index = 0
        
        # 添加内容到PDF
        for i, (img, name) in enumerate(result_images):
            if name == "header":
                # 添加类别标题
                if header_index < len(category_order):
                    category_name = category_order[header_index]
                    header_index += 1
                else:
                    category_name = "未知类别"
                       
                c.setFont("SimHei" if font_available else "Helvetica", self.config.PDF_TITLE_FONT_SIZE)
                c.drawString(layout['start_x'], y_position, f"【{category_name}】")
                y_position -= 30
                c.setFont("SimHei" if font_available else "Helvetica", self.config.PDF_CONTENT_FONT_SIZE)
            else:
                # 添加题目图像
                img_path = os.path.join(self.organized_dir, f"{i + 1:03d}_{name}.png")
                img_reader = ImageReader(img_path)
                
                # 计算图像尺寸
                img_width, img_height = calculate_image_dimensions(img.size, layout['content_width'])
                
                # 检查是否需要新页面
                if y_position - img_height < layout['page_height'] - layout['content_height']:
                    c.showPage()
                    y_position = layout['start_y']
                    c.setFont("SimHei" if font_available else "Helvetica", self.config.PDF_CONTENT_FONT_SIZE)
                
                # 绘制图像
                c.drawImage(img_reader, layout['start_x'], y_position - img_height, 
                           width=img_width, height=img_height)
                y_position -= img_height + 20

        c.save()
        print(f"{category_name} PDF文件已生成: {pdf_path}")
        return pdf_path
    
    def _setup_font(self, canvas_obj):
        """设置中文字体"""
        try:
            # 尝试使用系统中的中文字体
            pdfmetrics.registerFont(TTFont('SimHei', self.config.DEFAULT_FONT_PATH))
            canvas_obj.setFont("SimHei", self.config.PDF_CONTENT_FONT_SIZE)
            return True
        except Exception as e:
            print(f"无法加载中文字体: {str(e)}")
            # 如果找不到指定字体，使用默认字体
            canvas_obj.setFont("Helvetica", self.config.PDF_CONTENT_FONT_SIZE)
            return False


class ResultSaver:
    """结果保存器 - 保存处理后的图像和生成PDF"""
    
    def __init__(self, config, output_dir):
        self.config = config
        self.output_dir = output_dir
        self.organized_dir = os.path.join(output_dir, "organized")
        self.pdf_generator = PDFGenerator(config, output_dir)
    
    def save_question(self, question_img, filename):
        """保存单个题目图像"""
        filepath = os.path.join(self.output_dir, filename)
        Image.fromarray(question_img).save(filepath)
        return filepath
    
    def save_organized_results(self, result_images, category_order):
        """保存整理后的结果并生成PDF"""
        # 保存为PNG图像
        for i, (img, name) in enumerate(result_images):
            img.save(os.path.join(self.organized_dir, f"{i + 1:03d}_{name}.png"))
        
        # 生成PDF
        return self.pdf_generator.generate_pdf(result_images, category_order)
    
    def prepare_category_results(self, category, questions, config):
        """准备类别结果数据"""
        result_images = []
        # 添加类别标题
        title = create_category_header(category, config)
        result_images.append((title, "header"))
        # 添加当前类别的题目
        for q in questions:
            result_images.append((Image.fromarray(q['image']), q['id']))
        return result_images