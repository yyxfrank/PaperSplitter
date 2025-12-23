# 配置文件 - 存储系统配置和常量

API_RATE_LIMIT_DELAY=30
# Tesseract OCR配置
TESSERACT_PATH = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# 基本OCR配置 - 中英文和数学公式
CUSTOM_CONFIG = r'--oem 3 --psm 6 -l chi_sim+eng+equ'

# 数学符号识别配置
MATH_CONFIG = r'--oem 3 --psm 4 -l chi_sim+eng+equ'

# LaTeX公式识别配置
LATEX_CONFIG = r'--oem 3 --psm 11 -l chi_sim+eng+equ'

# Poppler路径配置
POPPLER_PATH = r'C:\Program Files\Release-25.07.0-0\Library\bin'

# 默认分类类别
DEFAULT_CATEGORIES = ["代数", "几何", "概率统计", "函数", "解析几何", "三角函数"]

# 默认关键词映射
DEFAULT_KEYWORD_MAP = {
    "代数": ["方程", "不等式", "代数", "多项式", "因式分解", "有理数", "无理数"],
    "几何": ["图形", "几何", "三角形", "四边形", "圆", "面积", "体积", "相似", "全等"],
    "概率统计": ["概率", "统计", "期望", "方差", "频率", "分布", "抽样"],
    "函数": ["函数", "定义域", "值域", "单调性", "奇偶性", "周期性", "反函数"],
    "解析几何": ["坐标", "直线", "圆", "椭圆", "双曲线", "抛物线", "向量"],
    "三角函数": ["正弦", "余弦", "正切", "三角函数", "周期", "振幅", "相位"]
}

# 中文字体配置
DEFAULT_FONT_PATH = r"C:\Windows\Fonts\simhei.ttf"

# 标题图像配置
HEADER_IMAGE_WIDTH = 1200
HEADER_IMAGE_HEIGHT = 80
HEADER_COLOR = (73, 109, 137)
HEADER_TEXT_COLOR = (255, 255, 0)
HEADER_FONT_SIZE = 40

# PDF生成配置
PDF_PAGE_SIZE = 'letter'
PDF_MARGIN_LEFT = 50
PDF_MARGIN_TOP = 50
PDF_MARGIN_RIGHT = 50
PDF_MARGIN_BOTTOM = 50
PDF_TITLE_FONT_SIZE = 16
PDF_CONTENT_FONT_SIZE = 12