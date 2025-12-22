# 主入口文件 - 使用模块化结构的试卷处理工具

import os
import argparse
from config import *
from pipeline import PaperProcessingPipeline


def build_config():
    """构建配置对象，加载所有常量"""
    class Config:
        pass

    config = Config()
    config.TESSERACT_PATH = TESSERACT_PATH
    config.CUSTOM_CONFIG = CUSTOM_CONFIG
    config.MATH_CONFIG = MATH_CONFIG
    config.LATEX_CONFIG = LATEX_CONFIG
    config.POPPLER_PATH = POPPLER_PATH
    config.DEFAULT_CATEGORIES = DEFAULT_CATEGORIES
    config.DEFAULT_KEYWORD_MAP = DEFAULT_KEYWORD_MAP
    config.DEFAULT_FONT_PATH = DEFAULT_FONT_PATH
    config.HEADER_IMAGE_WIDTH = HEADER_IMAGE_WIDTH
    config.HEADER_IMAGE_HEIGHT = HEADER_IMAGE_HEIGHT
    config.HEADER_COLOR = HEADER_COLOR
    config.HEADER_TEXT_COLOR = HEADER_TEXT_COLOR
    config.HEADER_FONT_SIZE = HEADER_FONT_SIZE
    config.API_RATE_LIMIT_DELAY = API_RATE_LIMIT_DELAY
    config.PDF_PAGE_SIZE = PDF_PAGE_SIZE
    config.PDF_MARGIN_LEFT = PDF_MARGIN_LEFT
    config.PDF_MARGIN_TOP = PDF_MARGIN_TOP
    config.PDF_MARGIN_RIGHT = PDF_MARGIN_RIGHT
    config.PDF_MARGIN_BOTTOM = PDF_MARGIN_BOTTOM
    config.PDF_TITLE_FONT_SIZE = PDF_TITLE_FONT_SIZE
    config.PDF_CONTENT_FONT_SIZE = PDF_CONTENT_FONT_SIZE
    return config


def main():
    """主函数 - 解析命令行参数并启动处理管道"""
    parser = argparse.ArgumentParser(description='📘 数学试卷自动拆分与分类工具')
    parser.add_argument('pdf_path', nargs='?', help='PDF文件路径')
    parser.add_argument('--output_dir', default='output_questions', help='输出目录')
    parser.add_argument('--categories', help='自定义分类类别，用逗号分隔（如：代数,几何,概率统计）')
    args = parser.parse_args()

    # 如果未提供 pdf_path，则交互式输入
    pdf_path = args.pdf_path
    while not pdf_path:
        pdf_path = input("请输入 PDF 文件路径: ").strip()
        if not pdf_path:
            print("❌ 错误：PDF 文件路径不能为空！")

    pdf_path = pdf_path.strip('"').strip("'")

    # 处理输出目录
    output_dir = args.output_dir
    confirm = input(f"使用默认输出目录 '{output_dir}'？(y/n): ").lower()
    if confirm == 'n':
        new_dir = input("请输入新的输出目录: ").strip()
        if new_dir:
            output_dir = new_dir
            print(f"✅ 使用新输出目录: {output_dir}")
        else:
            print("⚠️ 未提供有效目录，继续使用默认目录。")
    else:
        print(f"✅ 使用默认输出目录: {output_dir}")

    # 处理分类类别
    custom_categories = None
    if args.categories:
        custom_categories = [c.strip() for c in args.categories.split(',')]
        print(f"✅ 使用命令行自定义分类: {custom_categories}")
    else:
        user_input = input("请输入分类类别（用逗号分隔，留空使用默认）: ").strip()
        if user_input:
            custom_categories = [c.strip() for c in user_input.split(',')]
            print(f"✅ 使用交互输入的分类类别: {custom_categories}")
        else:
            print("📘 使用默认分类类别。")

    # 构建配置
    config = build_config()

    # 创建并执行管道
    print("\n🚀 启动试卷处理管道...\n")
    pipeline = PaperProcessingPipeline(
        config=config,
        pdf_path=pdf_path,
        output_dir=output_dir,
        custom_categories=custom_categories
    )

    try:
        result = pipeline.execute()
    except Exception as e:
        print(f"❌ 处理过程中出现错误: {e}")
        return

    pdf_paths = result.get('pdf_paths', [])
    print("\n✅ 处理完成！")
    if pdf_paths:
        print(f"📂 分类结果 PDF 文件已生成：\n  {chr(10).join(pdf_paths)}")
    else:
        print("⚠️ 未生成分类 PDF，请检查题目检测阶段是否正常运行。")
    print(f"\n所有结果保存在：{os.path.abspath(output_dir)}")


if __name__ == "__main__":
    main()
