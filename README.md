# 试卷拆分工具（模块化版本）

一个基于pipeline模式的模块化试卷处理工具，可以自动将PDF试卷中的题目分割、识别、分类并整理成PDF文件。

## 项目结构

```
paper_splitter/
├── __init__.py        # 包初始化文件
├── config.py          # 配置模块 - 存储常量和配置
├── utils.py           # 工具函数模块 - 通用工具函数
├── data_processor.py  # 数据处理模块 - PDF转换和题目分割
├── classifier.py      # 分类器模块 - 题目分类功能
├── result_processor.py # 结果处理模块 - 组织和保存结果
├── pipeline.py        # 管道模块 - 核心流程控制
├── main.py            # 主入口文件
├── requirements.txt   # 依赖包列表
└── README.md          # 项目说明文档
```

## 功能模块说明

1. **配置模块 (config.py)**
   - 存储所有常量和配置项
   - 包括路径配置、API配置、默认分类等

2. **工具模块 (utils.py)**
   - 文件处理工具
   - 图像预处理函数
   - PDF页面布局计算

3. **数据处理模块 (data_processor.py)**
   - PDF转图像功能
   - 题目检测和分割
   - OCR文本提取

4. **分类器模块 (classifier.py)**
   - 基于DeepSeek API的分类器
   - 基于关键词的后备分类器
   - 分类器接口设计

5. **结果处理模块 (result_processor.py)**
   - 结果组织功能
   - PDF生成器
   - 图像保存功能

6. **管道模块 (pipeline.py)**
   - PipelineStage基类
   - 各个处理阶段的实现
   - 完整处理流程控制

## 使用方法

### 命令行使用

```bash
python -m paper_splitter.main path/to/your/paper.pdf --output_dir output_folder --api_key your_api_key
```

### 参数说明

- `pdf_path`: PDF文件路径（必需）
- `--output_dir`: 输出目录（默认：output_questions）
- `--api_key`: DeepSeek API密钥（可选，默认使用配置中的密钥）
- `--categories`: 自定义分类类别，用逗号分隔（例如：代数,几何,概率统计）

### 作为库使用

```python
from paper_splitter.config import *
from paper_splitter.pipeline import PaperProcessingPipeline

# 创建配置实例
class Config:
    pass
config = Config()
# 加载配置项...（参考main.py）

# 创建处理管道
pipeline = PaperProcessingPipeline(
    config=config,
    pdf_path="path/to/paper.pdf",
    output_dir="output_folder",
    api_key="your_api_key",
    custom_categories=["代数", "几何", "概率统计"]
)

# 执行处理
result = pipeline.execute()
```

## 安装依赖

```bash
pip install -r requirements.txt
```

## 注意事项

1. 确保安装了Tesseract OCR，并在config.py中设置正确的路径
2. 确保安装了Poppler，并在config.py中设置正确的路径
3. DeepSeek API密钥需要有效才能使用API分类功能
4. 目前支持中文和英文的混合识别

## 扩展方法

1. **添加新的处理阶段**：
   - 继承PipelineStage基类
   - 实现execute方法
   - 在PaperProcessingPipeline中添加到stages列表

2. **使用自定义分类器**：
   - 继承QuestionClassifier基类
   - 实现classify方法
   - 在ClassificationStage中替换默认分类器

3. **自定义配置**：
   - 可以在运行时修改config对象的属性
   - 也可以直接修改config.py中的默认配置


English version:
Paper Splitting Tool (Modular Version)
A modular exam paper processing tool based on the pipeline pattern that can automatically split, recognize, classify, and organize questions from PDF exam papers into separate PDF files.

Project Structure

```
paper_splitter/
├── __init__.py        # Package initialization file
├── config.py          # Configuration module - stores constants and settings
├── utils.py           # Utilities module - common utility functions
├── data_processor.py  # Data processing module - PDF conversion and question splitting
├── classifier.py      # Classifier module - question classification functionality
├── result_processor.py # Result processing module - organizes and saves results
├── pipeline.py        # Pipeline module - core process control
├── main.py            # Main entry file
├── requirements.txt   # Dependency list
└── README.md          # Project documentation
```
Module Descriptions
1. **Configuration Module (config.py)**
   - Stores all constants and configuration items
   - Includes path configurations, API settings, default classifications, etc.

2. **Utilities Module (utils.py)**
   - File handling utilities
   - Image preprocessing functions
   - PDF page layout calculations

3. **Data Processing Module (data_processor.py)**
   - PDF to image conversion functionality

   - Question detection and splitting

   - OCR text extraction

4. **Classifier Module (classifier.py)**

   - DeepSeek API-based classifier
   - Keyword-based fallback classifier

   - Classifier interface design

5. **Result Processing Module (result_processor.py)**

   - Result organization functionality

   - PDF generator

   - Image saving functionality

6. **Pipeline Module (pipeline.py)**

   - PipelineStage base class

   - Implementation of each processing stage

   - Complete process flow control

## Usage

### Command Line Usage

```bash
python -m paper_splitter.main path/to/your/paper.pdf --output_dir output_folder --api_key your_api_key
```

### Parameter Description

- `pdf_path`: PDF file path (required)
- `--output_dir`:Output directory (default: output_questions)
- `--api_key`: DeepSeek API key (optional, defaults to the key in configuration)
- `--categories`: Custom classification categories, comma-separated (e.g., algebra,geometry,probability_statistics)

### Usage as a Library

```python
from paper_splitter.config import *
from paper_splitter.pipeline import PaperProcessingPipeline

# Create configuration instance
class Config:
    pass
config = Config()
# Load configuration items... (refer to main.py)

# Create processing pipeline
pipeline = PaperProcessingPipeline(
    config=config,
    pdf_path="path/to/paper.pdf",
    output_dir="output_folder",
    api_key="your_api_key",
    custom_categories=["代数", "几何", "概率统计"]
)

# Execute processing
result = pipeline.execute()
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

## Notes

1. Ensure Tesseract OCR is installed and set the correct path in config.py
2. Ensure Poppler is installed and set the correct path in config.py
3. DeepSeek API key needs to be valid to use API classification functionality
4. Currently supports mixed Chinese and English recognition

## Extension Methods

1. **Adding New Processing Stages**：
   - Inherit from the PipelineStage base class
   - Implement the execute method
   - Add to the stages list in PaperProcessingPipeline
   
2. **Using Custom Classifiers**：
    - Inherit from the QuestionClassifier base class
    - Implement the classify method
    - Replace the default classifier in ClassificationStage

3. **Custom Configuration**：
    - You can modify properties of the config object at runtime

    - You can also directly modify the default configuration in config.py