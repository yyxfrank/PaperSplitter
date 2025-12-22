# 分类器模块 - 使用 OpenAI 官方 SDK 实现题目分类（支持文本和文件）

import logging
from typing import List, Dict
from openai import OpenAI

from config import *


class QuestionClassifier:
    """题目分类器基类"""

    def __init__(self, config, categories=None, keyword_map=None):
        self.config = config
        self.categories = categories or getattr(config, "DEFAULT_CATEGORIES", [])
        self.keyword_map = keyword_map or getattr(config, "DEFAULT_KEYWORD_MAP", {})

    def classify(self, text: str):
        raise NotImplementedError("子类必须实现 classify 方法")

    def set_categories(self, categories, keyword_map=None):
        self.categories = categories
        if keyword_map:
            self.keyword_map = keyword_map
        else:
            self.keyword_map = {k: v for k, v in self.keyword_map.items() if k in categories}

    def get_categories(self):
        return self.categories

    def get_keyword_map(self):
        return self.keyword_map


class KeywordBasedClassifier(QuestionClassifier):
    """基于关键词匹配的后备分类器"""

    def classify(self, text: str):
        scores = {label: 0 for label in self.categories}
        for label, keywords in self.keyword_map.items():
            for kw in keywords:
                if kw in text:
                    scores[label] += 1

        max_score = max(scores.values())
        if max_score > 0:
            for label in self.categories:
                if scores[label] == max_score:
                    return label
        return self.categories[0]


class OpenAIClassifier(QuestionClassifier):
    """基于 OpenAI 官方 SDK 的题目分类器，支持文本与文件"""

    def __init__(self, config, categories=None, keyword_map=None, fallback_classifier=None):
        super().__init__(config, categories, keyword_map)
        self.model = getattr(config, "OPENAI_MODEL", "gpt-4o-mini")
        self.client = OpenAI()
        self.fallback_classifier = fallback_classifier or KeywordBasedClassifier(config, categories, keyword_map)

    def classify(self, text: str):
        """文本直接分类"""

        try:
            prompt = (
                f"请将以下题目分类到这些类别之一：{', '.join(self.categories)}。\n\n"
                f"题目内容：{text}\n\n"
                "只返回类别名称，不要解释或补充说明。"
            )

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "你是一名教育测评专家，负责将题目准确分类。"},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
                max_tokens=50,
            )

            category = response.choices[0].message.content.strip()
            return self._validate_category(category)

        except Exception as e:
            logging.error(f"OpenAI API 调用失败: {e}")
            return self.fallback_classifier.classify(text)

    def classify_from_file(self, file_path: str, question_text: str):
        """结合文件和问题内容进行分类（如整份试卷PDF）"""
        try:
            # 上传文件
            file = self.client.files.create(file=open(file_path, "rb"), purpose="user_data")

            # 构造请求
            response = self.client.responses.create(
                model=self.model,
                input=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_file", "file_id": file.id},
                            {"type": "input_text", "text": (
                                f"请根据上传的试卷文件，判断以下题目属于哪个类别：{', '.join(self.categories)}。\n\n"
                                f"题目内容：{question_text}\n\n"
                                "只返回类别名称，不要任何解释。"
                            )},
                        ],
                    }
                ],
            )

            # 取出结果文本
            category = response.output_text.strip()
            return self._validate_category(category)

        except Exception as e:
            logging.error(f"文件分类调用失败: {e}")
            return self.fallback_classifier.classify(question_text)

    def _validate_category(self, category: str):
        """验证模型输出是否有效"""
        if category in self.categories:
            return category
        for label in self.categories:
            if label in category:
                return label
        return self.categories[0]
