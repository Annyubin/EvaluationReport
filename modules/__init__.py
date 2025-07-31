"""
의약품 문서 평가 시스템 모듈 패키지
"""

from .document_loader import DocumentLoader
from .document_classifier import DocumentClassifier
from .template_loader import TemplateLoader
from .attention_evaluator import AttentionEvaluator
from .feedback_formatter import FeedbackFormatter
from .evaluation_selector import EvaluationSelector

__all__ = [
    'DocumentLoader',
    'DocumentClassifier', 
    'TemplateLoader',
    'AttentionEvaluator',
    'FeedbackFormatter',
    'EvaluationSelector'
]

__version__ = '1.0.0'
__author__ = '의약품 문서 평가 시스템 개발팀' 