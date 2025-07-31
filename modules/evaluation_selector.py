"""
문서 내용 기반 제품 추론 및 평가 기준 자동 선택 모듈 (새로운 구조)
"""

import json
import logging
import re
from typing import Dict, List, Optional, Tuple
from pathlib import Path
try:
    from langchain.llms import Ollama
    from langchain.prompts import PromptTemplate
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EvaluationSelector:
    """문서 내용 기반 제품 추론 및 평가 기준 선택 클래스 (새로운 구조)"""
    
    def __init__(self, model_name: str = "mistral"):
        self.model_name = model_name
        self.product_keywords = {}
        self.llm = None
        
        # 초기화
        self._load_product_keywords()
        self._init_llm()
    
    def _init_llm(self):
        """LLM 모델 초기화"""
        self.llm = None
        if OLLAMA_AVAILABLE:
            try:
                self.llm = Ollama(model=self.model_name)
                logger.info(f"Ollama 모델 '{self.model_name}' 초기화 완료")
            except Exception as e:
                logger.warning(f"Ollama 모델 초기화 실패: {e}")
        else:
            logger.warning("Ollama 라이브러리가 설치되지 않아 AI 추론 기능을 사용할 수 없습니다.")
    
    def _load_product_keywords(self):
        """제품 키워드 로드"""
        self.product_keywords = {
            "흡수성마그네슘": ["흡수성 마그네슘", "마그네슘 합금", "흡수성", "마그네슘", "흡수성마그네슘"],
            "혈액제제": ["혈액", "혈장", "혈액제", "혈액 제제"],
            "휴대용초음파": ["초음파", "휴대용", "진단기기", "초음파 진단"],
            "혈관용스텐트": ["스텐트", "혈관", "혈관용", "혈관 스텐트"],
            "화장품": ["화장품", "화장", "화장용품"],
            "환자맞춤형3D바이오프린팅": ["바이오프린팅", "3D", "환자맞춤형", "바이오 프린팅"],
            "혈관카테터": ["카테터", "혈관 카테터", "혈관카테터"],
            "혈압감시기": ["혈압", "혈압계", "혈압 감시", "혈압 모니터"],
            "혈액냉동고": ["혈액 냉동", "냉동고", "혈액 보관"],
            "환자감시장치": ["환자 감시", "모니터링", "환자 모니터"],
            "휠체어동력보조장치": ["휠체어", "동력 보조", "휠체어 보조"],
            "혁신의료기기": ["혁신", "혁신 의료", "혁신 기기"],
            "희소의료기기": ["희소", "희소 의료", "희소 기기"],
            "확장보관온도조건": ["확장 보관", "온도 조건", "보관 조건"],
            "휴대형의료영상전송장치": ["의료 영상", "영상 전송", "휴대형 영상"]
        }
    
    def infer_product_type(self, text: str) -> Tuple[str, float, str]:
        """
        문서 내용에서 제품 유형 추론
        
        Returns:
            Tuple[str, float, str]: (제품 슬러그, 신뢰도, 추론 방법)
        """
        try:
            logger.info("제품 유형 추론 시작")
            
            # 1. 키워드 기반 추론
            keyword_result = self._infer_by_keywords(text)
            
            # 2. AI 모델 기반 추론 (가능한 경우)
            ai_result = None
            if self.llm:
                ai_result = self._infer_by_ai(text)
            
            # 3. 결과 통합
            if ai_result and ai_result[1] > keyword_result[1]:
                product_slug, confidence, method = ai_result
            else:
                product_slug, confidence, method = keyword_result
            
            logger.info(f"제품 유형 추론 완료: {product_slug} (신뢰도: {confidence:.2f}, 방법: {method})")
            return product_slug, confidence, method
            
        except Exception as e:
            logger.error(f"제품 유형 추론 실패: {e}")
            return "기타", 0.0, "추론 실패"
    
    def _infer_by_keywords(self, text: str) -> Tuple[str, float, str]:
        """키워드 기반 제품 유형 추론"""
        text_lower = text.lower()
        best_match = "기타"
        best_score = 0.0
        
        for product_slug, keywords in self.product_keywords.items():
            score = 0.0
            for keyword in keywords:
                if keyword.lower() in text_lower:
                    score += 1.0
            
            if score > best_score:
                best_score = score
                best_match = product_slug
        
        # 신뢰도 계산 (키워드 매칭 수 / 전체 키워드 수)
        confidence = min(best_score / 3.0, 0.8) if best_score > 0 else 0.0
        
        return best_match, confidence, "키워드 매칭"
    
    def _infer_by_ai(self, text: str) -> Tuple[str, float, str]:
        """AI 모델 기반 제품 유형 추론"""
        try:
            prompt_template = PromptTemplate(
                input_variables=["text"],
                template="""
다음 의약품 관련 문서 내용을 분석하여 제품 유형을 추론해주세요.

문서 내용:
{text}

다음 제품 유형 중에서 가장 적합한 것을 선택해주세요:
- 흡수성마그네슘: 흡수성 마그네슘 합금 관련 의료기기
- 혈액제제: 혈액, 혈장 관련 제제
- 휴대용초음파: 휴대용 초음파 진단기기
- 혈관용스텐트: 혈관용 스텐트
- 화장품: 화장품 관련
- 혈관카테터: 혈관 카테터
- 혈압감시기: 혈압 감시 장치
- 혈액냉동고: 혈액 냉동 보관 장치
- 환자감시장치: 환자 모니터링 장치
- 휠체어동력보조장치: 휠체어 동력 보조 장치
- 혁신의료기기: 혁신 의료기기
- 희소의료기기: 희소 의료기기
- 확장보관온도조건: 확장 보관 온도 조건
- 휴대형의료영상전송장치: 휴대형 의료 영상 전송 장치
- 기타: 위에 해당하지 않는 경우

응답 형식:
제품유형: [선택한 제품 유형]
신뢰도: [0.0-1.0 사이의 신뢰도]
"""
            )
            
            prompt = prompt_template.format(text=text[:2000])  # 텍스트 길이 제한
            if self.llm:
                response = self.llm(prompt)
            else:
                return "기타", 0.0, "LLM 없음"
            
            # 응답 파싱
            product_match = re.search(r'제품유형:\s*(\w+)', response)
            confidence_match = re.search(r'신뢰도:\s*([0-9.]+)', response)
            
            if product_match and confidence_match:
                product_slug = product_match.group(1)
                confidence = float(confidence_match.group(1))
                
                # 제품명 매핑 (AI가 축약형으로 응답할 경우)
                if product_slug == "흡수성":
                    product_slug = "흡수성마그네슘"
                
                return product_slug, confidence, "AI 추론"
            else:
                return "기타", 0.0, "AI 응답 파싱 실패"
            
        except Exception as e:
            logger.error(f"AI 추론 실패: {e}")
            return "기타", 0.0, "AI 추론 실패"
    
    def get_available_products(self) -> List[str]:
        """사용 가능한 제품 목록 반환"""
        return list(self.product_keywords.keys())
    
    def add_product_keyword(self, product_slug: str, keywords: List[str]):
        """새로운 제품 키워드 추가 (관리자용)"""
        self.product_keywords[product_slug] = keywords
        logger.info(f"제품 키워드 추가: {product_slug} -> {keywords}")
    
    def select_evaluation_criteria(self, text: str, template_loader=None) -> Tuple[Optional[Dict], str, float]:
        """
        문서 내용을 기반으로 평가 기준 선택 (새로운 구조)
        
        Args:
            text: 문서 텍스트
            template_loader: TemplateLoader 인스턴스 (None이면 새로 생성)
        
        Returns:
            Tuple[Optional[Dict], str, float]: (평가 기준 데이터, 제품 유형, 신뢰도)
        """
        try:
            # TemplateLoader 인스턴스 생성
            if template_loader is None:
                from .template_loader import TemplateLoader
                template_loader = TemplateLoader()
            
            # 제품 유형 추론
            product_slug, confidence, method = self.infer_product_type(text)
            
            # 제품별 가이드라인/평가 기준 로드
            product_guidelines = template_loader.get_guidelines(product_slug, type_hint="product")
            
            if product_guidelines:
                logger.info(f"제품별 가이드라인 로드 완료: {product_slug}")
                # 가이드라인을 평가 기준 형태로 변환
                criteria_data = {
                    "document_type": product_slug,
                    "evaluation_criteria": product_guidelines
                }
                return criteria_data, product_slug, confidence
            else:
                logger.warning(f"제품별 가이드라인을 찾을 수 없습니다: {product_slug}")
                return None, product_slug, confidence
                
        except Exception as e:
            logger.error(f"평가 기준 선택 실패: {e}")
            return None, "기타", 0.0 