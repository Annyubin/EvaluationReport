"""
문서 유형을 감지하는 모듈
선택지 기반과 Mistral 모델을 사용한 자동 감지 지원
"""

import json
import logging
from typing import List, Dict, Optional
import ollama
from langchain.llms import Ollama
from langchain.prompts import PromptTemplate

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DocumentClassifier:
    """문서 유형을 감지하는 클래스"""
    
    def __init__(self, model_name: str = "mistral"):
        self.model_name = model_name
        self.document_types = [
            "CTD 문서",
            "DMF 자료",
            "MFDS 제출자료",
            "기술문서 요약서",
            "변경보고서",
            "사용자 설명서(첨부문서)",
            "약물감시계획서",
            "위험관리계획서",
            "임상시험계획서",
            "품질관리문서",
            "제품설명서",
            "안전성자료",
            "기타"
        ]
        
        # Ollama 모델 초기화
        try:
            self.llm = Ollama(model=model_name)
            logger.info(f"Ollama 모델 '{model_name}' 초기화 완료")
        except Exception as e:
            logger.warning(f"Ollama 모델 초기화 실패: {e}")
            self.llm = None
    
    def classify_with_keywords(self, text: str) -> str:
        """키워드 기반으로 문서 유형 분류"""
        text_lower = text.lower()
        
        # 키워드 매핑
        keyword_mapping = {
            "CTD 문서": ["ctd", "common technical document", "기술문서"],
            "DMF 자료": ["dmf", "drug master file", "마스터파일"],
            "MFDS 제출자료": ["mfds", "식약처", "제출", "신청"],
            "기술문서 요약서": ["기술문서", "요약서", "summary", "기술요약"],
            "변경보고서": ["변경", "보고서", "change", "report"],
            "사용자 설명서(첨부문서)": ["사용자", "설명서", "user", "manual", "첨부문서"],
            "약물감시계획서": ["약물감시", "감시", "pharmacovigilance", "pvp"],
            "위험관리계획서": ["위험관리", "위험관리계획", "risk management", "rmp"],
            "임상시험계획서": ["임상시험", "clinical trial", "protocol"],
            "품질관리문서": ["품질관리", "품질", "quality", "제조"],
            "제품설명서": ["제품설명", "product information", "pi", "제품정보"],
            "안전성자료": ["안전성", "safety", "부작용", "adverse"]
        }
        
        # 점수 계산
        scores = {}
        for doc_type, keywords in keyword_mapping.items():
            score = 0
            for keyword in keywords:
                if keyword in text_lower:
                    score += 1
            scores[doc_type] = score
        
        # 가장 높은 점수의 문서 유형 반환
        if scores:
            best_match = max(scores.items(), key=lambda x: x[1])
            if best_match[1] > 0:
                return best_match[0]
        
        return "기타"
    
    def classify_with_mistral(self, text: str) -> str:
        """Mistral 모델을 사용하여 문서 유형 분류"""
        if not self.llm:
            logger.warning("Mistral 모델을 사용할 수 없어 키워드 기반 분류로 대체합니다.")
            return self.classify_with_keywords(text)
        
        try:
            # 프롬프트 템플릿
            prompt_template = PromptTemplate(
                input_variables=["text", "document_types"],
                template="""
다음 텍스트를 분석하여 문서 유형을 분류해주세요.

가능한 문서 유형:
{document_types}

텍스트:
{text}

위 텍스트가 어떤 문서 유형에 해당하는지 정확히 판단하여 위의 목록에서 하나만 선택해주세요.
답변은 반드시 위 목록에 있는 정확한 문서 유형명만 출력해주세요.
"""
            )
            
            # 프롬프트 생성
            prompt = prompt_template.format(
                text=text[:2000],  # 너무 긴 텍스트는 잘라서 사용
                document_types=", ".join(self.document_types)
            )
            
            # 모델 추론
            response = self.llm(prompt)
            
            # 응답 정리
            response = response.strip().lower()
            
            # 응답에서 문서 유형 찾기
            for doc_type in self.document_types:
                if doc_type.lower() in response:
                    logger.info(f"Mistral 모델이 '{doc_type}'로 분류했습니다.")
                    return doc_type
            
            # 매칭되지 않으면 키워드 기반으로 대체
            logger.warning("Mistral 응답에서 문서 유형을 찾을 수 없어 키워드 기반으로 대체합니다.")
            return self.classify_with_keywords(text)
            
        except Exception as e:
            logger.error(f"Mistral 분류 중 오류 발생: {e}")
            return self.classify_with_keywords(text)
    
    def classify_document(self, text: str, method: str = "auto") -> Dict[str, any]:
        """문서 유형 분류 (메인 메서드)"""
        try:
            logger.info(f"문서 유형 분류 시작 (방법: {method})")
            
            if method == "keyword":
                doc_type = self.classify_with_keywords(text)
                confidence = "중간"
            elif method == "mistral":
                doc_type = self.classify_with_mistral(text)
                confidence = "높음"
            else:  # auto
                # 먼저 Mistral로 시도, 실패 시 키워드 기반으로 대체
                try:
                    doc_type = self.classify_with_mistral(text)
                    confidence = "높음"
                except:
                    doc_type = self.classify_with_keywords(text)
                    confidence = "중간"
            
            result = {
                "document_type": doc_type,
                "confidence": confidence,
                "method": method,
                "available_types": self.document_types
            }
            
            logger.info(f"문서 유형 분류 완료: {doc_type} (신뢰도: {confidence})")
            return result
            
        except Exception as e:
            logger.error(f"문서 분류 중 오류 발생: {e}")
            return {
                "document_type": "기타",
                "confidence": "낮음",
                "method": "fallback",
                "available_types": self.document_types,
                "error": str(e)
            }
    
    def get_document_type_description(self, doc_type: str) -> str:
        """문서 유형에 대한 설명 반환"""
        descriptions = {
            "CTD 문서": "Common Technical Document의 기술 정보를 담은 문서",
            "DMF 자료": "Drug Master File의 제조 및 품질 관련 자료",
            "MFDS 제출자료": "식약처에 제출하는 의약품 허가 관련 자료",
            "기술문서 요약서": "기술문서의 핵심 내용을 요약한 문서",
            "변경보고서": "제품 변경사항을 보고하는 문서",
            "사용자 설명서(첨부문서)": "사용자를 위한 제품 사용 설명서",
            "약물감시계획서": "약물감시 활동을 위한 계획서",
            "위험관리계획서": "의약품의 위험요인을 식별하고 관리방안을 제시하는 문서",
            "임상시험계획서": "임상시험의 목적, 방법, 절차 등을 기술한 문서",
            "품질관리문서": "의약품의 품질 관리와 관련된 자료",
            "제품설명서": "의약품의 성분, 효능, 용법, 용량 등을 상세히 기술한 문서",
            "안전성자료": "의약품의 안전성과 관련된 데이터와 정보를 담은 문서",
            "기타": "기타 의약품 관련 문서"
        }
        
        return descriptions.get(doc_type, "알 수 없는 문서 유형")
    
    def get_detailed_document_type_description(self, doc_type: str) -> str:
        """문서 유형에 대한 상세한 설명 반환"""
        detailed_descriptions = {
            "CTD 문서": "Common Technical Document(CTD)의 기술 정보를 담은 문서입니다. 의약품 허가 신청 시 제출하는 기술 문서로, 모듈 3(품질), 모듈 4(비임상), 모듈 5(임상)의 상세한 기술 자료를 포함합니다.",
            "DMF 자료": "Drug Master File(DMF)의 제조 및 품질 관련 자료입니다. 원료약품, 첨가제, 포장재 등의 제조공정, 품질관리방법, 분석방법 등 상세한 기술 정보를 담고 있습니다.",
            "MFDS 제출자료": "식약처에 제출하는 의약품 허가 관련 자료입니다. 제품의 안전성, 유효성, 품질에 대한 종합적인 자료를 포함하며, 법적 요구사항에 따라 체계적으로 구성됩니다.",
            "기술문서 요약서": "기술문서의 핵심 내용을 요약한 문서입니다. 복잡한 기술 자료를 간결하게 정리하여 검토자와 의사결정자가 쉽게 이해할 수 있도록 구성됩니다.",
            "변경보고서": "제품 변경사항을 보고하는 문서입니다. 제조공정, 품질관리방법, 포장, 라벨링 등의 변경사항과 그에 대한 영향평가, 안전성 검증 결과를 포함합니다.",
            "사용자 설명서(첨부문서)": "사용자를 위한 제품 사용 설명서입니다. 의료진과 환자에게 제품의 올바른 사용법, 주의사항, 부작용 등을 명확하게 전달하는 중요한 자료입니다.",
            "약물감시계획서": "약물감시 활동을 위한 계획서입니다. 시판 후 안전성 모니터링, 부작용 수집 및 분석, 위험 신호 감지 및 대응 방안 등을 체계적으로 기술합니다.",
            "위험관리계획서": "의약품의 위험요인을 체계적으로 식별하고, 각 위험에 대한 관리방안을 제시하는 문서입니다. ISO 14971 표준에 따라 작성되며, 제품의 안전성을 보장하기 위한 핵심 문서입니다.",
            "임상시험계획서": "임상시험의 목적, 방법, 절차, 통계 분석 계획 등을 상세히 기술한 문서입니다. 임상시험의 과학적 타당성과 윤리적 적절성을 검토하는 기준이 됩니다.",
            "품질관리문서": "의약품의 품질 관리와 관련된 자료입니다. 제조공정, 품질관리방법, 안정성 시험 결과, 분석방법, GMP 준수사항 등을 포함합니다.",
            "제품설명서": "의약품의 성분, 효능, 용법, 용량, 주의사항, 부작용 등을 상세히 기술한 문서입니다. 의료진과 환자에게 제품 정보를 제공하는 중요한 자료입니다.",
            "안전성자료": "의약품의 안전성과 관련된 모든 데이터와 정보를 담은 문서입니다. 부작용 보고, 약물상호작용, 특수인군 사용 경험 등을 포함합니다.",
            "기타": "위에 분류되지 않는 기타 의약품 관련 문서입니다. 필요에 따라 추가 분류가 가능합니다."
        }
        
        return detailed_descriptions.get(doc_type, "해당 문서 유형에 대한 상세한 설명이 준비되지 않았습니다.")
    
    def validate_document_type(self, doc_type: str) -> bool:
        """문서 유형이 유효한지 검증"""
        return doc_type in self.document_types 