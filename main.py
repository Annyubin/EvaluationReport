"""
의약품 문서 자동 평가 시스템 - 메인 스크립트
"""

import sys
import os
import logging
from pathlib import Path
from typing import Optional

# 모듈 import
from modules.document_loader import DocumentLoader
from modules.document_classifier import DocumentClassifier
from modules.template_loader import TemplateLoader
from modules.attention_evaluator import AttentionEvaluator
from modules.feedback_formatter import FeedbackFormatter
from modules.evaluation_selector import EvaluationSelector

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DocumentEvaluationSystem:
    """의약품 문서 평가 시스템 메인 클래스"""
    
    def __init__(self, model_name: str = "mistral"):
        self.model_name = model_name
        self.document_loader = DocumentLoader()
        self.document_classifier = DocumentClassifier(model_name)
        self.template_loader = TemplateLoader()
        self.evaluator = AttentionEvaluator(model_name)
        self.formatter = FeedbackFormatter()
        self.evaluation_selector = EvaluationSelector()
        
        logger.info("의약품 문서 평가 시스템 초기화 완료")
    
    def evaluate_document(self, file_path: str, document_type: Optional[str] = None, 
                         auto_classify: bool = True) -> Optional[dict]:
        """문서 평가 메인 프로세스"""
        try:
            logger.info(f"문서 평가 시작: {file_path}")
            
            # 1. 문서 텍스트 추출
            logger.info("1단계: 문서 텍스트 추출 중...")
            extracted_text = self.document_loader.load_document(file_path)
            if not extracted_text:
                logger.error("문서 텍스트 추출에 실패했습니다.")
                return None
            page_count = 0  # document_loader는 페이지 수를 반환하지 않음
            logger.info(f"텍스트 추출 완료: {len(extracted_text)} 문자, {page_count} 페이지")
            
            # 2. 제품 유형 추론 및 평가 기준 선택
            logger.info("2단계: 제품 유형 추론 중...")
            criteria_data, product_type, confidence = self.evaluation_selector.select_evaluation_criteria(extracted_text, self.template_loader)
            
            if criteria_data:
                logger.info(f"제품 유형 추론 완료: {product_type} (신뢰도: {confidence:.2f})")
                # 새로운 형식의 평가 기준을 기존 형식으로 변환
                if isinstance(criteria_data.get("evaluation_criteria"), list):
                    evaluation_criteria = {}
                    for criteria in criteria_data["evaluation_criteria"]:
                        evaluation_criteria[criteria["name"]] = {
                            "weight": criteria["weight"],
                            "description": criteria["description"],
                            "sub_criteria": [sub["name"] for sub in criteria["sub_criteria"]]
                        }
                else:
                    evaluation_criteria = criteria_data.get("evaluation_criteria", criteria_data)
            else:
                logger.warning(f"제품 유형 추론 실패: {product_type}")
                evaluation_criteria = self._get_default_criteria()
            
            # 3. 문서 유형 분류 (자동 또는 수동)
            if auto_classify and not document_type:
                logger.info("3단계: 문서 유형 자동 분류 중...")
                classification_result = self.document_classifier.classify_document(extracted_text)
                document_type = classification_result.get("document_type", "기타")
                logger.info(f"문서 유형 분류 완료: {document_type}")
            elif not document_type:
                document_type = "기타"
            
            # 4. 문서 평가
            logger.info("4단계: 문서 평가 중...")
            evaluation_result = self.evaluator.evaluate_document(
                extracted_text, evaluation_criteria, str(document_type)
            )
            logger.info(f"평가 완료 - 총점: {evaluation_result.get('total_score', 0)}")
            
            # 5. 결과 포맷팅
            logger.info("5단계: 결과 포맷팅 중...")
            document_name = Path(file_path).stem
            
            # 콘솔 출력
            console_output = self.formatter.format_console_output(evaluation_result, str(document_type))
            print(console_output)
            
            # 결과 저장
            try:
                md_file_path = self.formatter.save_markdown_report(
                    evaluation_result, str(document_type), document_name
                )
                logger.info(f"마크다운 보고서 저장 완료: {md_file_path}")
                
                json_file_path = self.formatter.save_json_report(
                    evaluation_result, str(document_type), document_name
                )
                logger.info(f"JSON 보고서 저장 완료: {json_file_path}")
                
            except Exception as e:
                logger.error(f"결과 저장 중 오류 발생: {e}")
            
            # 종합 결과 반환
            result = {
                "success": True,
                "document_type": document_type,
                "page_count": page_count,
                "text_length": len(extracted_text),
                "evaluation_result": evaluation_result,
                "files": {
                    "markdown": md_file_path if 'md_file_path' in locals() else None,
                    "json": json_file_path if 'json_file_path' in locals() else None
                }
            }
            
            logger.info("문서 평가 완료")
            return result
            
        except Exception as e:
            logger.error(f"문서 평가 중 오류 발생: {e}")
            return {
                "success": False,
                "error": str(e),
                "document_type": document_type if 'document_type' in locals() else None
            }
    
    def _get_default_criteria(self) -> dict:
        """기본 평가 기준 반환"""
        return {
            "정확성": {
                "weight": 0.3,
                "description": "정확한 정보가 기술되었는가",
                "sub_criteria": ["정보의 정확성", "데이터의 신뢰성"]
            },
            "표현력": {
                "weight": 0.2,
                "description": "자연스럽고 명확한 표현인가",
                "sub_criteria": ["문장의 명확성", "이해의 용이성"]
            },
            "항목누락": {
                "weight": 0.3,
                "description": "필수 항목이 모두 포함되었는가",
                "sub_criteria": ["필수 항목 포함", "완성도"]
            },
            "형식적합성": {
                "weight": 0.2,
                "description": "규제 양식에 맞게 작성되었는가",
                "sub_criteria": ["양식 준수", "규정 준수"]
            }
        }
    
    def get_available_document_types(self) -> list:
        """사용 가능한 문서 유형 목록 반환"""
        return self.template_loader.get_available_types()
    
    def get_document_type_description(self, document_type: str) -> str:
        """문서 유형 설명 반환"""
        return self.document_classifier.get_document_type_description(document_type)
    
    def get_supported_formats(self) -> list:
        """지원하는 파일 형식 목록 반환"""
        return self.document_loader.get_supported_formats()

def main():
    """메인 함수"""
    if len(sys.argv) < 2:
        print("사용법: python main.py <문서파일경로> [문서유형]")
        print("예시: python main.py document.hwp 위험관리계획서")
        print("예시: python main.py document.docx (자동 분류)")
        print("지원 형식: .hwp, .docx")
        return
    
    file_path = sys.argv[1]
    document_type = sys.argv[2] if len(sys.argv) > 2 else None
    
    # 파일 존재 확인
    if not os.path.exists(file_path):
        print(f"오류: 파일을 찾을 수 없습니다: {file_path}")
        return
    
    # 파일 확장자 확인
    supported_formats = ['.hwp', '.docx']
    file_ext = Path(file_path).suffix.lower()
    if file_ext not in supported_formats:
        print(f"오류: 지원하지 않는 파일 형식입니다: {file_ext}")
        print(f"지원 형식: {', '.join(supported_formats)}")
        return
    
    try:
        # 평가 시스템 초기화
        print("의약품 문서 평가 시스템을 초기화하는 중...")
        system = DocumentEvaluationSystem()
        
        # 문서 평가 실행
        print(f"\n문서 평가를 시작합니다: {file_path}")
        if document_type:
            print(f"지정된 문서 유형: {document_type}")
        else:
            print("문서 유형을 자동으로 분류합니다.")
        
        result = system.evaluate_document(file_path, document_type)
        
        if result and result["success"]:
            print(f"\n✅ 평가가 성공적으로 완료되었습니다!")
            print(f"📄 문서 유형: {result['document_type']}")
            print(f"📊 총점: {result['evaluation_result']['총점']}/10")
            
            if result["files"]["markdown"]:
                print(f"📝 마크다운 보고서: {result['files']['markdown']}")
            if result["files"]["json"]:
                print(f"📋 JSON 결과: {result['files']['json']}")
        elif result:
            print(f"\n❌ 평가 중 오류가 발생했습니다: {result['error']}")
            print("다시 시도해보거나 다른 문서를 사용해보세요.")
        else:
            print(f"\n❌ 문서 텍스트 추출에 실패했습니다.")
            print("파일 형식을 확인하거나 다른 문서를 사용해보세요.")
    
    except KeyboardInterrupt:
        print("\n\n사용자에 의해 평가가 중단되었습니다.")
    except Exception as e:
        print(f"\n❌ 예상치 못한 오류가 발생했습니다: {e}")
        print("시스템 관리자에게 문의하세요.")

if __name__ == "__main__":
    main() 