#!/usr/bin/env python3
"""
전체 시스템 통합 테스트 스크립트
"""

import sys
import logging
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
sys.path.append(str(Path(__file__).parent))

# 모듈 import
from modules.document_loader import DocumentLoader
from modules.document_classifier import DocumentClassifier
from modules.template_loader import TemplateLoader
from modules.attention_evaluator import AttentionEvaluator
from modules.feedback_formatter import FeedbackFormatter
from modules.evaluation_selector import EvaluationSelector

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_module_imports():
    """모듈 import 테스트"""
    print("🧪 모듈 Import 테스트")
    print("=" * 50)
    
    try:
        # 각 모듈 초기화 테스트
        document_loader = DocumentLoader()
        print("✅ DocumentLoader 초기화 성공")
        
        document_classifier = DocumentClassifier()
        print("✅ DocumentClassifier 초기화 성공")
        
        template_loader = TemplateLoader()
        print("✅ TemplateLoader 초기화 성공")
        
        attention_evaluator = AttentionEvaluator()
        print("✅ AttentionEvaluator 초기화 성공")
        
        feedback_formatter = FeedbackFormatter()
        print("✅ FeedbackFormatter 초기화 성공")
        
        evaluation_selector = EvaluationSelector()
        print("✅ EvaluationSelector 초기화 성공")
        
        print("\n🎉 모든 모듈 초기화 성공!")
        return True
        
    except Exception as e:
        print(f"❌ 모듈 초기화 실패: {e}")
        logger.error(f"모듈 초기화 실패: {e}")
        return False

def test_evaluation_flow():
    """평가 플로우 테스트"""
    print("\n\n🔄 평가 플로우 테스트")
    print("=" * 50)
    
    try:
        # 시스템 초기화
        document_loader = DocumentLoader()
        document_classifier = DocumentClassifier()
        template_loader = TemplateLoader()
        attention_evaluator = AttentionEvaluator()
        feedback_formatter = FeedbackFormatter()
        evaluation_selector = EvaluationSelector()
        
        # 테스트 텍스트
        test_text = """
        흡수성 마그네슘 합금을 이용한 정형용 이식의료기기 허가심사 가이드라인
        
        본 문서는 흡수성 마그네슘 합금을 이용한 정형용 이식의료기기의 허가심사에 
        필요한 기술문서 작성 방법과 평가 기준을 제시합니다.
        
        제품의 안전성과 효과성을 검증하기 위한 임상시험 데이터와 비임상 데이터가 
        포함되어야 하며, 위험관리계획서와 함께 제출되어야 합니다.
        """
        
        print("📝 테스트 텍스트 생성 완료")
        
        # 1. 제품 유형 추론
        print("\n1️⃣ 제품 유형 추론")
        criteria_data, product_type, confidence = evaluation_selector.select_evaluation_criteria(test_text, template_loader)
        print(f"   추론 결과: {product_type} (신뢰도: {confidence:.2f})")
        
        if criteria_data:
            print("   ✅ 평가 기준 로딩 성공")
        else:
            print("   ⚠️ 평가 기준 로딩 실패")
        
        # 2. 문서 유형 분류
        print("\n2️⃣ 문서 유형 분류")
        classification_result = document_classifier.classify_document(test_text)
        document_type = classification_result["document_type"]
        print(f"   분류 결과: {document_type} (신뢰도: {classification_result['confidence']})")
        
        # 3. 평가 기준 준비
        print("\n3️⃣ 평가 기준 준비")
        if criteria_data:
            # 새로운 형식을 기존 형식으로 변환
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
            # 기본 평가 기준 사용
            evaluation_criteria = {
                "정확성": {
                    "weight": 0.3,
                    "description": "정확한 정보가 기술되었는가",
                    "sub_criteria": ["정보의 정확성", "데이터의 신뢰성"]
                },
                "표현력": {
                    "weight": 0.2,
                    "description": "자연스럽고 명확한 표현인가",
                    "sub_criteria": ["문장의 명확성", "이해의 용이성"]
                }
            }
        
        print(f"   평가 기준 수: {len(evaluation_criteria)}")
        
        # 4. 문서 평가
        print("\n4️⃣ 문서 평가")
        evaluation_result = attention_evaluator.evaluate_document(
            test_text, evaluation_criteria, document_type
        )
        
        if evaluation_result:
            print("   ✅ 평가 완료")
            print(f"   총점: {evaluation_result.get('총점', 'N/A')}")
        else:
            print("   ❌ 평가 실패")
        
        print("\n🎉 평가 플로우 테스트 완료!")
        return True
        
    except Exception as e:
        print(f"❌ 평가 플로우 테스트 실패: {e}")
        logger.error(f"평가 플로우 테스트 실패: {e}")
        return False

def test_file_operations():
    """파일 작업 테스트"""
    print("\n\n📁 파일 작업 테스트")
    print("=" * 50)
    
    try:
        # 지원 파일 형식 확인
        document_loader = DocumentLoader()
        supported_formats = document_loader.get_supported_formats()
        print(f"지원 파일 형식: {supported_formats}")
        
        # 새로운 가이드라인 구조 확인
        guidelines_dir = Path("guidelines")
        if guidelines_dir.exists():
            print("✅ guidelines 디렉토리 존재")
            
            # document 폴더 확인
            document_dir = guidelines_dir / "document"
            if document_dir.exists():
                doc_files = list(document_dir.glob("*.json"))
                print(f"문서 유형별 가이드라인 파일 수: {len(doc_files)}")
                for file_path in doc_files:
                    print(f"   📄 {file_path.name}")
            
            # product 폴더 확인
            product_dir = guidelines_dir / "product"
            if product_dir.exists():
                prod_files = list(product_dir.glob("*.json"))
                print(f"제품별 가이드라인 파일 수: {len(prod_files)}")
                for file_path in prod_files[:5]:  # 처음 5개만 표시
                    print(f"   📄 {file_path.name}")
                if len(prod_files) > 5:
                    print(f"   ... 외 {len(prod_files) - 5}개 파일")
        else:
            print("⚠️ guidelines 디렉토리가 존재하지 않습니다.")
        
        print("\n🎉 파일 작업 테스트 완료!")
        return True
        
    except Exception as e:
        print(f"❌ 파일 작업 테스트 실패: {e}")
        logger.error(f"파일 작업 테스트 실패: {e}")
        return False

def test_available_options():
    """사용 가능한 옵션 테스트"""
    print("\n\n📋 사용 가능한 옵션 테스트")
    print("=" * 50)
    
    try:
        # 문서 유형 목록
        template_loader = TemplateLoader()
        document_types = template_loader.get_available_types()
        print(f"문서 유형 수: {len(document_types)}")
        for doc_type in document_types:
            print(f"   📄 {doc_type}")
        
        # 제품 유형 목록
        evaluation_selector = EvaluationSelector()
        product_types = evaluation_selector.get_available_products()
        print(f"\n제품 유형 수: {len(product_types)}")
        for product_type in product_types:
            print(f"   🏷️ {product_type}")
        
        # 지원 파일 형식
        document_loader = DocumentLoader()
        supported_formats = document_loader.get_supported_formats()
        print(f"\n지원 파일 형식: {supported_formats}")
        
        print("\n🎉 사용 가능한 옵션 테스트 완료!")
        return True
        
    except Exception as e:
        print(f"❌ 사용 가능한 옵션 테스트 실패: {e}")
        logger.error(f"사용 가능한 옵션 테스트 실패: {e}")
        return False

def main():
    """메인 테스트 함수"""
    print("🚀 전체 시스템 통합 테스트 시작")
    print("=" * 60)
    
    test_results = []
    
    # 각 테스트 실행
    test_results.append(test_module_imports())
    test_results.append(test_evaluation_flow())
    test_results.append(test_file_operations())
    test_results.append(test_available_options())
    
    # 결과 요약
    print("\n\n📊 테스트 결과 요약")
    print("=" * 50)
    
    passed = sum(test_results)
    total = len(test_results)
    
    print(f"통과: {passed}/{total}")
    
    if passed == total:
        print("🎉 모든 테스트 통과!")
        print("✅ 시스템이 정상적으로 작동합니다.")
    else:
        print("⚠️ 일부 테스트 실패")
        print("❌ 시스템에 문제가 있을 수 있습니다.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 