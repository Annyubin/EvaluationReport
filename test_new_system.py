"""
새로운 평가 시스템 테스트 스크립트
"""

import sys
import os
import tempfile
import json
from pathlib import Path

# 모듈 경로 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from modules.document_loader import DocumentLoader
from modules.attention_evaluator import AttentionEvaluator
from modules.template_loader import TemplateLoader

def test_document_loader():
    """DocumentLoader 테스트"""
    print("=== DocumentLoader 테스트 ===")
    
    loader = DocumentLoader()
    
    # 지원 형식 확인
    supported_formats = loader.get_supported_formats()
    print(f"지원 형식: {supported_formats}")
    
    # 파일 형식 검증 테스트
    test_files = [
        "test.hwp",
        "test.docx", 
        "test.pdf",
        "test.txt"
    ]
    
    for file_path in test_files:
        is_supported = loader.validate_file_format(file_path)
        print(f"{file_path}: {'지원됨' if is_supported else '지원되지 않음'}")
    
    print()

def test_template_loader():
    """TemplateLoader 테스트"""
    print("=== TemplateLoader 테스트 ===")
    
    loader = TemplateLoader()
    
    # 사용 가능한 템플릿 확인
    available_types = loader.get_available_types()
    print(f"사용 가능한 문서 유형: {available_types}")
    
    # 특정 템플릿 로드 테스트
    for doc_type in ["위험관리계획서", "제품설명서"]:
        template = loader.get_template(doc_type)
        if template:
            print(f"{doc_type} 템플릿 로드 성공")
            print(f"  - 평가 기준 수: {len(template.get('evaluation_criteria', {}))}")
            print(f"  - 필수 섹션 수: {len(template.get('required_sections', []))}")
            # 간단 평가 기준 JSON 테스트
            simple_criteria = loader.get_simple_criteria_json(doc_type)
            print(f"  - 간단 평가 기준 JSON: {simple_criteria}")
        else:
            print(f"{doc_type} 템플릿 로드 실패")
    
    print()

def test_evaluator():
    """AttentionEvaluator 테스트"""
    print("=== AttentionEvaluator 테스트 ===")
    
    evaluator = AttentionEvaluator()
    
    # 샘플 텍스트
    sample_text = """
    위험관리계획서
    
    서론
    본 제품은 새로운 의약품으로, 안전성과 효능을 보장하기 위한 위험관리계획을 수립합니다.
    
    위험요인 식별
    주요 위험요인으로는 부작용, 약물상호작용, 과민반응 등이 있습니다.
    
    위험도 평가
    각 위험요인에 대해 발생 가능성과 심각성을 평가한 결과, 대부분의 위험은 낮은 수준입니다.
    
    위험관리 방안
    의료진 교육, 환자 모니터링, 부작용 보고 체계를 구축하여 위험을 관리합니다.
    
    모니터링 계획
    임상시험과 시판후 조사를 통해 지속적으로 안전성을 모니터링합니다.
    
    결론
    본 위험관리계획을 통해 제품의 안전성을 확보할 수 있습니다.
    """
    
    # 평가 기준 로드
    template_loader = TemplateLoader()
    criteria = template_loader.get_template("위험관리계획서")
    
    if criteria:
        print("평가 기준 로드 성공")
        
        # 평가 실행
        try:
            result = evaluator.evaluate_document(sample_text, criteria, "위험관리계획서")
            
            print("평가 결과:")
            print(f"  - 총점: {result.get('total_score', 0):.1f}/100")
            print(f"  - 등급: {result.get('grade', 'F')}")
            print(f"  - 누락 섹션: {len(result.get('missing_sections', []))}개")
            
            # 항목별 점수
            evaluation_results = result.get('evaluation_results', {})
            for criterion_name, criterion_result in evaluation_results.items():
                score = criterion_result.get('score', 0)
                weight = criterion_result.get('weight', 0)
                print(f"  - {criterion_name}: {score:.1f}/10 (가중치: {weight:.2f})")
            
            print(f"  - 전체 피드백: {result.get('overall_feedback', '')}")
            
        except Exception as e:
            print(f"평가 중 오류 발생: {e}")
    else:
        print("평가 기준 로드 실패")
    
    print()

def test_integration():
    """통합 테스트"""
    print("=== 통합 테스트 ===")
    
    # 직접 텍스트로 테스트 (파일 로딩 대신)
    sample_text = """
    제품설명서
    
    제품명: 테스트 의약품
    
    성분 및 함량
    주성분: 테스트 성분 100mg
    
    효능효과
    감염증 치료에 사용됩니다.
    
    용법용량
    성인 1일 3회, 1회 1정씩 복용합니다.
    
    주의사항
    알레르기 반응이 있는 경우 복용을 중단하세요.
    
    부작용
    구역질, 두통 등이 발생할 수 있습니다.
    
    상호작용
    다른 약물과 함께 복용 시 의사와 상담하세요.
    
    보관방법
    실온에서 보관하세요.
    """
    
    try:
        # TemplateLoader 테스트
        template_loader = TemplateLoader()
        criteria = template_loader.get_template("제품설명서")
        
        if criteria:
            print("평가 기준 로드 성공")
            
            # AttentionEvaluator 테스트
            evaluator = AttentionEvaluator()
            result = evaluator.evaluate_document(sample_text, criteria, "제품설명서")
            
            print("통합 평가 결과:")
            print(f"  - 총점: {result.get('total_score', 0):.1f}/100")
            print(f"  - 등급: {result.get('grade', 'F')}")
            print(f"  - 누락 섹션: {result.get('missing_sections', [])}")
            
            # 결과를 JSON으로 저장
            with open('test_result.json', 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            print("  - 결과가 test_result.json에 저장되었습니다.")
            
        else:
            print("평가 기준 로드 실패")
            
    except Exception as e:
        print(f"통합 테스트 중 오류: {e}")
        import traceback
        traceback.print_exc()

def main():
    """메인 테스트 함수"""
    print("새로운 평가 시스템 테스트 시작\n")
    
    try:
        test_document_loader()
        test_template_loader()
        test_evaluator()
        test_integration()
        
        print("모든 테스트 완료!")
        
    except Exception as e:
        print(f"테스트 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 