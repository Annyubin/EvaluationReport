"""
평가 결과를 마크다운 or JSON 형식으로 정리
"""

import json
import logging
from typing import Dict, List, Optional
from datetime import datetime
import os

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FeedbackFormatter:
    """평가 결과를 다양한 형식으로 포맷팅하는 클래스"""
    
    def __init__(self):
        self.timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    def format_to_markdown(self, evaluation_result: Dict, document_type: str, 
                          document_name: str = "업로드된 문서") -> str:
        """평가 결과를 마크다운 형식으로 변환"""
        try:
            md_content = []
            md_content.append(f"# 의약품 문서 평가 보고서\n")
            md_content.append("## 📋 기본 정보")
            md_content.append(f"- **문서명**: {document_name}")
            md_content.append(f"- **문서 유형**: {document_type}")
            md_content.append(f"- **평가 일시**: {self.timestamp}\n")
            total_score = evaluation_result.get("total_score", 0)
            md_content.append("## 🎯 평가 결과 요약")
            md_content.append(f"**총점: {total_score}/100**\n")
            grade = self._get_grade(total_score)
            md_content.append(f"**등급: {grade}**\n")
            md_content.append("## 📊 항목별 평가 결과\n")
            evaluation_results = evaluation_result.get("evaluation_results", {})
            for criterion, detail in evaluation_results.items():
                score = detail.get("score", 0)
                good = detail.get("good", "")
                bad = detail.get("bad", "")
                suggestion = detail.get("suggestion", "")
                feedback = detail.get("feedback", "")
                md_content.append(f"### {criterion}")
                md_content.append(f"- **점수**: {score}/10\n")
                if good:
                    md_content.append(f"- **잘한 점**: {good}")
                if bad:
                    md_content.append(f"- **아쉬운 점**: {bad}")
                if suggestion:
                    md_content.append(f"- **개선 제안**: {suggestion}")
                if feedback:
                    md_content.append(f"- **전체 요약**: {feedback}")
                md_content.append("")
            if "중요문장" in evaluation_result:
                md_content.append("## 🔍 중요 문장\n")
                key_sentences = evaluation_result["중요문장"]
                for i, sentence in enumerate(key_sentences, 1):
                    md_content.append(f"{i}. {sentence}")
                md_content.append("")
            md_content.append("## 💡 종합 개선 권장사항\n")
            recommendations = self._generate_recommendations(evaluation_result)
            for i, recommendation in enumerate(recommendations, 1):
                md_content.append(f"{i}. {recommendation}")
            md_content.append("")
            md_content.append("---")
            md_content.append(f"*이 보고서는 {self.timestamp}에 자동 생성되었습니다.*")
            return "\n".join(md_content)
        except Exception as e:
            logger.error(f"마크다운 포맷팅 중 오류 발생: {e}")
            return self._create_error_markdown(str(e))

    def format_to_json(self, evaluation_result: Dict, document_type: str, 
                      document_name: str = "업로드된 문서") -> str:
        """평가 결과를 JSON 형식으로 변환"""
        try:
            json_data = {
                "metadata": {
                    "document_name": document_name,
                    "document_type": document_type,
                    "evaluation_timestamp": self.timestamp,
                    "version": "1.0"
                },
                "evaluation_result": evaluation_result,
                "summary": {
                    "total_score": evaluation_result.get("total_score", 0),
                    "grade": self._get_grade(evaluation_result.get("total_score", 0)),
                    "recommendations": self._generate_recommendations(evaluation_result)
                }
            }
            return json.dumps(json_data, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"JSON 포맷팅 중 오류 발생: {e}")
            return json.dumps({"error": str(e)}, ensure_ascii=False, indent=2)
    
    def save_to_file(self, content: str, filename: str, output_dir: str = "output") -> str:
        """평가 결과를 파일로 저장"""
        try:
            # 출력 디렉토리 생성
            os.makedirs(output_dir, exist_ok=True)
            
            # 파일 경로 생성
            file_path = os.path.join(output_dir, filename)
            
            # 파일 저장
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            logger.info(f"평가 결과가 저장되었습니다: {file_path}")
            return file_path
            
        except Exception as e:
            logger.error(f"파일 저장 중 오류 발생: {e}")
            raise
    
    def save_markdown_report(self, evaluation_result: Dict, document_type: str, 
                           document_name: str = "업로드된 문서") -> str:
        """마크다운 보고서를 파일로 저장"""
        try:
            # 마크다운 내용 생성
            md_content = self.format_to_markdown(evaluation_result, document_type, document_name)
            
            # 파일명 생성
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_doc_name = "".join(c for c in document_name if c.isalnum() or c in (' ', '-', '_')).rstrip()
            filename = f"평가보고서_{safe_doc_name}_{timestamp}.md"
            
            # 파일 저장
            return self.save_to_file(md_content, filename)
            
        except Exception as e:
            logger.error(f"마크다운 보고서 저장 중 오류 발생: {e}")
            raise
    
    def save_json_report(self, evaluation_result: Dict, document_type: str, 
                        document_name: str = "업로드된 문서") -> str:
        """JSON 보고서를 파일로 저장"""
        try:
            # JSON 내용 생성
            json_content = self.format_to_json(evaluation_result, document_type, document_name)
            
            # 파일명 생성
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_doc_name = "".join(c for c in document_name if c.isalnum() or c in (' ', '-', '_')).rstrip()
            filename = f"평가결과_{safe_doc_name}_{timestamp}.json"
            
            # 파일 저장
            return self.save_to_file(json_content, filename)
            
        except Exception as e:
            logger.error(f"JSON 보고서 저장 중 오류 발생: {e}")
            raise
    
    def _get_grade(self, score: float) -> str:
        """점수에 따른 등급 반환"""
        if score >= 9.0:
            return "A+ (우수)"
        elif score >= 8.0:
            return "A (양호)"
        elif score >= 7.0:
            return "B+ (보통)"
        elif score >= 6.0:
            return "B (미흡)"
        elif score >= 5.0:
            return "C (부족)"
        else:
            return "D (매우 부족)"
    
    def _generate_recommendations(self, evaluation_result: Dict) -> List[str]:
        """종합 개선 권장사항 생성"""
        recommendations = []
        
        # 각 항목별 권장사항
        criteria_names = ["정확성", "표현력", "항목누락", "형식적합성"]
        for criterion in criteria_names:
            if criterion in evaluation_result:
                criterion_eval = evaluation_result[criterion]
                if isinstance(criterion_eval, dict):
                    score = criterion_eval.get("점수", 0)
                    
                    if score < 6:
                        if criterion == "정확성":
                            recommendations.append("정보의 정확성을 높이기 위해 전문가 검토를 받으세요.")
                        elif criterion == "표현력":
                            recommendations.append("문장을 더 명확하고 이해하기 쉽게 작성하세요.")
                        elif criterion == "항목누락":
                            recommendations.append("필수 항목이 누락되지 않도록 체크리스트를 활용하세요.")
                        elif criterion == "형식적합성":
                            recommendations.append("규제 양식에 맞게 문서를 작성하세요.")
        
        # 총점 기반 권장사항
        total_score = evaluation_result.get("총점", 0)
        if total_score < 6:
            recommendations.append("전반적인 문서 품질 향상이 필요합니다. 전문가의 도움을 받아보세요.")
        elif total_score < 8:
            recommendations.append("문서 품질이 양호하지만, 일부 개선이 필요합니다.")
        else:
            recommendations.append("문서 품질이 우수합니다. 현재 수준을 유지하세요.")
        
        return recommendations
    
    def _create_error_markdown(self, error_message: str) -> str:
        """오류 발생 시 기본 마크다운 생성"""
        return f"""# 평가 보고서 생성 오류

평가 보고서 생성 중 오류가 발생했습니다.

**오류 내용**: {error_message}

**해결 방법**:
1. 문서를 다시 업로드해보세요.
2. 다른 문서 유형을 선택해보세요.
3. 시스템 관리자에게 문의하세요.

---
*이 메시지는 {self.timestamp}에 생성되었습니다.*
"""
    
    def format_console_output(self, evaluation_result: Dict, document_type: str) -> str:
        """콘솔 출력용 간단한 포맷"""
        try:
            lines = []
            lines.append(f"[문서 유형] {document_type}")
            lines.append(f"[총점] {evaluation_result.get('total_score', 0)}/100")
            lines.append(f"[등급] {self._get_grade(evaluation_result.get('total_score', 0))}")
            lines.append("")
            evaluation_results = evaluation_result.get("evaluation_results", {})
            for criterion, detail in evaluation_results.items():
                score = detail.get("score", 0)
                good = detail.get("good", "")
                bad = detail.get("bad", "")
                suggestion = detail.get("suggestion", "")
                feedback = detail.get("feedback", "")
                lines.append(f"- {criterion}: {score}/10")
                if good:
                    lines.append(f"  잘한 점: {good}")
                if bad:
                    lines.append(f"  아쉬운 점: {bad}")
                if suggestion:
                    lines.append(f"  개선 제안: {suggestion}")
                if feedback:
                    lines.append(f"  전체 요약: {feedback}")
                lines.append("")
            return "\n".join(lines)
        except Exception as e:
            logger.error(f"콘솔 포맷팅 중 오류 발생: {e}")
            return f"[오류] {e}" 