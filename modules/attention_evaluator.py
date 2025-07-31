"""
개선된 문서 평가 모듈 - weight와 sub_criteria 기반 평가
"""

import logging
import json
from typing import Dict, List, Tuple, Optional
from langchain.llms import Ollama
from langchain.prompts import PromptTemplate
import re
import time

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def _is_streamlit():
    try:
        import streamlit as st
        return True
    except ImportError:
        return False

# LLM 호출 재시도 래퍼
    
class AttentionEvaluator:
    """문서 평가 및 중요 문장 추출 클래스"""
    
    def __init__(self, model_name: str = "mistral", temperature: float = 0.3):
        self.model_name = model_name
        self.temperature = temperature
        try:
            self.llm = Ollama(model=model_name, temperature=temperature)
            logger.info(f"Ollama 모델 '{model_name}' 초기화 완료 (temperature={temperature})")
        except Exception as e:
            logger.error(f"Ollama 모델 초기화 실패: {e}")
            self.llm = None
    
    def safe_llm_call(self, prompt, max_retries=2):
        for attempt in range(max_retries):
            try:
                return self.llm(prompt)
            except Exception as e:
                if attempt == max_retries - 1:
                    logger.error(f"LLM 호출 실패(최대 재시도): {e}")
                    return "{}"
                logger.warning(f"LLM 호출 실패, 재시도 중... ({attempt+1}/{max_retries})")

    def _animate_progress(self, progress_bar, current, target, total):
        import streamlit as st
        for percent in range(current, target + 1):
            progress_bar.progress(percent / total)
            time.sleep(0.01)

    def evaluate_document(self, text: str, evaluation_criteria: Dict, 
                         document_type: str, feedback_length: str = "보통") -> Dict:
        """
        문서 평가 수행 - 새로운 평가 기준 구조 적용
        """
        try:
            logger.info(f"문서 평가 시작: {document_type}")
            if not self._validate_criteria_structure(evaluation_criteria):
                error_msg = "[평가 기준 구조 오류] 평가 기준 구조가 올바르지 않습니다. (evaluation_criteria 필드/구조 확인 필요)"
                logger.error(error_msg)
                if _is_streamlit():
                    try:
                        import streamlit as st
                    except ImportError:
                        st = None
                    if st is not None:
                        st.error(error_msg)
                return self._create_error_evaluation(error_detail=error_msg)
            if len(text) > 4000:
                return self._evaluate_document_chunked(text, evaluation_criteria, document_type)
            evaluation_results = {}
            total_score = 0.0
            total_weight = 0.0
            criteria_items = list(evaluation_criteria["evaluation_criteria"].items())
            total_criteria = len(criteria_items)
            if total_criteria > 100:
                logger.warning(f"평가 기준이 {total_criteria}개로 매우 많음. 속도가 느려질 수 있습니다.")
            start_time = time.time()
            last_progress = 0
            progress_bar = None
            st = None
            if _is_streamlit():
                try:
                    import streamlit as _st
                    st = _st
                except ImportError:
                    st = None
                if st is not None:
                    progress_bar = st.progress(0)
            for idx, (criterion_name, criterion_data) in enumerate(criteria_items, 1):
                weight = criterion_data.get("weight", 0.0)
                description = criterion_data.get("description", "")
                sub_criteria = criterion_data.get("sub_criteria", [])
                elapsed = time.time() - start_time
                avg_time = elapsed / idx if idx > 0 else 0
                est_total = avg_time * total_criteria
                est_remain = est_total - elapsed
                if idx == 1:
                    msg = f"[기준 {idx}/{total_criteria}] '{criterion_name}' 평가 중... (예상 남은: 계산 중)"
                else:
                    est_remain = max(1, int(est_remain))
                    msg = f"[기준 {idx}/{total_criteria}] '{criterion_name}' 평가 중... (예상 남은: {est_remain}초)"
                logger.info(msg)
                # 카운트다운 남은 시간 표시
                if st is not None:
                    if idx > 1 and est_remain > 0:
                        countdown_placeholder = st.empty()
                        for sec in range(int(est_remain), 0, -1):
                            countdown_placeholder.info(f"⏳ 남은 시간: {sec}초")
                            time.sleep(1)
                        countdown_placeholder.info(f"⏳ 남은 시간: 0초")
                    if progress_bar is not None:
                        self._animate_progress(progress_bar, last_progress, idx, total_criteria)
                        if idx == 1 or idx == total_criteria or idx % max(1, total_criteria // 10) == 0:
                            st.info(msg)
                last_progress = idx
                try:
                    criterion_result = self._evaluate_criterion(
                        text, criterion_name, description, sub_criteria, document_type, feedback_length
                    )
                except Exception as e:
                    error_msg = f"[항목 평가 오류] '{criterion_name}' 평가 중 예외 발생: {e}"
                    logger.error(error_msg)
                    if st is not None:
                        st.error(error_msg)
                    criterion_result = {"score": 0, "feedback": error_msg, "weight": weight, "weighted_score": 0, "error": str(e)}
                weighted_score = float(criterion_result.get("score", 0)) * weight
                criterion_result["weight"] = weight
                criterion_result["weighted_score"] = weighted_score
                evaluation_results[criterion_name] = criterion_result
                total_score += weighted_score
                total_weight += weight
            final_total_score = (total_score / total_weight * 100) if total_weight > 0 else 0
            missing_sections = self._check_missing_sections(text, evaluation_criteria)
            result = {
                "document_type": document_type,
                "total_score": round(final_total_score, 2),
                "evaluation_results": evaluation_results,
                "missing_sections": missing_sections,
                "grade": self._calculate_grade(final_total_score),
                "overall_feedback": self._generate_overall_feedback(evaluation_results, missing_sections),
                "recommendations": self._generate_recommendations(evaluation_results, missing_sections)
            }
            logger.info(f"평가 완료 - 총점: {final_total_score}")
            return result
        except Exception as e:
            error_msg = f"[문서 평가 전체 오류] 문서 평가 중 오류 발생: {e}"
            logger.error(error_msg)
            if _is_streamlit():
                try:
                    import streamlit as st
                except ImportError:
                    st = None
                if st is not None:
                    st.error(error_msg)
            return self._create_error_evaluation(error_detail=error_msg)
    
    def _validate_criteria_structure(self, criteria_data: Dict) -> bool:
        """평가 기준 구조 검증 (새로운 구조 지원)"""
        try:
            if "evaluation_criteria" not in criteria_data:
                logger.error("evaluation_criteria 필드가 없습니다.")
                if _is_streamlit():
                    try:
                        import streamlit as st
                    except ImportError:
                        st = None
                    if st is not None:
                        st.error("[평가 기준 구조 오류] evaluation_criteria 필드가 없습니다.")
                return False
            
            criteria = criteria_data["evaluation_criteria"]
            
            # 새로운 구조: 딕셔너리 형태의 간단한 구조
            if isinstance(criteria, dict):
                for key, value in criteria.items():
                    if not isinstance(value, dict) or "description" not in value:
                        logger.error(f"평가 기준 '{key}'에 description이 없습니다.")
                        if _is_streamlit():
                            try:
                                import streamlit as st
                            except ImportError:
                                st = None
                            if st is not None:
                                st.error(f"[평가 기준 구조 오류] '{key}'에 description이 없습니다.")
                        return False
                return True
            
            # 기존 구조: weight, sub_criteria 포함된 구조
            elif isinstance(criteria, dict):
                for key, value in criteria.items():
                    if not isinstance(value, dict):
                        logger.error(f"평가 기준 '{key}'가 딕셔너리가 아닙니다.")
                        return False
                    if "weight" not in value:
                        logger.error(f"평가 기준 '{key}'에 weight가 없습니다.")
                        return False
                    if "description" not in value:
                        logger.error(f"평가 기준 '{key}'에 description이 없습니다.")
                        return False
                    if "sub_criteria" not in value:
                        logger.error(f"평가 기준 '{key}'에 sub_criteria가 없습니다.")
                    return False
                return True
            
            else:
                logger.error("evaluation_criteria가 딕셔너리가 아닙니다.")
                if _is_streamlit():
                    try:
                        import streamlit as st
                    except ImportError:
                        st = None
                    if st is not None:
                        st.error("[평가 기준 구조 오류] evaluation_criteria가 딕셔너리가 아닙니다.")
                return False
        
        except Exception as e:
            logger.error(f"평가 기준 구조 검증 실패: {e}")
            if _is_streamlit():
                try:
                    import streamlit as st
                except ImportError:
                    st = None
                if st is not None:
                    st.error(f"[평가 기준 구조 오류] 평가 기준 구조 검증 실패: {e}")
            return False
    
    def _evaluate_criterion(self, text: str, criterion_name: str, description: str, 
                          sub_criteria: Optional[List[str]] = None, document_type: str = "기타", feedback_length: str = "보통") -> Dict:
        """개별 평가 항목 평가"""
        if sub_criteria is None:
            sub_criteria = []
            
        if not self.llm:
            return self._evaluate_criterion_fallback(text, criterion_name, description, sub_criteria)
        
        try:
            # 피드백 길이 안내
            if feedback_length == "간단":
                length_instruction = "각 항목별로 1~2문장씩 간단하게 작성하세요."
            elif feedback_length == "상세":
                length_instruction = "각 항목별로 3~5문장 이상, 구체적 예시와 함께 작성하세요."
            else:
                length_instruction = "각 항목별로 2~3문장씩 작성하세요."
            # 평가 프롬프트
            prompt_template = PromptTemplate(
                input_variables=["text", "criterion_name", "description", "sub_criteria", "document_type", "length_instruction"],
                template="""
다음 의약품 문서를 평가해주세요.

문서 유형: {document_type}
평가 항목: {criterion_name}
평가 설명: {description}

세부 평가 기준:
{sub_criteria}

문서 내용:
{text}

아래 JSON 형식으로 답변하세요:
{{
    "score": 1-10점,
    "good": "잘한 점 (2~3문장)",
    "bad": "아쉬운 점 (2~3문장)",
    "suggestion": "구체적 개선 제안 (2~3문장)",
    "feedback": "전체 요약 피드백 (선택)",
    "sub_criteria_evaluation": [
        {{
            "criterion": "세부 기준명",
            "score": 1-10점,
            "feedback": "해당 기준에 대한 피드백"
        }}
    ]
}}
{length_instruction}
"""
            )
            
            # 세부 기준을 문자열로 변환
            sub_criteria_text = "\n".join([f"- {criterion}" for criterion in sub_criteria]) if sub_criteria else "세부 기준 없음"
            
            prompt = prompt_template.format(
                text=text[:3000],  # 텍스트 길이 제한
                criterion_name=criterion_name,
                description=description,
                sub_criteria=sub_criteria_text,
                document_type=document_type,
                length_instruction=length_instruction
            )
            
            response = self.safe_llm_call(prompt)
            return self._parse_criterion_response(response, sub_criteria)
            
        except Exception as e:
            logger.error(f"평가 항목 '{criterion_name}' 평가 중 오류: {e}")
            return self._evaluate_criterion_fallback(text, criterion_name, description, sub_criteria)
    
    def _parse_criterion_response(self, response: str, sub_criteria: List[str]) -> Dict:
        """평가 응답 파싱"""
        try:
            # JSON 추출 시도
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                
                # 기본 구조 검증
                if "score" not in result:
                    result["score"] = 5
                if "good" not in result:
                    result["good"] = ""
                if "bad" not in result:
                    result["bad"] = ""
                if "suggestion" not in result:
                    result["suggestion"] = ""
                if "feedback" not in result:
                    result["feedback"] = ""
                if "sub_criteria_evaluation" not in result:
                    result["sub_criteria_evaluation"] = []
                
                return result
            else:
                # JSON 파싱 실패 시 기본값 반환
                error_msg = "[LLM 응답 파싱 오류] LLM 응답에서 JSON을 추출하지 못했습니다."
                logger.error(error_msg)
                if _is_streamlit():
                    try:
                        import streamlit as st
                    except ImportError:
                        st = None
                    if st is not None:
                        st.error(error_msg)
                return {
                    "score": 0,
                    "good": "",
                    "bad": "",
                    "suggestion": "",
                    "feedback": error_msg,
                    "sub_criteria_evaluation": [],
                    "error": error_msg
                }
                
        except Exception as e:
            error_msg = f"[LLM 응답 파싱 예외] 응답 파싱 중 오류: {e}"
            logger.error(error_msg)
            if _is_streamlit():
                try:
                    import streamlit as st
                except ImportError:
                    st = None
                if st is not None:
                    st.error(error_msg)
            return {
                "score": 0,
                "good": "",
                "bad": "",
                "suggestion": "",
                "feedback": error_msg,
                "sub_criteria_evaluation": [],
                "error": error_msg
            }
    
    def _evaluate_criterion_fallback(self, text: str, criterion_name: str, 
                                   description: str, sub_criteria: List[str]) -> Dict:
        """평가 항목 fallback 평가"""
        # 간단한 키워드 기반 점수 계산
        score = self._calculate_basic_score(text, criterion_name, sub_criteria)
        
        # 기본 피드백 생성
        feedback = f"{criterion_name} 항목에 대한 기본 평가가 완료되었습니다."
        
        # 세부 기준 평가
        sub_criteria_evaluation = []
        for criterion in sub_criteria:
            sub_score = self._calculate_basic_score(text, criterion, [criterion])
            sub_criteria_evaluation.append({
                "criterion": criterion,
                "score": sub_score,
                "feedback": f"{criterion}에 대한 기본 평가"
            })
        
        return {
            "score": score,
            "feedback": feedback,
            "sub_criteria_evaluation": sub_criteria_evaluation
        }
    
    def _calculate_basic_score(self, text: str, criterion_name: str, sub_criteria: List[str]) -> int:
        """기본 점수 계산"""
        score = 5  # 기본 점수
        
        # 키워드 매칭으로 점수 조정
        text_lower = text.lower()
        criterion_lower = criterion_name.lower()
        
        # 평가 항목과 관련된 키워드들
        positive_keywords = ["정확", "완전", "적절", "충분", "명확", "구체", "상세"]
        negative_keywords = ["부족", "누락", "불명확", "모호", "부정확", "미흡"]
        
        # 긍정적 키워드 확인
        for keyword in positive_keywords:
            if keyword in text_lower:
                score += 1
        
        # 부정적 키워드 확인
        for keyword in negative_keywords:
            if keyword in text_lower:
                score -= 1
        
        # 세부 기준 관련 키워드 확인
        for sub_criterion in sub_criteria:
            sub_lower = sub_criterion.lower()
            if any(keyword in text_lower for keyword in sub_lower.split()):
                score += 1
        
        # 점수 범위 제한
        return max(1, min(10, score))
    
    def _check_missing_sections(self, text: str, evaluation_criteria: Dict) -> List[str]:
        """필수 섹션 누락 체크"""
        missing_sections = []
        
        if "required_sections" not in evaluation_criteria:
            return missing_sections
        
        required_sections = evaluation_criteria["required_sections"]
        text_lower = text.lower()
        
        for section in required_sections:
            section_lower = section.lower()
            # 섹션명이나 관련 키워드가 텍스트에 없는 경우
            if section_lower not in text_lower:
                # 관련 키워드도 확인
                related_keywords = self._get_section_keywords(section)
                if not any(keyword in text_lower for keyword in related_keywords):
                    missing_sections.append(section)
        
        return missing_sections
    
    def _get_section_keywords(self, section_name: str) -> List[str]:
        """섹션명에 따른 관련 키워드 반환"""
        keyword_map = {
            "서론": ["서론", "개요", "목적", "배경"],
            "위험요인 식별": ["위험", "요인", "식별", "위험요인"],
            "위험도 평가": ["위험도", "평가", "분석", "위험분석"],
            "위험관리 방안": ["관리", "방안", "대책", "위험관리"],
            "모니터링 계획": ["모니터링", "계획", "감시", "추적"],
            "결론": ["결론", "요약", "종합", "결과"],
            "제품명": ["제품명", "상품명", "약품명"],
            "성분 및 함량": ["성분", "함량", "구성", "성분함량"],
            "효능효과": ["효능", "효과", "적응증", "치료효과"],
            "용법용량": ["용법", "용량", "투여", "복용"],
            "주의사항": ["주의", "경고", "주의사항", "안전"],
            "부작용": ["부작용", "부정반응", "사이드이펙트"],
            "상호작용": ["상호작용", "약물상호작용", "병용"],
            "보관방법": ["보관", "저장", "보관방법", "보관조건"]
        }
        
        return keyword_map.get(section_name, [section_name])
    
    def _calculate_grade(self, score: float) -> str:
        """점수에 따른 등급 계산"""
        if score >= 90:
            return "A+"
        elif score >= 80:
            return "A"
        elif score >= 70:
            return "B+"
        elif score >= 60:
            return "B"
        elif score >= 50:
            return "C+"
        elif score >= 40:
            return "C"
        else:
            return "D"
    
    def _generate_overall_feedback(self, evaluation_results: Dict, missing_sections: List[str]) -> str:
        """전체 피드백 생성"""
        feedback_parts = []
        
        # 각 평가 항목별 피드백
        for criterion_name, result in evaluation_results.items():
            score = result["score"]
            if score >= 8:
                feedback_parts.append(f"{criterion_name}: 우수한 수준입니다.")
            elif score >= 6:
                feedback_parts.append(f"{criterion_name}: 양호한 수준입니다.")
            elif score >= 4:
                feedback_parts.append(f"{criterion_name}: 개선이 필요합니다.")
            else:
                feedback_parts.append(f"{criterion_name}: 대폭 개선이 필요합니다.")
        
        # 누락 섹션 피드백
        if missing_sections:
            feedback_parts.append(f"누락된 필수 섹션: {', '.join(missing_sections)}")
        
        return " ".join(feedback_parts)
    
    def _generate_recommendations(self, evaluation_results: Dict, missing_sections: List[str]) -> List[str]:
        """개선 권장사항 생성"""
        recommendations = []
        
        # 낮은 점수 항목에 대한 권장사항
        for criterion_name, result in evaluation_results.items():
            score = result["score"]
            if score < 6:
                recommendations.append(f"{criterion_name} 항목을 개선하여 점수를 향상시켜주세요.")
        
        # 누락 섹션에 대한 권장사항
        for section in missing_sections:
            recommendations.append(f"{section} 섹션을 추가해주세요.")
        
        # 세부 기준별 권장사항
        for criterion_name, result in evaluation_results.items():
            for sub_eval in result.get("sub_criteria_evaluation", []):
                if sub_eval["score"] < 6:
                    recommendations.append(f"{sub_eval['criterion']}에 대한 개선이 필요합니다.")
        
        return recommendations
    
    def _evaluate_document_chunked(self, text: str, evaluation_criteria: Dict, 
                                 document_type: str) -> Dict:
        """긴 문서를 청크로 나누어 평가 (진행률/예상 소요 시간/상태 로그 추가)"""
        try:
            chunk_size = 3000
            chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
            chunk_evaluations = []
            total_chunks = len(chunks)
            if total_chunks > 100:
                logger.warning(f"청크 수가 {total_chunks}개로 매우 많음. 속도가 느려질 수 있습니다.")
            start_time = time.time()
            last_progress = 0
            progress_bar = None
            st = None
            if _is_streamlit():
                try:
                    import streamlit as _st
                    st = _st
                except ImportError:
                    st = None
                if st is not None:
                    progress_bar = st.progress(0)
            for i, chunk in enumerate(chunks, 1):
                elapsed = time.time() - start_time
                avg_time = elapsed / i if i > 0 else 0
                est_total = avg_time * total_chunks
                est_remain = est_total - elapsed
                if i == 1:
                    msg = f"[청크 {i}/{total_chunks}] 평가 중... (예상 남은: 계산 중)"
                else:
                    est_remain = max(1, int(est_remain))
                    msg = f"[청크 {i}/{total_chunks}] 평가 중... (예상 남은: {est_remain}초)"
                logger.info(msg)
                # 카운트다운 남은 시간 표시
                if st is not None:
                    if i > 1 and est_remain > 0:
                        countdown_placeholder = st.empty()
                        for sec in range(int(est_remain), 0, -1):
                            countdown_placeholder.info(f"⏳ 남은 시간: {sec}초")
                            time.sleep(1)
                        countdown_placeholder.info(f"⏳ 남은 시간: 0초")
                    if progress_bar is not None:
                        self._animate_progress(progress_bar, last_progress, i, total_chunks)
                        if i == 1 or i == total_chunks or i % max(1, total_chunks // 10) == 0:
                            st.info(msg)
                last_progress = i
                chunk_result = self.evaluate_document(chunk, evaluation_criteria, document_type)
                chunk_evaluations.append(chunk_result)
            return self._combine_chunk_evaluations(chunk_evaluations, evaluation_criteria)
        except Exception as e:
            logger.error(f"청크 평가 중 오류: {e}")
            return self._create_error_evaluation()
    
    def _combine_chunk_evaluations(self, evaluations: List[Dict], 
                                 evaluation_criteria: Dict) -> Dict:
        """청크 평가 결과 통합"""
        if not evaluations:
            return self._create_error_evaluation()
        
        # 첫 번째 평가 결과를 기본으로 사용
        combined_result = evaluations[0].copy()
        
        # 점수 평균 계산
        total_scores = []
        for eval_result in evaluations:
            if "total_score" in eval_result:
                total_scores.append(eval_result["total_score"])
        
        if total_scores:
            combined_result["total_score"] = sum(total_scores) / len(total_scores)
            combined_result["grade"] = self._calculate_grade(combined_result["total_score"])
        
        # 평가 결과 통합
        if len(evaluations) > 1:
            combined_evaluation_results = {}
            
            # 모든 평가 항목에 대해 평균 계산
            criterion_names = evaluations[0]["evaluation_results"].keys()
            
            for criterion_name in criterion_names:
                scores = []
                feedbacks = []
                
                for eval_result in evaluations:
                    if criterion_name in eval_result["evaluation_results"]:
                        criterion_result = eval_result["evaluation_results"][criterion_name]
                        scores.append(criterion_result["score"])
                        feedbacks.append(criterion_result["feedback"])
                
                if scores:
                    avg_score = sum(scores) / len(scores)
                    combined_feedback = " ".join(feedbacks)
                    
                    combined_evaluation_results[criterion_name] = {
                        "score": round(avg_score, 2),
                        "feedback": combined_feedback,
                        "weight": evaluations[0]["evaluation_results"][criterion_name].get("weight", 0),
                        "weighted_score": round(avg_score * evaluations[0]["evaluation_results"][criterion_name].get("weight", 0), 2)
                    }
            
            combined_result["evaluation_results"] = combined_evaluation_results
        
        return combined_result
    
    def _create_error_evaluation(self, error_detail: str = None) -> Dict:
        """오류 시 기본 평가 결과 생성"""
        return {
            "document_type": "알 수 없음",
            "total_score": 0,
            "evaluation_results": {},
            "missing_sections": [],
            "grade": "F",
            "overall_feedback": "평가 중 오류가 발생했습니다.",
            "recommendations": ["시스템을 다시 시작해주시거나, 관리자에게 문의하세요."],
            "error_detail": error_detail or "평가 중 알 수 없는 오류 발생"
        }
    
    def extract_key_sentences(self, text: str, num_sentences: int = 3) -> List[str]:
        """중요 문장 추출 (기존 메서드 유지)"""
        if not self.llm:
            return self._extract_key_sentences_fallback(text, num_sentences)
        
        try:
            prompt_template = PromptTemplate(
                input_variables=["text", "num_sentences"],
                template="""
다음 텍스트에서 가장 중요한 {num_sentences}개의 문장을 추출해주세요.

텍스트:
{text}

중요한 문장들을 번호를 매겨서 나열해주세요:
1. [첫 번째 중요 문장]
2. [두 번째 중요 문장]
3. [세 번째 중요 문장]

각 문장은 원문에서 그대로 추출해주세요.
"""
            )
            
            prompt = prompt_template.format(
                text=text[:3000],
                num_sentences=num_sentences
            )
            
            response = self.llm(prompt)
            return self._parse_sentences_from_response(response)
            
        except Exception as e:
            logger.error(f"중요 문장 추출 중 오류: {e}")
            return self._extract_key_sentences_fallback(text, num_sentences)
    
    def _extract_key_sentences_fallback(self, text: str, num_sentences: int = 3) -> List[str]:
        """기본 방법으로 중요 문장 추출"""
        sentences = self._split_sentences(text)
        
        if len(sentences) <= num_sentences:
            return sentences
        
        # 간단한 키워드 기반 점수 계산
        important_keywords = [
            "중요", "핵심", "주요", "필수", "반드시", "꼭",
            "위험", "안전", "효능", "효과", "부작용", "주의",
            "결론", "요약", "결과", "발견", "관찰", "확인"
        ]
        
        sentence_scores = []
        for sentence in sentences:
            score = 0
            sentence_lower = sentence.lower()
            
            for keyword in important_keywords:
                if keyword in sentence_lower:
                    score += 1
            
            if 10 <= len(sentence) <= 200:
                score += 1
            
            sentence_scores.append((sentence, score))
        
        sentence_scores.sort(key=lambda x: x[1], reverse=True)
        return [sentence for sentence, score in sentence_scores[:num_sentences]]
    
    def _split_sentences(self, text: str) -> List[str]:
        """텍스트를 문장 단위로 분리"""
        sentence_pattern = r'[.!?。！？]\s*'
        sentences = re.split(sentence_pattern, text)
        
        cleaned_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence and len(sentence) > 5:
                cleaned_sentences.append(sentence)
        
        return cleaned_sentences
    
    def _parse_sentences_from_response(self, response: str) -> List[str]:
        """응답에서 문장 추출"""
        sentences = []
        lines = response.strip().split('\n')
        
        for line in lines:
            line = line.strip()
            if line and re.match(r'^\d+\.', line):
                sentence = re.sub(r'^\d+\.\s*', '', line)
                if sentence:
                    sentences.append(sentence)
        
        return sentences 