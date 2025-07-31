"""
의약품 문서 평가 시스템 - Streamlit 웹 애플리케이션
"""

import streamlit as st
import os
import tempfile
import logging
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import json # Added for JSON loading

# 모듈 import
from modules.document_loader import DocumentLoader
from modules.document_classifier import DocumentClassifier
from modules.template_loader import TemplateLoader
from modules.attention_evaluator import AttentionEvaluator
from modules.feedback_formatter import FeedbackFormatter
from modules.evaluation_selector import EvaluationSelector

# Ollama 등 주요 모듈 캐싱 함수 추가
@st.cache_resource
def get_document_loader():
    from modules.document_loader import DocumentLoader
    return DocumentLoader()

@st.cache_resource
def get_document_classifier():
    from modules.document_classifier import DocumentClassifier
    return DocumentClassifier()

@st.cache_resource
def get_template_loader():
    from modules.template_loader import TemplateLoader
    return TemplateLoader()

@st.cache_resource
def get_attention_evaluator():
    from modules.attention_evaluator import AttentionEvaluator
    return AttentionEvaluator()

@st.cache_resource
def get_feedback_formatter():
    from modules.feedback_formatter import FeedbackFormatter
    return FeedbackFormatter()

@st.cache_resource
def get_evaluation_selector():
    from modules.evaluation_selector import EvaluationSelector
    return EvaluationSelector()

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 페이지 설정
st.set_page_config(
    page_title="의약품 문서 평가 시스템",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 레이아웃 최적화를 위한 CSS
st.markdown("""
<style>
    /* 메인 컨테이너 최적화 */
    .main .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
        max-width: 100%;
        padding-left: 1rem;
        padding-right: 1rem;
    }
    
    .stApp {
        max-width: 100%;
    }
    
    /* 사이드바 스타일 */
    [data-testid="stSidebar"] {
        background-color: #f8f9fa;
    }
    
    .sidebar .sidebar-content {
        padding: 1rem;
    }
    
    /* 설정 패널 스타일 */
    .settings-panel {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #dee2e6;
        margin-bottom: 1rem;
    }
    
    .settings-header {
        font-size: 1.2rem;
        font-weight: bold;
        color: #495057;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    /* 반응형 디자인 */
    @media (max-width: 768px) {
        .main .block-container {
            padding-left: 0.5rem;
            padding-right: 0.5rem;
        }
    }
</style>
""", unsafe_allow_html=True)

# CSS 스타일
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #2c3e50;
        margin-bottom: 1rem;
    }
    .score-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 0.5rem 0;
    }
    .success-message {
        background-color: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 5px;
        border: 1px solid #c3e6cb;
    }
    .error-message {
        background-color: #f8d7da;
        color: #721c24;
        padding: 1rem;
        border-radius: 5px;
        border: 1px solid #f5c6cb;
    }
    .info-box {
        background-color: #e7f3ff;
        padding: 1rem;
        border-radius: 5px;
        border: 1px solid #b3d9ff;
    }
</style>
""", unsafe_allow_html=True)

def dedup_and_join(feedbacks):
    seen = set()
    result = []
    for fb in feedbacks:
        fb = fb.strip()
        if fb and fb not in seen:
            seen.add(fb)
            result.append(fb)
    return " ".join(result)

# 긴 텍스트를 나눠서 출력하는 함수 추가

def print_long_text(text, chunk_size=1000):
    for i in range(0, len(text), chunk_size):
        st.markdown(text[i:i+chunk_size])

class StreamlitApp:
    """Streamlit 애플리케이션 클래스"""
    
    def __init__(self):
        self.initialize_session_state()
        # 캐싱된 인스턴스 사용
        self.document_loader = get_document_loader()
        self.document_classifier = get_document_classifier()
        self.template_loader = get_template_loader()
        self.evaluator = get_attention_evaluator()
        self.formatter = get_feedback_formatter()
        self.evaluation_selector = get_evaluation_selector()
    
    def initialize_session_state(self):
        """세션 상태 초기화"""
        if 'evaluation_result' not in st.session_state:
            st.session_state.evaluation_result = None
        if 'document_type' not in st.session_state:
            st.session_state.document_type = None
        if 'uploaded_file' not in st.session_state:
            st.session_state.uploaded_file = None
    
    def load_system(self):
        """평가 시스템 로드"""
        try:
            self.document_loader = DocumentLoader()
            self.document_classifier = DocumentClassifier()
            self.template_loader = TemplateLoader()
            self.evaluator = AttentionEvaluator()
            self.formatter = FeedbackFormatter()
            self.evaluation_selector = EvaluationSelector()
            logger.info("평가 시스템 로드 완료")
        except Exception as e:
            logger.error(f"시스템 로드 실패: {e}")
            st.error(f"시스템 초기화 중 오류가 발생했습니다: {e}")
    
    def render_header(self):
        """헤더 렌더링"""
        st.markdown('<h1 class="main-header">💊 의약품 문서 평가 시스템</h1>', unsafe_allow_html=True)
        st.markdown("""
        <div style="text-align: center; margin-bottom: 2rem;">
            <p style="font-size: 1.2rem; color: #666;">
                HWP, DOCX 문서를 업로드하여 의약품 인허가 관련 문서의 품질을 자동으로 평가받으세요
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    def render_sidebar(self):
        """사이드바 렌더링"""
        with st.sidebar:
            st.markdown("## ⚙️ 설정")
            
            # 모델 선택
            model_name = st.selectbox(
                "🤖 AI 모델",
                ["mistral", "llama2", "codellama"],
                help="사용할 Ollama 모델을 선택하세요"
            )
            
            # 문서 유형 선택
            st.markdown("**📄 문서 유형**")
            auto_classify = st.checkbox("자동 분류", value=True, help="문서 유형을 자동으로 감지합니다")
            
            if not auto_classify:
                available_types = self.template_loader.get_available_types()
                selected_type = st.selectbox(
                    "문서 유형 선택",
                    available_types,
                    help="평가할 문서의 유형을 선택하세요"
                )
                st.session_state.document_type = selected_type
            else:
                st.session_state.document_type = None
            
            # 제품 유형 선택 (fallback)
            st.markdown("**🏷️ 제품 유형**")
            auto_product = st.checkbox("자동 제품 추론", value=True, help="문서 내용에서 제품 유형을 자동으로 추론합니다")
            
            if not auto_product:
                available_products = self.evaluation_selector.get_available_products()
                selected_product = st.selectbox(
                    "제품 유형 선택",
                    available_products,
                    help="평가할 제품의 유형을 선택하세요"
                )
                st.session_state.selected_product = selected_product
            else:
                st.session_state.selected_product = None
            
            # 평가 옵션 안내문 추가
            st.info("""
📦 **청크(chunk)란?**\n\n문서를 인공지능이 이해하기 쉬운 작은 단위(조각)로 나누는 방법입니다. 너무 작으면 문맥이 끊기고, 너무 크면 일부 정보가 누락될 수 있습니다.\n\n**청크 크기 설정 가이드**\n- 권장 기본값: 500~1000자\n- 문서가 짧거나 단순: 1000자 이상\n- 문서가 길거나 복잡: 500~800자\n- 표, 양식, 리스트가 많음: 300~500자\n\n특별한 경우가 아니라면 기본값(500~1000자)을 그대로 사용해도 무방합니다. 평가 결과가 너무 단편적이거나 문맥이 어색하게 끊긴다면 청크 크기를 늘려보세요. 반대로, 너무 많은 내용이 한 번에 묶여서 평가가 부정확하다면 청크 크기를 줄여보세요.\n\n자세한 설명은 [도움말]을 참고하세요.
""")
            # 평가 옵션
            st.markdown("**📊 평가 옵션**")
            chunk_size = st.slider("청크 크기", 2000, 6000, 4000, help="긴 문서를 나누는 크기")
            
            # 피드백 길이 옵션
            st.markdown("**📝 피드백 길이**")
            feedback_length = st.radio("피드백 길이 선택", ["간단", "보통", "상세"], index=1, help="피드백의 상세 정도를 선택하세요")
            st.session_state.feedback_length = feedback_length
            
            # 정보 표시
            st.markdown("---")
            st.markdown("**ℹ️ 지원 문서 유형**")
            available_types = self.template_loader.get_available_types()
            for i, doc_type in enumerate(available_types[:5]):  # 처음 5개만 표시
                st.markdown(f"• {doc_type}")
            
            if len(available_types) > 5:
                with st.expander(f"더보기 ({len(available_types)-5}개)"):
                    for doc_type in available_types[5:]:
                        detailed_description = self.document_classifier.get_detailed_document_type_description(doc_type)
                        st.markdown(f"**{doc_type}**")
                        st.markdown(f"<small>{detailed_description}</small>", unsafe_allow_html=True)
                        st.markdown("")
            
            # 지원 파일 형식
            st.markdown("### 📁 지원 파일 형식")
            supported_formats = self.document_loader.get_supported_formats()
            for fmt in supported_formats:
                st.markdown(f"• {fmt}")
    
    def render_upload_section(self):
        """파일 업로드 및 가이드라인 후보 선택 섹션 렌더링 (상태 안내 및 시각화 개선, 추론 캐싱)"""
        st.markdown('<h2 class="sub-header">📁 문서 업로드</h2>', unsafe_allow_html=True)
        supported_formats = self.document_loader.get_supported_formats()
        st.info(f"지원 파일 형식: {', '.join(supported_formats)}")
        uploaded_file = st.file_uploader(
            "문서 파일을 선택하세요",
            type=["hwp", "docx"],
            help="HWP 또는 DOCX 파일만 업로드 가능합니다"
        )
        if uploaded_file is not None:
            st.session_state.uploaded_file = uploaded_file
            file_details = {
                "파일명": uploaded_file.name,
                "파일 크기": f"{uploaded_file.size / 1024:.1f} KB",
                "파일 타입": uploaded_file.type
            }
            st.markdown("### 📋 업로드된 파일 정보")
            for key, value in file_details.items():
                st.write(f"**{key}**: {value}")
            # 1. 문서 텍스트 추출 안내
            st.info("문서 텍스트 추출 중...")
            file_ext = Path(uploaded_file.name).suffix.lower()
            with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_file_path = tmp_file.name
            try:
                extracted_text = self.document_loader.load_document(tmp_file_path)
                if extracted_text:
                    st.success(f"텍스트 추출 완료 ({len(extracted_text)}자)")
                else:
                    st.warning("텍스트 추출 실패: 파일에서 텍스트를 추출할 수 없습니다.")
            except Exception as e:
                st.warning(f"텍스트 추출 실패: {e}")
                extracted_text = ""
            finally:
                os.unlink(tmp_file_path)
            # 2. 제품 유형 추론 캐싱
            product_type = None
            candidates = []
            selected_candidate = None
            if extracted_text:
                st.info("🏷️ 제품 유형 추론 중...")
                # 텍스트 해시로 캐싱
                text_hash = hash(extracted_text)
                if 'cached_product_type' in st.session_state and st.session_state.get('cached_text_hash') == text_hash:
                    product_type = st.session_state['cached_product_type']
                    confidence = st.session_state.get('cached_product_confidence', 1.0)
                    st.success(f"🏷️ (캐시) 제품 유형 자동 추론 완료: {product_type} (신뢰도: {confidence:.2f})")
                else:
                    _, product_type, confidence = self.evaluation_selector.select_evaluation_criteria(extracted_text, self.template_loader)
                    st.session_state['cached_product_type'] = product_type
                    st.session_state['cached_product_confidence'] = confidence
                    st.session_state['cached_text_hash'] = text_hash
                    if product_type:
                        st.success(f"🏷️ 제품 유형 자동 추론 완료: {product_type} (신뢰도: {confidence:.2f})")
                    else:
                        st.warning("🏷️ 제품 유형 추론 실패")
                        product_type = "기타"
                # 3. 유사한 가이드라인 후보 안내 및 시각화
                if product_type and product_type != "기타":
                    candidates = self.template_loader.get_similar_product_candidates(product_type, top_n=5)
                    if candidates:
                        st.info(f"🔍 유사한 가이드라인 후보들 (상위 {len(candidates)}개):")
                        for i, (file_path, score) in enumerate(candidates, 1):
                            if score >= 0.8:
                                color = "#6bcf7f"; emoji = "✅"
                            elif score >= 0.5:
                                color = "#ffd93d"; emoji = "⚠️"
                            else:
                                color = "#ff6b6b"; emoji = "❌"
                            display_name = file_path.stem
                            if len(display_name) > 50:
                                display_name = display_name[:47] + "..."
                            bar_width = int(score * 100)
                            st.markdown(f"""
                            <div class="candidate-item">
                                <div style="display: flex; align-items: center; margin-bottom: 5px;">
                                    <span style="font-size: 20px; margin-right: 10px;">{emoji}</span>
                                    <strong>{i}. {display_name}</strong>
                                </div>
                                <div style="display: flex; align-items: center; gap: 10px;">
                                    <div style="flex: 1;">
                                        <div class="similarity-bar" style="background: linear-gradient(90deg, #ff6b6b 0%, #ffd93d 50%, #6bcf7f 100%); height: 20px; border-radius: 10px; position: relative; margin: 5px 0;">
                                            <div class="similarity-fill" style="width: {bar_width}%; background: {color}; height: 100%; border-radius: 10px; transition: width 0.3s ease;"></div>
                                            <div class="similarity-text" style="position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); color: white; font-weight: bold; font-size: 12px; text-shadow: 1px 1px 2px rgba(0,0,0,0.5);">{score:.2f}</div>
                                        </div>
                                    </div>
                                    <span style="font-weight: bold; color: {color}; min-width: 50px;">{score:.2f}</span>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                        candidate_names = [f"{file_path.stem} (유사도: {score:.2f})" for file_path, score in candidates]
                        selected_candidate = st.selectbox(
                            "가이드라인 선택:",
                            candidate_names,
                            index=0,
                            help="평가에 사용할 가이드라인을 선택하세요"
                        )
                        st.session_state.selected_guideline = candidates[candidate_names.index(selected_candidate)][0]  # 파일 경로 저장
            return uploaded_file
        return None
    
    def evaluate_document(self, uploaded_file, document_type=None, guideline_path=None):
        """문서 평가 실행 (가이드라인 경로를 명시적으로 받음)"""
        try:
            with st.spinner("문서를 평가하는 중입니다..."):
                file_ext = Path(uploaded_file.name).suffix.lower()
                with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_file_path = tmp_file.name
                try:
                    # 문서 텍스트 추출
                    with st.status("문서 텍스트 추출 중...", expanded=True) as status:
                        try:
                            extracted_text = self.document_loader.load_document(tmp_file_path)
                            if not extracted_text:
                                status.update(label="텍스트 추출 실패", state="error")
                                st.error("해당 파일에서 텍스트를 추출할 수 없습니다.")
                                return False
                            status.update(label=f"텍스트 추출 완료 ({len(extracted_text)} 문자)", state="complete")
                        except Exception as e:
                            status.update(label="텍스트 추출 실패", state="error")
                            st.error("해당 파일에서 텍스트를 추출할 수 없습니다.")
                            logger.error(f"텍스트 추출 실패: {e}")
                            return False
                    # 문서 유형별 평가 기준 로드
                    document_criteria = None
                    if document_type:
                        st.info(f"📄 선택된 문서 유형: {document_type}")
                        document_criteria = self.template_loader.get_guidelines(str(document_type), type_hint="document")
                        if document_criteria:
                            st.success(f"✅ 문서 유형별 평가 기준 로드 완료: {document_type}")
                        else:
                            st.warning(f"⚠️ 문서 유형 '{document_type}'에 대한 평가 기준을 찾을 수 없습니다.")
                    else:
                        st.info("📄 자동 분류 모드: 문서 유형을 자동으로 감지합니다.")
                    # 선택된 가이드라인 로드
                    product_criteria = None
                    selected_product_guideline = None
                    if guideline_path:
                        try:
                            product_criteria = self.template_loader._load_and_clean_json(guideline_path)
                            if product_criteria:
                                selected_product_guideline = Path(guideline_path).stem
                                st.success(f"✅ 선택된 가이드라인: {selected_product_guideline}")
                            else:
                                st.error(f"❌ 선택된 가이드라인 로드 실패: {Path(guideline_path).stem}")
                        except Exception as e:
                            st.error(f"❌ 선택된 가이드라인 로드 실패: {e}")
                    # 평가 실행 (이하 기존과 동일)
                    with st.status("문서 평가 중...", expanded=True) as status:
                        combined_criteria = {}
                        if document_criteria and "evaluation_criteria" in document_criteria:
                            for key, value in document_criteria["evaluation_criteria"].items():
                                combined_criteria[f"[문서유형] {key}"] = value
                        if product_criteria and "evaluation_criteria" in product_criteria:
                            for key, value in product_criteria["evaluation_criteria"].items():
                                combined_criteria[f"[제품가이드라인] {key}"] = value
                        if not combined_criteria:
                            st.warning("⚠️ 평가 기준을 찾을 수 없어 기본 기준을 사용합니다.")
                            combined_criteria = self._get_default_criteria()
                        feedback_length = st.session_state.get("feedback_length", "보통")
                        evaluation_result = self.evaluator.evaluate_document(
                            extracted_text, 
                            {"evaluation_criteria": combined_criteria}, 
                            str(document_type) if document_type else "기타",
                            feedback_length=feedback_length
                        )
                        evaluation_result["evaluation_metadata"] = {
                            "document_type": document_type,
                            "selected_product_guideline": selected_product_guideline,
                            "document_criteria_used": document_criteria is not None,
                            "product_criteria_used": product_criteria is not None,
                            "total_criteria_count": len(combined_criteria)
                        }
                        status.update(label="평가 완료", state="complete")
                    st.session_state['evaluation_result'] = evaluation_result
                    st.session_state['document_type'] = document_type
                    return True
                finally:
                    os.unlink(tmp_file_path)
        except Exception as e:
            st.error(f"평가 중 오류가 발생했습니다: {e}")
            logger.error(f"문서 평가 실패: {e}")
            return False
    
    def _get_default_criteria(self):
        """기본 평가 기준"""
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
    
    def render_evaluation_results(self):
        """평가 결과 렌더링"""
        if st.session_state.evaluation_result is None:
            return
        
        result = st.session_state.evaluation_result
        document_type = st.session_state.document_type
        
        st.markdown('<h2 class="sub-header">📊 평가 결과</h2>', unsafe_allow_html=True)
        
        # 평가 메타데이터 표시
        if "evaluation_metadata" in result:
            metadata = result["evaluation_metadata"]
            st.markdown("### 📋 평가 설정 정보")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.info(f"**문서 유형**: {metadata.get('document_type', '자동 감지')}")
                st.info(f"**제품 유형**: {metadata.get('product_type', '기타')}")
            with col2:
                st.info(f"**선택된 가이드라인**: {metadata.get('selected_product_guideline', '없음')}")
                st.info(f"**총 평가 기준 수**: {metadata.get('total_criteria_count', 0)}개")
            with col3:
                st.success(f"**문서 유형 기준 사용**: {'✅' if metadata.get('document_criteria_used') else '❌'}")
                st.success(f"**제품 가이드라인 사용**: {'✅' if metadata.get('product_criteria_used') else '❌'}")
        
        # 기본 정보
        st.markdown("### 📈 평가 점수 요약")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("문서 유형", document_type or "자동 감지")
        with col2:
            total_score = result.get("total_score", 0)
            st.metric("총점", f"{total_score:.1f}/100")
        with col3:
            grade = result.get("grade", "F")
            st.metric("등급", grade)
        with col4:
            missing_count = len(result.get("missing_sections", []))
            st.metric("누락 섹션", missing_count)
        
        # 점수 차트
        st.markdown("### 📊 항목별 점수")
        self._render_score_chart(result)
        
        # 항목별 상세 평가
        st.markdown("### 📋 항목별 상세 평가")
        self._render_detailed_evaluation(result)
        
        # 누락 섹션
        missing_sections = result.get("missing_sections", [])
        if missing_sections:
            st.markdown("### ⚠️ 누락된 필수 섹션")
            for section in missing_sections:
                print_long_text(f"• {section}")
        
        # 전체 피드백
        st.markdown("### 💬 전체 피드백")
        overall_feedback = result.get("overall_feedback", "피드백이 없습니다.")
        print_long_text(overall_feedback)
        
        # 개선 권장사항
        recommendations = result.get("recommendations", [])
        if recommendations:
            st.markdown("### 🔧 개선 권장사항")
            for i, rec in enumerate(recommendations, 1):
                print_long_text(f"{i}. {rec}")
        
        # 중요 문장
        if "중요문장" in result:
            st.markdown("### 🔍 중요 문장")
            key_sentences = result["중요문장"]
            for i, sentence in enumerate(key_sentences, 1):
                print_long_text(f"**{i}.** {sentence}")
        
        # 개선 권장사항
        st.markdown("### 💡 개선 권장사항")
        recommendations = self._generate_recommendations(result)
        for i, recommendation in enumerate(recommendations, 1):
            print_long_text(f"**{i}.** {recommendation}")
        
        # 다운로드 버튼
        st.markdown("### 📥 결과 다운로드")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📝 마크다운 보고서 다운로드"):
                self._download_markdown_report(result, document_type)
        
        with col2:
            if st.button("📋 JSON 결과 다운로드"):
                self._download_json_report(result, document_type)
    
    def _render_score_chart(self, result):
        """점수 차트 렌더링"""
        # 평가 항목별 점수 추출
        scores = []
        labels = []
        weights = []
        
        evaluation_results = result.get("evaluation_results", {})
        for criterion_name, criterion_result in evaluation_results.items():
            if isinstance(criterion_result, dict) and 'score' in criterion_result:
                scores.append(criterion_result['score'])
                labels.append(criterion_name)
                weights.append(criterion_result.get('weight', 0.25))
        
        if not scores:
            st.warning("표시할 점수 데이터가 없습니다.")
            return
        
        # 막대 차트
        fig = go.Figure(data=[
            go.Bar(
                x=labels,
                y=scores,
                text=[f"{score:.1f}" for score in scores],
                textposition='auto',
                marker_color=['#1f77b4' if score >= 8 else '#ff7f0e' if score >= 6 else '#d62728' for score in scores],
                hovertemplate='<b>%{x}</b><br>점수: %{y:.1f}/10<br>가중치: %{customdata:.2f}<extra></extra>',
                customdata=weights
            )
        ])
        
        fig.update_layout(
            title="항목별 평가 점수",
            xaxis_title="평가 항목",
            yaxis_title="점수 (1-10)",
            yaxis=dict(range=[0, 10]),
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # 레이더 차트
        if len(scores) >= 3:
            fig_radar = go.Figure()
            fig_radar.add_trace(go.Scatterpolar(
                r=scores,
                theta=labels,
                fill='toself',
                name='평가 점수'
            ))
            
            fig_radar.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 10]
                    )),
                showlegend=False,
                title="평가 점수 레이더 차트",
                height=400
            )
            
            st.plotly_chart(fig_radar, use_container_width=True)
    
    def _render_detailed_evaluation(self, result):
        """상세 평가 결과 렌더링"""
        evaluation_results = result.get("evaluation_results", {})
        
        for criterion_name, criterion_result in evaluation_results.items():
            if isinstance(criterion_result, dict):
                score = criterion_result.get("score", 0)
                good = criterion_result.get("good", "")
                bad = criterion_result.get("bad", "")
                suggestion = criterion_result.get("suggestion", "")
                feedback = criterion_result.get("feedback", "")
                weight = criterion_result.get("weight", 0)
                weighted_score = criterion_result.get("weighted_score", 0)
                sub_criteria_evaluation = criterion_result.get("sub_criteria_evaluation", [])
                with st.expander(f"{criterion_name} (점수: {score:.1f}/10, 가중치: {weight:.2f})"):
                    if good:
                        print_long_text(f"**잘한 점:** {good}")
                    if bad:
                        print_long_text(f"**아쉬운 점:** {bad}")
                    if suggestion:
                        print_long_text(f"**개선 제안:** {suggestion}")
                    if feedback:
                        print_long_text(f"**전체 요약:** {feedback}")
                    st.markdown(f"**가중 점수:** {weighted_score:.2f}")
                    if sub_criteria_evaluation:
                        st.markdown("**세부 기준 평가:**")
                        for sub_eval in sub_criteria_evaluation:
                            sub_score = sub_eval.get("score", 0)
                            sub_feedback = sub_eval.get("feedback", "")
                            print_long_text(f"- **{sub_eval.get('criterion', '')}**: {sub_score}/10 - {sub_feedback}")
    
    def _get_grade(self, score):
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
    
    def _generate_recommendations(self, result):
        """개선 권장사항 생성"""
        recommendations = []
        
        criteria_names = ["정확성", "표현력", "항목누락", "형식적합성"]
        for criterion in criteria_names:
            if criterion in result:
                criterion_eval = result[criterion]
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
        
        total_score = result.get("총점", 0)
        if total_score < 6:
            recommendations.append("전반적인 문서 품질 향상이 필요합니다. 전문가의 도움을 받아보세요.")
        elif total_score < 8:
            recommendations.append("문서 품질이 양호하지만, 일부 개선이 필요합니다.")
        else:
            recommendations.append("문서 품질이 우수합니다. 현재 수준을 유지하세요.")
        
        return recommendations
    
    def _download_markdown_report(self, result, document_type):
        """마크다운 보고서 다운로드"""
        try:
            md_content = self.formatter.format_to_markdown(
                result, document_type, st.session_state.uploaded_file.name
            )
            
            st.download_button(
                label="📝 마크다운 다운로드",
                data=md_content,
                file_name=f"평가보고서_{document_type}_{st.session_state.uploaded_file.name}.md",
                mime="text/markdown"
            )
        except Exception as e:
            st.error(f"마크다운 생성 중 오류: {e}")
    
    def _download_json_report(self, result, document_type):
        """JSON 보고서 다운로드"""
        try:
            json_content = self.formatter.format_to_json(
                result, document_type, st.session_state.uploaded_file.name
            )
            
            st.download_button(
                label="📋 JSON 다운로드",
                data=json_content,
                file_name=f"평가결과_{document_type}_{st.session_state.uploaded_file.name}.json",
                mime="application/json"
            )
        except Exception as e:
            st.error(f"JSON 생성 중 오류: {e}")
    
    def run(self):
        """애플리케이션 실행"""
        self.render_sidebar()
        self.render_header()
        uploaded_file = self.render_upload_section()
        guideline_path = st.session_state.get('selected_guideline', None)
        if uploaded_file is not None and guideline_path is not None:
            if st.button("🚀 평가 시작", type="primary", use_container_width=True):
                success = self.evaluate_document(uploaded_file, st.session_state.document_type, guideline_path=guideline_path)
                if success:
                    st.success("✅ 평가가 완료되었습니다!")
                    st.rerun()
            if st.session_state.evaluation_result is not None:
                self.render_evaluation_results()

def main():
    """메인 함수"""
    app = StreamlitApp()
    app.run()

if __name__ == "__main__":
    main() 