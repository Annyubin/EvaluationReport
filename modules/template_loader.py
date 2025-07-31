"""
평가 기준 및 가이드라인 로드 및 검증 모듈 (새로운 구조)
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import re # Added for similarity matching
import math # Added for NaN handling

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TemplateLoader:
    """평가 기준 및 가이드라인 로드 및 검증 클래스 (새로운 구조)"""
    
    def __init__(self):
        self.guidelines_dir = Path("guidelines")
        self.document_dir = self.guidelines_dir / "document"
        self.product_dir = self.guidelines_dir / "product"
        
        # 디렉토리 생성
        self._ensure_directories()
        
        # 기본 템플릿 생성 (필요한 경우)
        self._create_default_templates()
    
    def _clean_json_data(self, data: Any) -> Any:
        """JSON 데이터에서 NaN, 공백, 이상한 문자들을 정리"""
        if isinstance(data, dict):
            cleaned = {}
            for key, value in data.items():
                # 키 정리
                clean_key = self._clean_string(key)
                if clean_key:
                    cleaned[clean_key] = self._clean_json_data(value)
            return cleaned
        elif isinstance(data, list):
            return [self._clean_json_data(item) for item in data if item is not None]
        elif isinstance(data, str):
            return self._clean_string(data)
        elif isinstance(data, (int, float)):
            # NaN, inf 값 처리
            if math.isnan(data) or math.isinf(data):
                return None
            return data
        else:
            return data
    
    def _clean_string(self, text: str) -> str:
        """문자열에서 이상한 문자들을 정리"""
        if not isinstance(text, str):
            return str(text) if text is not None else ""
        
        # NaN, null 문자열 제거
        if text.lower() in ['nan', 'null', 'none', 'undefined']:
            return ""
        
        # 공백 정리
        text = re.sub(r'\s+', ' ', text.strip())
        
        # 제어 문자 제거 (탭, 개행 등은 공백으로 변환)
        text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', ' ', text)
        
        # 연속된 공백을 하나로
        text = re.sub(r' +', ' ', text)
        
        return text.strip()
    
    def _load_and_clean_json(self, file_path: Path) -> Optional[Dict]:
        """JSON 파일을 로드하고 전처리"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 파일이 너무 크면 경고
            if len(content) > 10 * 1024 * 1024:  # 10MB
                logger.warning(f"파일이 너무 큽니다: {file_path.name} ({len(content)/1024/1024:.1f}MB)")
                
                # 큰 파일의 경우 기본 구조만 확인
                try:
                    # 첫 1000자만 읽어서 JSON 구조 확인
                    sample_content = content[:1000]
                    if '"evaluation_criteria"' in sample_content or '"guidelines"' in sample_content:
                        logger.info(f"큰 파일에서 평가 기준 구조 발견: {file_path.name}")
                        # 기본 구조 반환
                        return {
                            "document_type": file_path.stem,
                            "evaluation_criteria": {
                                "기본 평가": {"description": f"{file_path.stem} 관련 기본 평가 기준"}
                            },
                            "file_size": f"{len(content)/1024/1024:.1f}MB",
                            "note": "대용량 파일로 인해 기본 구조만 로드됨"
                        }
                except Exception as e:
                    logger.error(f"큰 파일 구조 확인 실패: {e}")
            
            # JSON 파싱
            data = json.loads(content)
            
            # 전처리
            cleaned_data = self._clean_json_data(data)
            
            logger.info(f"JSON 파일 로드 및 전처리 완료: {file_path.name}")
            return cleaned_data
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON 파싱 오류 {file_path.name}: {e}")
            # 파싱 실패 시 기본 구조 반환
            return {
                "document_type": file_path.stem,
                "evaluation_criteria": {
                    "기본 평가": {"description": f"{file_path.stem} 관련 기본 평가 기준"}
                },
                "error": f"JSON 파싱 오류: {str(e)}"
            }
        except Exception as e:
            logger.error(f"파일 로드 오류 {file_path.name}: {e}")
            return None
    
    def _ensure_directories(self):
        """필요한 디렉토리 생성"""
        self.document_dir.mkdir(parents=True, exist_ok=True)
        self.product_dir.mkdir(parents=True, exist_ok=True)
        logger.info("가이드라인 디렉토리 구조 확인 완료")
    
    def _create_default_templates(self):
        """기본 템플릿 생성 (필요한 경우)"""
        default_document_templates = {
            "위험관리계획서": {
                "document_type": "위험관리계획서",
                "evaluation_criteria": {
                    "위험 식별": {"description": "제품 사용 시 발생 가능한 위험을 명확히 식별하였는가?"},
                    "위험 평가 방법": {"description": "발생 가능성과 심각도 등 정량적/정성적 평가가 수행되었는가?"},
                    "위험 통제 조치": {"description": "식별된 위험에 대해 적절한 통제 방안이 제시되었는가?"}
                }
            },
            "임상시험계획서": {
                "document_type": "임상시험계획서", 
                "evaluation_criteria": {
                    "연구 목적": {"description": "임상시험의 목적이 명확히 기술되었는가?"},
                    "연구 설계": {"description": "적절한 연구 설계가 제시되었는가?"},
                    "통계 분석 계획": {"description": "통계 분석 방법이 명확히 기술되었는가?"}
                }
            },
            "CTD 문서": {
                "document_type": "CTD 문서",
                "evaluation_criteria": {
                    "모듈 1": {"description": "행정 정보 및 제품 정보가 완전한가?"},
                    "모듈 2": {"description": "CTD 요약서가 적절히 작성되었는가?"},
                    "모듈 3": {"description": "품질 관련 자료가 충분한가?"}
                }
            },
            "DMF 자료": {
                "document_type": "DMF 자료",
                "evaluation_criteria": {
                    "제조 정보": {"description": "제조 공정 정보가 상세히 기술되었는가?"},
                    "품질 관리": {"description": "품질 관리 방법이 적절한가?"},
                    "안정성 자료": {"description": "안정성 관련 자료가 충분한가?"}
                }
            }
        }
        
        # 기본 문서 유형별 템플릿 생성
        for doc_type, template in default_document_templates.items():
            file_path = self.document_dir / f"{doc_type}.json"
            if not file_path.exists():
                self._save_template(file_path, template)
                logger.info(f"기본 템플릿 생성 완료: {doc_type}")
    
    def _save_template(self, file_path: Path, template: Dict):
        """템플릿을 JSON 파일로 저장"""
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(template, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"템플릿 저장 실패: {e}")
    
    def get_guidelines(self, document_type: str, type_hint: str = "auto") -> Optional[Dict]:
        """
        가이드라인 로드 (새로운 구조)
        
        Args:
            document_type: 문서/제품 유형
            type_hint: "document", "product", "auto" 중 하나
        
        Returns:
            가이드라인 데이터 또는 None
        """
        try:
            # type_hint에 따른 로드 우선순위 결정
            if type_hint == "document":
                return self._load_from_document_dir(document_type)
            elif type_hint == "product":
                return self._load_from_product_dir(document_type)
            else:  # auto
                # 제품별 가이드라인 우선, 없으면 문서 유형별 기준
                result = self._load_from_product_dir(document_type)
                if result:
                    return result
                return self._load_from_document_dir(document_type)
                
        except Exception as e:
            logger.error(f"가이드라인 로드 실패: {e}")
            return None
    
    def _load_from_document_dir(self, document_type: str) -> Optional[Dict]:
        """문서 유형별 가이드라인 로드"""
        file_path = self.document_dir / f"{document_type}.json"
        if file_path.exists():
            data = self._load_and_clean_json(file_path)
            if data:
                logger.info(f"문서 유형별 가이드라인 로드 완료: {document_type}")
                return data
            else:
                logger.error(f"문서 유형별 가이드라인 로드 실패: {document_type}")
        return None
    
    def _load_from_product_dir(self, product_type: str) -> Optional[Dict]:
        """제품별 가이드라인 로드 (유사도 기반 매칭)"""
        # 1. 정확한 파일명 매칭 시도
        file_path = self.product_dir / f"{product_type}.json"
        if file_path.exists():
            data = self._load_and_clean_json(file_path)
            if data:
                logger.info(f"제품별 가이드라인 로드 완료 (정확 매칭): {product_type}")
                return data
            else:
                logger.error(f"제품별 가이드라인 로드 실패: {product_type}")
        
        # 2. 유사도 기반 매칭 시도
        best_match = self._find_similar_product_file(product_type)
        if best_match:
            data = self._load_and_clean_json(best_match)
            if data:
                logger.info(f"제품별 가이드라인 로드 완료 (유사도 매칭): {product_type} -> {best_match.name}")
                return data
            else:
                logger.error(f"유사도 매칭 파일 로드 실패: {best_match.name}")
        
        # 3. 기본 템플릿 생성
        logger.info(f"제품별 가이드라인 파일이 없어 기본 템플릿을 생성합니다: {product_type}")
        default_template = self._create_default_product_template(product_type)
        if default_template:
            self._save_template(file_path, default_template)
            return default_template
        
        return None
    
    def _find_similar_product_file(self, product_type: str, top_n: int = 3) -> Optional[Path]:
        """유사도 기반으로 가장 유사한 제품 파일 찾기 (상위 N개 후보 반환)"""
        if not self.product_dir.exists():
            return None
        
        # 모든 파일의 유사도 점수 계산
        similarity_scores = []
        for file_path in self.product_dir.glob("*.json"):
            similarity_score = self._calculate_similarity(product_type, file_path.stem)
            if similarity_score > 0.1:  # 임계값을 0.1로 낮춤
                similarity_scores.append((file_path, similarity_score))
        
        # 유사도 점수로 정렬
        similarity_scores.sort(key=lambda x: x[1], reverse=True)
        
        if not similarity_scores:
            return None
        
        # 상위 N개 후보 로깅
        top_candidates = similarity_scores[:top_n]
        logger.info(f"유사도 매칭 후보 (상위 {len(top_candidates)}개):")
        for i, (file_path, score) in enumerate(top_candidates, 1):
            logger.info(f"  {i}. {file_path.stem} (점수: {score:.2f})")
        
        # 가장 높은 점수의 파일 반환
        best_match, best_score = similarity_scores[0]
        logger.info(f"선택된 파일: {best_match.stem} (점수: {best_score:.2f})")
        
        return best_match
    
    def get_similar_product_candidates(self, product_type: str, top_n: int = 5) -> List[Tuple[Path, float]]:
        """제품 유형에 대한 유사한 가이드라인 파일 후보 목록 반환"""
        if not self.product_dir.exists():
            return []
        
        # 모든 파일의 유사도 점수 계산
        similarity_scores = []
        for file_path in self.product_dir.glob("*.json"):
            similarity_score = self._calculate_similarity(product_type, file_path.stem)
            if similarity_score > 0.05:  # 더 낮은 임계값으로 더 많은 후보 포함
                similarity_scores.append((file_path, similarity_score))
        
        # 유사도 점수로 정렬하고 상위 N개 반환
        similarity_scores.sort(key=lambda x: x[1], reverse=True)
        return similarity_scores[:top_n]
    
    def _calculate_similarity(self, product_type: str, filename: str) -> float:
        """제품명과 파일명 간의 유사도 계산"""
        # 1. 정규화 (공백, 특수문자 제거, 소문자 변환)
        product_clean = re.sub(r'[^\w가-힣]', '', product_type.lower())
        filename_clean = re.sub(r'[^\w가-힣]', '', filename.lower())
        
        # 2. 부분 매칭 점수 계산
        partial_score = 0.0
        
        # 제품명이 파일명에 포함되는지 확인
        if product_clean in filename_clean:
            partial_score += 0.9
        
        # 파일명이 제품명에 포함되는지 확인
        if filename_clean in product_clean:
            partial_score += 0.7
        
        # 3. 키워드 매칭 점수 계산 (더 세밀하게)
        product_words = set(product_clean.split())
        filename_words = set(filename_clean.split())
        
        if product_words and filename_words:
            intersection = product_words.intersection(filename_words)
            union = product_words.union(filename_words)
            jaccard_score = len(intersection) / len(union) if union else 0.0
            partial_score += jaccard_score * 0.6
            
            # 공통 키워드가 있으면 추가 점수
            if intersection:
                partial_score += 0.3
        
        # 4. 문자열 유사도 (Levenshtein distance 기반)
        if len(product_clean) > 0 and len(filename_clean) > 0:
            max_len = max(len(product_clean), len(filename_clean))
            distance = self._levenshtein_distance(product_clean, filename_clean)
            similarity = 1 - (distance / max_len)
            partial_score += similarity * 0.3
        
        # 5. 특별한 키워드 매칭 (의료기기 관련)
        medical_keywords = ['의료기기', '의약품', '가이드라인', '가이던스', '허가', '심사', '평가']
        for keyword in medical_keywords:
            if keyword in filename_clean:
                partial_score += 0.1
        
        return min(partial_score, 1.0)
    
    def _levenshtein_distance(self, s1: str, s2: str) -> int:
        """Levenshtein 거리 계산"""
        if len(s1) < len(s2):
            return self._levenshtein_distance(s2, s1)
        
        if len(s2) == 0:
            return len(s1)
        
        previous_row = list(range(len(s2) + 1))
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        
        return previous_row[-1]
    
    def _create_default_product_template(self, product_type: str) -> Optional[Dict]:
        """제품별 기본 템플릿 생성"""
        default_templates = {
            "흡수성마그네슘": {
                "document_type": "흡수성마그네슘",
                "evaluation_criteria": {
                    "재료 특성": {"description": "흡수성 마그네슘 합금의 물리화학적 특성이 명확히 기술되었는가?"},
                    "생체적합성": {"description": "생체적합성 평가 결과가 적절히 제시되었는가?"},
                    "흡수 특성": {"description": "체내 흡수 특성과 분해 메커니즘이 설명되었는가?"},
                    "안전성 평가": {"description": "임상적 안전성 평가가 충분히 수행되었는가?"}
                }
            },
            "혈액제제": {
                "document_type": "혈액제제",
                "evaluation_criteria": {
                    "제조 품질": {"description": "혈액제제의 제조 품질 관리가 적절한가?"},
                    "안전성 검증": {"description": "바이러스 검증 등 안전성 검증이 수행되었는가?"},
                    "효능 평가": {"description": "임상적 효능이 충분히 입증되었는가?"},
                    "안정성 자료": {"description": "안정성 관련 자료가 충분한가?"}
                }
            },
            "혈관용스텐트": {
                "document_type": "혈관용스텐트",
                "evaluation_criteria": {
                    "기계적 특성": {"description": "스텐트의 기계적 특성이 적절히 평가되었는가?"},
                    "생체적합성": {"description": "생체적합성 평가가 충분한가?"},
                    "임상적 성능": {"description": "임상적 성능이 입증되었는가?"},
                    "안전성": {"description": "안전성 관련 자료가 충분한가?"}
                }
            }
        }
        
        return default_templates.get(product_type, {
            "document_type": product_type,
            "evaluation_criteria": {
                "기본 평가": {"description": f"{product_type} 제품에 대한 기본적인 품질 평가를 수행합니다."}
            }
        })
    
    def get_simple_criteria_json(self, document_type: str) -> Optional[Dict]:
        """
        간단한 평가 기준 JSON 반환 (기존 호환성 유지)
        
        Args:
            document_type: 문서 유형
        
        Returns:
            평가 기준 데이터 또는 None
        """
        # 문서 유형별 기준에서 로드
        data = self._load_from_document_dir(document_type)
        if data:
            return data
        
        # 기본 템플릿 반환
        return {
            "document_type": document_type,
            "evaluation_criteria": {
                "기본 평가": {"description": "문서의 기본적인 품질을 평가합니다."}
            }
        }
    
    def get_available_document_types(self) -> List[str]:
        """사용 가능한 문서 유형 목록 반환"""
        if not self.document_dir.exists():
            return []
        
        json_files = list(self.document_dir.glob("*.json"))
        return [f.stem for f in json_files]
    
    def get_available_product_types(self) -> List[str]:
        """사용 가능한 제품 유형 목록 반환"""
        if not self.product_dir.exists():
            return []
        
        json_files = list(self.product_dir.glob("*.json"))
        return [f.stem for f in json_files]
    
    def validate_template_structure(self, template: Dict) -> bool:
        """템플릿 구조 검증"""
        try:
            # 필수 필드 확인
            if "document_type" not in template:
                return False
            
            # evaluation_criteria 또는 guidelines 필드 중 하나는 있어야 함
            if "evaluation_criteria" not in template and "guidelines" not in template:
                return False
            
            return True
            
        except Exception:
            return False
    
    def create_template(self, document_type: str, criteria: Dict, template_type: str = "document") -> bool:
        """
        새로운 템플릿 생성
        
        Args:
            document_type: 문서/제품 유형
            criteria: 평가 기준 또는 가이드라인 데이터
            template_type: "document" 또는 "product"
        
        Returns:
            생성 성공 여부
        """
        try:
            template = {
                "document_type": document_type,
                **criteria
            }
            
            if template_type == "document":
                file_path = self.document_dir / f"{document_type}.json"
            else:
                file_path = self.product_dir / f"{document_type}.json"
            
            if self.validate_template_structure(template):
                self._save_template(file_path, template)
                logger.info(f"템플릿 생성 완료: {document_type}")
                return True
            else:
                logger.error(f"템플릿 구조가 올바르지 않습니다: {document_type}")
                return False
                
        except Exception as e:
            logger.error(f"템플릿 생성 실패: {e}")
            return False

    def get_available_types(self) -> List[str]:
        """사용 가능한 문서 유형 목록 반환 (기존 호환성)"""
        return self.get_available_document_types()
    
    def get_template(self, document_type: str) -> Optional[Dict]:
        """특정 문서 유형의 템플릿 반환 (기존 호환성)"""
        return self._load_from_document_dir(document_type)
    
    def get_all_templates(self) -> Dict[str, Dict]:
        """모든 템플릿 반환 (기존 호환성)"""
        templates = {}
        if self.document_dir.exists():
            for json_file in self.document_dir.glob("*.json"):
                try:
                    with open(json_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        templates[json_file.stem] = data
                except Exception as e:
                    logger.error(f"템플릿 로드 실패 {json_file}: {e}")
        return templates 