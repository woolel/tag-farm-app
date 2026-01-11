import streamlit as st
import duckdb
import torch
from sentence_transformers import SentenceTransformer

# [페이지 설정]
st.set_page_config(page_title="주간농사정보 AI 검색", layout="wide")

# [1. 모델 로드]
@st.cache_resource
def load_model():
    # 무료 서버 용량에 맞는 가볍고 빠른 한국어 모델 사용
    model_name = 'jhgan/ko-sroberta-multitask'
    model = SentenceTransformer(model_name, device='cpu')
    return model

# [2. DB 연결]
@st.cache_resource
def get_db_connection():
    db_path = "farming_granular.duckdb"
    
    try:
        # [핵심] config를 통해 연결과 동시에 확장 설정 허용 (에러 방지)
        con = duckdb.connect(
            db_path, 
            read_only=True, 
            config={'allow_unsigned_extensions': 'true'}
        )
        
        # VSS(벡터 검색) 확장 설치 및 로드
        con.execute("INSTALL vss; LOAD vss;")
        return con
        
    except Exception as e:
        st.error(f"❌ 데이터베이스 연결 실패: {e}")
        return None

# [UI 구성]
st.title("🌾 주간농사정보 AI 검색 서비스")
st.caption("2023~2025년 농사 정보 (질문 예: 고추 탄저병 방제, 벼 이앙 적기)")

# 리소스 로드
with st.spinner("AI 모델과 데이터를 준비 중입니다..."):
    model = load_model()
    con = get_db_connection()

if not con:
    st.stop()

# 검색 인터페이스
query = st.text_input("질문을 입력하세요:", placeholder="예: 봄배추 육묘 온도 관리")

if query:
    if len(query) < 2:
        st.warning("검색어를 2글자 이상 입력해주세요.")
    else:
        # 1. 질문 임베딩 (768차원 벡터 생성)
        query_vector = model.encode(query).tolist()
        
        # 2. SQL 검색
        # [핵심] ?::FLOAT[768] -> 모델에 맞춰 차원수 변경 필수
        sql = """
            SELECT year, month, title, content_md, array_cosine_similarity(embedding, ?::FLOAT[768]) as score
            FROM farm_info
            ORDER BY score DESC
            LIMIT 5
        """
        
        try:
            results = con.execute(sql, [query_vector]).fetchall()
            
            if not results:
                st.info("검색 결과가 없습니다.")
            else:
                for row in results:
                    year, month, title, content, score = row
                    
                    # [핵심] 마크다운 취소선 문제 해결 (물결표 이스케이프)
                    safe_content = content.replace("~", "\~")
                    
                    # 결과 카드 출력
                    st.subheader(f"📅 [{year}-{month:02d}] {title}")
                    st.caption(f"유사도: {score:.4f}")
                    st.markdown(safe_content)
                    st.divider()
                    
        except Exception as e:
            st.error(f"검색 중 오류가 발생했습니다: {e}")
            st.caption("팁: DB 파일이 'jhgan/ko-sroberta-multitask' 모델로 생성되었는지 확인해주세요.")