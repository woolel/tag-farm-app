import streamlit as st
import duckdb
import torch
from sentence_transformers import SentenceTransformer

# [페이지 설정]
st.set_page_config(page_title="주간농사정보 AI 검색", layout="wide")

# [1. 모델 로드 및 최적화]
# @st.cache_resource는 모델을 한 번만 불러오고 메모리에 저장해둡니다.
@st.cache_resource
def load_model():
    # [수정] 가벼운 모델로 변경
    model_name = 'jhgan/ko-sroberta-multitask'
    model = SentenceTransformer(model_name, device='cpu')
    return model

# [2. DB 연결]
# [2. DB 연결 부분 수정]
@st.cache_resource
def get_db_connection():
    # 파일명이 맞는지 꼭 확인하세요
    db_path = "farming_granular.duckdb"
    
    try:
        # [핵심 수정] config 파라미터를 통해 연결과 동시에 설정을 적용합니다.
        # 이렇게 하면 "database is running" 에러가 발생하지 않습니다.
        con = duckdb.connect(
            db_path, 
            read_only=True, 
            config={'allow_unsigned_extensions': 'true'}
        )
        
        # VSS 확장 설치 및 로드
        # (이미 unsigned 설정을 켰으므로 에러 없이 진행됩니다)
        con.execute("INSTALL vss; LOAD vss;")
        return con
        
    except Exception as e:
        st.error(f"❌ 데이터베이스 연결 실패: {e}")
        return None
        
# [UI 구성]
st.title("🌾 주간농사정보 AI 검색 서비스")
st.caption("2023~2025년 농사 정보 (질문 예: 고추 탄저병 방제 시기는?, 벼 이앙 적기)")

# 로딩 인디케이터
with st.spinner("AI 모델과 농사 데이터를 불러오는 중입니다..."):
    model = load_model()
    con = get_db_connection()

if not con:
    st.stop() # DB 연결 실패 시 중단

# 검색창
query = st.text_input("질문을 입력하세요:", placeholder="예: 봄배추 육묘 온도 관리")

if query:
    if len(query) < 2:
        st.warning("검색어를 2글자 이상 입력해주세요.")
    else:
        # 1. 질문을 벡터로 변환
        query_vector = model.encode(query).tolist()
        
        # 2. SQL로 유사도 검색 (상위 5개)
        # BGE-M3 모델은 1024차원이므로 FLOAT[1024]로 형변환
        sql = """
            SELECT year, month, title, content_md, array_cosine_similarity(embedding, ?::FLOAT[768]) as score
            FROM farm_info
            ORDER BY score DESC
            LIMIT 5
        """
        
        try:
            results = con.execute(sql, [query_vector]).fetchall()
            
            # 3. 결과 출력
            if not results:
                st.info("검색 결과가 없습니다.")
            else:
                for row in results:
                    year, month, title, content, score = row
                    # 유사도 점수 표시 (선택사항)
                    st.subheader(f"📅 [{year}-{month:02d}] {title}")
                    st.caption(f"유사도: {score:.4f}")
                    # 마크다운 내용 출력
                    st.markdown(content)
                    st.divider()
                    
        except Exception as e:
            st.error(f"검색 중 오류가 발생했습니다: {e}")