import streamlit as st
import duckdb
import torch
from sentence_transformers import SentenceTransformer

# [페이지 설정]
st.set_page_config(page_title="주간농사정보 AI 검색", layout="wide")

# [1. 모델 로드 - 캐싱 필수]
# 매번 로딩하지 않도록 캐싱(@st.cache_resource) 사용
@st.cache_resource
def load_model():
    # CPU 환경에 맞춰 로드 및 양자화
    model = SentenceTransformer('BAAI/bge-m3', device='cpu')
    model = torch.quantization.quantize_dynamic(
        model, {torch.nn.Linear}, dtype=torch.qint8
    )
    return model

# [2. DB 연결 - VSS 확장 설치 필수]
@st.cache_resource
def get_db_connection():
    # read_only=True로 설정하여 파일 손상 방지
    con = duckdb.connect("farm_data_2026.duckdb", read_only=True)
    # Streamlit Cloud(Linux)에 맞는 VSS 확장 자동 설치
    con.execute("INSTALL vss; LOAD vss;")
    return con

st.title("🌾 주간농사정보 AI 검색 서비스")
st.caption("2023~2025년 농사 정보 (질문 예: 고추 탄저병 방제 시기는?)")

# 리소스 로드
with st.spinner("AI 모델과 데이터를 불러오는 중... (최초 1회만 느림)"):
    model = load_model()
    con = get_db_connection()

# 검색 인터페이스
query = st.text_input("질문을 입력하세요:", placeholder="예: 벼 이앙 시기")

if query:
    # 1. 질문 임베딩
    query_vector = model.encode(query).tolist()
    
    # 2. 벡터 검색 (상위 5개)
    sql = """
        SELECT year, month, title, content_md, array_cosine_similarity(embedding, ?::FLOAT[1024]) as score
        FROM farm_info
        ORDER BY score DESC
        LIMIT 5
    """
    results = con.execute(sql, [query_vector]).fetchall()
    
    # 3. 결과 출력
    for row in results:
        year, month, title, content, score = row
        with st.expander(f"[{year}-{month:02d}] {title} (유사도: {score:.4f})"):
            st.markdown(content)