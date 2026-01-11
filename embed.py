import re
import duckdb
import torch
import gc  # [추가] 메모리 청소용
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from typing import Dict, List, Any, Tuple

# [설정 수정됨]
MODEL_NAME = 'BAAI/bge-m3'
DB_PATH = "farming_granular.duckdb"

# 8GB 램 생존 설정
BATCH_SIZE = 1           # [중요] 한 번에 하나씩 처리 (RAM 폭증 방지)
DB_INSERT_BATCH = 50     # DB 저장은 50개씩 모아서
MAX_TEXT_LENGTH = 1536   # [타협] 2048 -> 1536 (약 25% 부하 감소, 여전히 충분히 김)

# [태그 사전]
TAG_SETS = {
    "crop": ["벼", "보리", "밀", "콩", "옥수수", "감자", "고구마", "고추", "배추", "무", "마늘", "양파", "오이", "토마토", "딸기", "수박", "복숭아", "사과", "배", "포도", "감", "인삼", "오미자", "깨", "소", "돼지", "닭", "꿀벌"],
    "task": ["파종", "육묘", "정식", "이앙", "물관리", "비료", "제초", "전정", "적과", "방제", "수확", "건조", "저장", "종자신청", "방역", "농기계점검", "요약"],
    "env": ["기상전망", "태풍", "장마", "가뭄", "폭염", "동해", "냉해", "집중호우", "일조량", "저수율", "시설하우스", "화재예방", "월동관리"],
    "pest": ["탄저병", "도열병", "흰가루병", "과수화상병", "진딧물", "응애", "총채벌레", "멸구", "구제역", "AI", "ASF"],
    "admin": ["PLS", "비료", "보급종", "재해보험", "시범사업", "농약"]
}

# [정규식 컴파일]
COMPILED_PATTERNS = {}
PARTICLES = "(?:은|는|이|가|을|를|의|와|과|도|로|에|서)?"

for category, tags in TAG_SETS.items():
    one_char_tags = [re.escape(tag) for tag in tags if len(tag) == 1]
    multi_char_tags = [re.escape(tag) for tag in tags if len(tag) > 1]
    patterns = []
    if one_char_tags:
        patterns.append(f"(?<![가-힣])((?:{'|'.join(one_char_tags)})){PARTICLES}(?![가-힣])")
    if multi_char_tags:
        patterns.append(f"((?:{'|'.join(multi_char_tags)}))")
    if patterns:
        COMPILED_PATTERNS[category] = re.compile("|".join(patterns))
    else:
        COMPILED_PATTERNS[category] = None

def init_db(con: duckdb.DuckDBPyConnection, embedding_dim: int) -> None:
    try:
        con.execute("INSTALL vss; LOAD vss;") 
    except Exception as e:
        print(f"⚠️ VSS 확장 로드 경고: {e}")
    con.execute("CREATE SEQUENCE IF NOT EXISTS seq_id START 1;")
    con.execute(f"""
        CREATE TABLE IF NOT EXISTS farm_info (
            id INTEGER PRIMARY KEY DEFAULT nextval('seq_id'),
            year INTEGER, month INTEGER,
            title TEXT,
            tags_crop VARCHAR[], tags_task VARCHAR[], tags_env VARCHAR[],
            tags_pest VARCHAR[], tags_admin VARCHAR[],
            content_md TEXT,
            embedding FLOAT[{embedding_dim}]
        )
    """)

def extract_smart_tags_optimized(text: str) -> Dict[str, List[str]]:
    extracted = {}
    for category, pattern in COMPILED_PATTERNS.items():
        if pattern:
            matches = pattern.findall(text)
            cleaned_matches = {next(filter(None, match), '') for match in matches if match}
            if '' in cleaned_matches: cleaned_matches.remove('')
            extracted[category] = sorted(list(cleaned_matches))
        else:
            extracted[category] = []
    return extracted

def clean_markdown(text: str) -> str:
    text = re.sub(r'\[.*?\]\(.*?\)', ' ', text)
    text = re.sub(r'[\|\-]', ' ', text) # 표 기호 제거
    text = re.sub(r'[#*`>]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def flush_buffer_to_db(con: duckdb.DuckDBPyConnection, buffer: List[Tuple]) -> None:
    if not buffer: return
    try:
        con.executemany("""
            INSERT INTO farm_info (year, month, title, tags_crop, tags_task, tags_env, tags_pest, tags_admin, content_md, embedding)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, buffer)
    except duckdb.Error as e:
        print(f"❌ DB 저장 중 오류 발생: {e}")

def build_database(md_file_path: str):
    print("📥 모델 로딩 중... (BGE-M3)")
    model = SentenceTransformer(MODEL_NAME, device='cpu')
    
    print("⚡ 모델 양자화 적용 중...")
    model = torch.quantization.quantize_dynamic(
        model, {torch.nn.Linear}, dtype=torch.qint8
    )
    
    embedding_dimension = model.get_sentence_embedding_dimension()
    print(f"✅ 모델 로딩 완료 (차원: {embedding_dimension})")

    con = duckdb.connect(DB_PATH)
    init_db(con, embedding_dimension)

    try:
        with open(md_file_path, 'r', encoding='utf-8') as f:
            data = f.read()
    except FileNotFoundError:
        print(f"❌ 파일을 찾을 수 없습니다: {md_file_path}")
        return

    raw_sections = re.split(r'\n#\s*(?=\[)', data)
    
    buffer_rows = []
    batch_texts = []
    batch_meta = []
    
    print("🔄 데이터 처리 및 임베딩 시작 (안전 모드)...")
    
    for section in tqdm(raw_sections):
        if not section.strip(): continue
        
        lines = section.strip().split('\n')
        header = lines[0]
        if not header.startswith('#'): header = '# ' + header
        body = "\n".join(lines[1:])

        if "목 차" in header: continue
        
        date_match = re.search(r'\[(\d{4})-(\d{2})', header)
        if not date_match: continue
        year, month = int(date_match.group(1)), int(date_match.group(2))
        
        clean_body = clean_markdown(body)
        full_text = (clean_markdown(header) + ". " + clean_body)[:MAX_TEXT_LENGTH]
        
        search_range = header + " " + body[:1000]
        tags = extract_smart_tags_optimized(search_range)
        
        batch_texts.append(full_text)
        batch_meta.append({
            'year': year, 'month': month, 'title': header,
            'tags': tags, 'content': body
        })
        
        # BATCH_SIZE = 1 이므로 매번 실행됨
        if len(batch_texts) >= BATCH_SIZE:
            try:
                embeddings = model.encode(batch_texts, show_progress_bar=False, batch_size=BATCH_SIZE)
                for meta, emb in zip(batch_meta, embeddings):
                    buffer_rows.append((
                        meta['year'], meta['month'], meta['title'],
                        meta['tags']['crop'], meta['tags']['task'], meta['tags']['env'],
                        meta['tags']['pest'], meta['tags']['admin'],
                        meta['content'], emb.tolist()
                    ))
            except Exception as e:
                print(f"⚠️ 임베딩 오류: {e}")
            finally:
                batch_texts = []
                batch_meta = []
        
        if len(buffer_rows) >= DB_INSERT_BATCH:
            flush_buffer_to_db(con, buffer_rows)
            buffer_rows = []
            
        # [중요] 반복마다 메모리 청소
        gc.collect()

    if batch_texts:
        embeddings = model.encode(batch_texts, show_progress_bar=False, batch_size=BATCH_SIZE)
        for meta, emb in zip(batch_meta, embeddings):
            buffer_rows.append((
                meta['year'], meta['month'], meta['title'],
                meta['tags']['crop'], meta['tags']['task'], meta['tags']['env'],
                meta['tags']['pest'], meta['tags']['admin'],
                meta['content'], emb.tolist()
            ))

    if buffer_rows:
        flush_buffer_to_db(con, buffer_rows)

    print("⏳ VSS 인덱스 생성 중... (HNSW)")
    try:
        con.execute("CREATE INDEX IF NOT EXISTS vss_idx ON farm_info USING HNSW (embedding);")
        print(f"🚀 성공: {DB_PATH} 생성 완료!")
    except Exception as e:
        print(f"❌ 인덱스 생성 실패: {e}")

    con.close()

if __name__ == "__main__":
    build_database("weekly.md")