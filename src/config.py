import os
from dotenv import load_dotenv

# 載入.env文件中的環境變數
load_dotenv()

# =============================================================================
# API配置
# =============================================================================
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-pro")

# =============================================================================
# 模型配置
# =============================================================================
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-large-zh-v1.5")
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-base")

# =============================================================================
# 路徑配置
# =============================================================================
VECTOR_DB_PATH = os.getenv("VECTOR_DB_PATH", "./vector_db/esg_db")
DATA_PATH = os.getenv("DATA_PATH", "./data")
RESULTS_PATH = os.getenv("RESULTS_PATH", "./results")

# =============================================================================
# 處理參數配置
# =============================================================================
# 文本分割參數
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))

# 搜尋參數
SEARCH_K = int(os.getenv("SEARCH_K", "10"))
RERANK_K = int(os.getenv("RERANK_K", "3"))

# 信心分數門檻
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.5"))

# =============================================================================
# 驗證必要配置
# =============================================================================
def validate_config():
    """驗證必要的配置是否存在"""
    errors = []
    
    if not GOOGLE_API_KEY:
        errors.append("GOOGLE_API_KEY 未設置")
    
    if not os.path.exists(DATA_PATH):
        try:
            os.makedirs(DATA_PATH, exist_ok=True)
        except Exception as e:
            errors.append(f"無法創建數據目錄 {DATA_PATH}: {e}")
    
    if not os.path.exists(RESULTS_PATH):
        try:
            os.makedirs(RESULTS_PATH, exist_ok=True)
        except Exception as e:
            errors.append(f"無法創建結果目錄 {RESULTS_PATH}: {e}")
    
    if errors:
        raise ValueError("配置錯誤:\n" + "\n".join(f"- {error}" for error in errors))
    
    return True

# 自動驗證配置（當模組被導入時）
if __name__ != "__main__":
    try:
        validate_config()
    except ValueError as e:
        print(f"⚠️  配置警告: {e}")

# =============================================================================
# 配置資訊顯示
# =============================================================================
def print_config():
    """打印當前配置資訊"""
    print("🔧 當前系統配置:")
    print(f"   Gemini Model: {GEMINI_MODEL}")
    print(f"   Embedding Model: {EMBEDDING_MODEL}")
    print(f"   Reranker Model: {RERANKER_MODEL}")
    print(f"   Vector DB Path: {VECTOR_DB_PATH}")
    print(f"   Data Path: {DATA_PATH}")
    print(f"   Results Path: {RESULTS_PATH}")
    print(f"   Chunk Size: {CHUNK_SIZE}")
    print(f"   Search K: {SEARCH_K}")
    print(f"   Confidence Threshold: {CONFIDENCE_THRESHOLD}")

if __name__ == "__main__":
    # 直接執行此文件時顯示配置
    print_config()
    validate_config()
    print("✅ 配置驗證通過！")