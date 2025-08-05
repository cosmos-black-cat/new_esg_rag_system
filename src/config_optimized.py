import os
from dotenv import load_dotenv

# 載入.env文件
load_dotenv()

# =============================================================================
# API配置
# =============================================================================
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")

# =============================================================================
# 模型配置
# =============================================================================
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-large-zh-v1.5")

# =============================================================================
# 路徑配置
# =============================================================================
VECTOR_DB_PATH = os.getenv("VECTOR_DB_PATH", "./vector_db/esg_db")
DATA_PATH = os.getenv("DATA_PATH", "./data")
RESULTS_PATH = os.getenv("RESULTS_PATH", "./results")

# =============================================================================
# 處理參數配置
# =============================================================================
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))
SEARCH_K = int(os.getenv("SEARCH_K", "10"))
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.6"))

# =============================================================================
# 自動創建必要目錄
# =============================================================================
for path in [DATA_PATH, RESULTS_PATH, os.path.dirname(VECTOR_DB_PATH)]:
    os.makedirs(path, exist_ok=True)

# =============================================================================
# 簡單驗證（僅在直接執行時顯示）
# =============================================================================
if __name__ == "__main__":
    if not GOOGLE_API_KEY:
        print("⚠️ 警告: 未設置 GOOGLE_API_KEY")
        print("請在 .env 文件中設置 GOOGLE_API_KEY=your_api_key")
    else:
        print(f"✅ API Key已載入: {GOOGLE_API_KEY[:10]}...")

    print(f"✅ 配置載入完成")
    print(f"   數據目錄: {DATA_PATH}")
    print(f"   結果目錄: {RESULTS_PATH}")
    print(f"   向量資料庫: {VECTOR_DB_PATH}")