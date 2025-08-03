import os
from dotenv import load_dotenv
from typing import List

# 載入.env文件中的環境變數
load_dotenv()

# =============================================================================
# API配置
# =============================================================================
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")

# 多API Keys支援
def get_gemini_api_keys() -> List[str]:
    """獲取所有可用的Gemini API Keys"""
    # 優先使用多key配置
    multi_keys = os.getenv("GEMINI_API_KEYS")
    if multi_keys:
        keys = [key.strip() for key in multi_keys.split(",") if key.strip()]
        if keys:
            print(f"🔑 載入 {len(keys)} 個API Keys")
            return keys
    
    # 回退到單key配置
    single_key = os.getenv("GOOGLE_API_KEY")
    if single_key:
        print("🔑 載入單個API Key")
        return [single_key]
    
    return []

# 獲取API Keys
GEMINI_API_KEYS = get_gemini_api_keys()

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

# API管理參數
MIN_REQUEST_INTERVAL = float(os.getenv("MIN_REQUEST_INTERVAL", "1.0"))  # 最小請求間隔
API_COOLDOWN_MINUTES = int(os.getenv("API_COOLDOWN_MINUTES", "2"))      # API冷卻時間

# =============================================================================
# 驗證必要配置
# =============================================================================
def validate_config():
    """驗證必要的配置是否存在"""
    errors = []
    
    if not GEMINI_API_KEYS:
        errors.append("未設置任何Gemini API Key")
        errors.append("請在.env文件中設置 GOOGLE_API_KEY 或 GEMINI_API_KEYS")
    
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

# =============================================================================
# 配置資訊顯示
# =============================================================================
def print_config():
    """打印當前配置資訊"""
    print("🔧 當前系統配置:")
    print(f"   Gemini Model: {GEMINI_MODEL}")
    print(f"   API Keys數量: {len(GEMINI_API_KEYS)}")
    print(f"   Embedding Model: {EMBEDDING_MODEL}")
    print(f"   Vector DB Path: {VECTOR_DB_PATH}")
    print(f"   Data Path: {DATA_PATH}")
    print(f"   Results Path: {RESULTS_PATH}")
    print(f"   請求間隔: {MIN_REQUEST_INTERVAL}秒")
    print(f"   API冷卻時間: {API_COOLDOWN_MINUTES}分鐘")

# 自動驗證配置（當模組被導入時）
if __name__ != "__main__":
    try:
        validate_config()
        if len(GEMINI_API_KEYS) > 1:
            print(f"✅ 多API模式已啟用，共 {len(GEMINI_API_KEYS)} 個Keys")
    except ValueError as e:
        print(f"⚠️  配置警告: {e}")

if __name__ == "__main__":
    # 直接執行此文件時顯示配置
    print_config()
    validate_config()
    print("✅ 配置驗證通過！")