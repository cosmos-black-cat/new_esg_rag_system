#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ESG報告書提取器配置文件 v1.0
載入環境變數並提供系統配置
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# 載入.env文件
env_path = Path(__file__).parent / '.env'
load_dotenv(dotenv_path=env_path)

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
# 基礎路徑
BASE_DIR = Path(__file__).parent

# 各目錄路徑
VECTOR_DB_PATH = os.getenv("VECTOR_DB_PATH", str(BASE_DIR / "vector_db" / "esg_db"))
DATA_PATH = os.getenv("DATA_PATH", str(BASE_DIR / "data"))
RESULTS_PATH = os.getenv("RESULTS_PATH", str(BASE_DIR / "results"))

# =============================================================================
# 文本處理參數
# =============================================================================
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))

# =============================================================================
# 搜索和匹配參數
# =============================================================================
SEARCH_K = int(os.getenv("SEARCH_K", "10"))
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.6"))

# =============================================================================
# 系統優化參數
# =============================================================================
MAX_DOCS_PER_RUN = int(os.getenv("MAX_DOCS_PER_RUN", "300"))
ENABLE_LLM_ENHANCEMENT = os.getenv("ENABLE_LLM_ENHANCEMENT", "true").lower() == "true"
LLM_MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", "3"))

# =============================================================================
# 自動創建必要目錄
# =============================================================================
def create_directories():
    """創建必要的目錄"""
    directories = [
        DATA_PATH,
        RESULTS_PATH,
        Path(VECTOR_DB_PATH).parent
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)

# 自動創建目錄
create_directories()

# =============================================================================
# 配置驗證
# =============================================================================
def validate_config():
    """驗證配置是否正確"""
    errors = []
    warnings = []
    
    # 檢查必需的API Key
    if not GOOGLE_API_KEY:
        errors.append("❌ GOOGLE_API_KEY 未設置")
        errors.append("   請在.env文件中設置 GOOGLE_API_KEY=your_api_key")
        errors.append("   獲取API Key: https://makersuite.google.com/app/apikey")
    
    # 檢查API Key格式
    elif not GOOGLE_API_KEY.startswith('AIza'):
        warnings.append("⚠️ API Key格式可能不正確，應以'AIza'開頭")
    
    # 檢查數值參數範圍
    if CHUNK_SIZE < 100 or CHUNK_SIZE > 2000:
        warnings.append(f"⚠️ CHUNK_SIZE ({CHUNK_SIZE}) 建議範圍：100-2000")
    
    if CONFIDENCE_THRESHOLD < 0 or CONFIDENCE_THRESHOLD > 1:
        errors.append(f"❌ CONFIDENCE_THRESHOLD ({CONFIDENCE_THRESHOLD}) 必須在0-1之間")
    
    if MAX_DOCS_PER_RUN < 10:
        warnings.append(f"⚠️ MAX_DOCS_PER_RUN ({MAX_DOCS_PER_RUN}) 過小，可能影響提取效果")
    
    # 檢查目錄權限
    try:
        test_file = Path(RESULTS_PATH) / "test_write.tmp"
        test_file.touch()
        test_file.unlink()
    except Exception:
        errors.append(f"❌ 無法寫入結果目錄: {RESULTS_PATH}")
    
    return errors, warnings

def print_config_status():
    """打印配置狀態"""
    errors, warnings = validate_config()
    
    if errors:
        print("🚫 配置錯誤:")
        for error in errors:
            print(f"   {error}")
        return False
    
    if warnings:
        print("⚠️ 配置警告:")
        for warning in warnings:
            print(f"   {warning}")
    
    print("✅ 配置驗證通過")
    return True

def get_config_summary():
    """獲取配置摘要"""
    return {
        "api_configured": bool(GOOGLE_API_KEY),
        "api_key_preview": GOOGLE_API_KEY[:10] + "..." if GOOGLE_API_KEY else "未設置",
        "gemini_model": GEMINI_MODEL,
        "embedding_model": EMBEDDING_MODEL,
        "data_path": DATA_PATH,
        "results_path": RESULTS_PATH,
        "vector_db_path": VECTOR_DB_PATH,
        "chunk_size": CHUNK_SIZE,
        "search_k": SEARCH_K,
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "max_docs": MAX_DOCS_PER_RUN,
        "llm_enhancement": ENABLE_LLM_ENHANCEMENT
    }

# =============================================================================
# 配置初始化檢查
# =============================================================================
def check_env_file():
    """檢查.env文件是否存在"""
    env_file = Path(__file__).parent / '.env'
    if not env_file.exists():
        print("⚠️ 未找到.env文件")
        print("請確保.env文件存在並設置API Key")
        return False
    return True

# =============================================================================
# 主執行部分
# =============================================================================
if __name__ == "__main__":
    print("🔧 ESG報告書提取器配置檢查")
    print("=" * 50)
    
    # 檢查.env文件
    if not check_env_file():
        sys.exit(1)
    
    # 驗證配置
    config_valid = print_config_status()
    
    if config_valid:
        print("\n📋 當前配置:")
        config = get_config_summary()
        for key, value in config.items():
            print(f"   {key}: {value}")
        
        print(f"\n📁 目錄狀態:")
        print(f"   數據目錄: {DATA_PATH} {'✅' if Path(DATA_PATH).exists() else '❌'}")
        print(f"   結果目錄: {RESULTS_PATH} {'✅' if Path(RESULTS_PATH).exists() else '❌'}")
        print(f"   向量DB目錄: {Path(VECTOR_DB_PATH).parent} {'✅' if Path(VECTOR_DB_PATH).parent.exists() else '❌'}")
        
        # 檢查PDF文件
        pdf_files = list(Path(DATA_PATH).glob("*.pdf"))
        print(f"\n📄 PDF文件: {len(pdf_files)} 個")
        for pdf in pdf_files[:3]:  # 只顯示前3個
            print(f"   - {pdf.name}")
        if len(pdf_files) > 3:
            print(f"   ... 還有 {len(pdf_files) - 3} 個文件")
    
    else:
        print("\n❌ 配置存在問題，請修復後重試")
        sys.exit(1)