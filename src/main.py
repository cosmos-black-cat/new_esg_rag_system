#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ESG資料提取系統 - 主程式 v2.1
專注於再生塑膠相關關鍵字的智能提取和去重，支援強排除模式
"""

import os
import sys
import shutil
import argparse
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple

# 添加當前目錄到路徑
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

# 模組級別導入配置，避免函數內import *問題
try:
    from config import (
        GOOGLE_API_KEY, GEMINI_MODEL, EMBEDDING_MODEL,
        VECTOR_DB_PATH, DATA_PATH, RESULTS_PATH, 
        CHUNK_SIZE, SEARCH_K, CONFIDENCE_THRESHOLD
    )
    CONFIG_LOADED = True
except ImportError as e:
    print(f"⚠️ 配置載入失敗: {e}")
    CONFIG_LOADED = False

# =============================================================================
# 系統檢查函數
# =============================================================================

def check_python_version():
    """檢查Python版本"""
    if sys.version_info < (3, 8):
        print("❌ Python版本過低")
        print(f"   當前版本: {sys.version}")
        print("   需要版本: >= 3.8")
        return False
    
    print(f"✅ Python版本: {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
    return True

def check_dependencies():
    """檢查必要的依賴包"""
    required_packages = {
        # 核心依賴
        'langchain': 'langchain',
        'langchain_community': 'langchain-community', 
        'langchain_google_genai': 'langchain-google-genai',
        'transformers': 'transformers',
        'sentence_transformers': 'sentence-transformers',
        'faiss': 'faiss-cpu',
        'google.generativeai': 'google-generativeai',
        'pypdf': 'pypdf',
        'dotenv': 'python-dotenv',
        'tqdm': 'tqdm',
        'numpy': 'numpy',
        
        # 數據處理依賴
        'pandas': 'pandas',
        'openpyxl': 'openpyxl',
    }
    
    missing_packages = []
    installed_packages = []
    
    print("🔍 檢查依賴包...")
    
    for import_name, package_name in required_packages.items():
        try:
            __import__(import_name)
            installed_packages.append(package_name)
        except ImportError:
            missing_packages.append(package_name)
    
    print(f"✅ 已安裝: {len(installed_packages)}/{len(required_packages)} 個包")
    
    if missing_packages:
        print("❌ 缺少以下依賴包:")
        for package in missing_packages:
            print(f"   • {package}")
        print("\n📦 安裝命令:")
        print(f"pip install {' '.join(missing_packages)}")
        print("或執行: pip install -r requirements.txt")
        return False
    
    return True

def check_environment():
    """檢查系統環境"""
    print("🔧 檢查系統環境...")
    print("=" * 50)
    
    # 檢查Python版本
    if not check_python_version():
        return False
    
    # 檢查依賴包
    if not check_dependencies():
        return False
    
    # 檢查配置載入
    if not CONFIG_LOADED:
        print("❌ 配置文件載入失敗")
        print("   請檢查 src/config.py 文件")
        return False
    
    # 檢查API Key
    if not GOOGLE_API_KEY:
        print("❌ Google API Key未設置")
        print("   請在 src/.env 文件中設置 GOOGLE_API_KEY")
        print("   獲取方式: https://makersuite.google.com/app/apikey")
        return False
    
    print(f"✅ Google API Key: {GOOGLE_API_KEY[:10]}...")
    print(f"✅ Gemini模型: {GEMINI_MODEL}")
    print(f"✅ Embedding模型: {EMBEDDING_MODEL}")
    
    # 檢查並創建目錄
    directories = {
        "數據目錄": DATA_PATH,
        "結果目錄": RESULTS_PATH,
        "向量資料庫目錄": os.path.dirname(VECTOR_DB_PATH)
    }
    
    for name, path in directories.items():
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
            print(f"✅ 創建{name}: {path}")
        else:
            print(f"✅ {name}: {path}")
    
    return True

def check_pdf_files() -> Tuple[bool, Optional[Path]]:
    """檢查PDF文件"""
    if not CONFIG_LOADED:
        print("❌ 配置未載入")
        return False, None
    
    try:
        data_dir = Path(DATA_PATH)
        pdf_files = list(data_dir.glob("*.pdf"))
        
        if not pdf_files:
            print(f"❌ 在 {DATA_PATH} 目錄中找不到PDF文件")
            print("   請將ESG報告書PDF文件放入data目錄")
            return False, None
        
        pdf_file = pdf_files[0]
        print(f"✅ 找到PDF文件: {pdf_file.name}")
        
        if len(pdf_files) > 1:
            print(f"ℹ️  找到多個PDF文件，將使用: {pdf_file.name}")
        
        return True, pdf_file
        
    except Exception as e:
        print(f"❌ 檢查PDF文件失敗: {e}")
        return False, None

def check_vector_database() -> bool:
    """檢查向量資料庫"""
    if not CONFIG_LOADED:
        print("❌ 配置未載入")
        return False
    
    try:
        if os.path.exists(VECTOR_DB_PATH):
            # 檢查資料庫是否完整
            required_files = ['index.faiss', 'index.pkl']
            missing_files = []
            
            for file in required_files:
                file_path = os.path.join(VECTOR_DB_PATH, file)
                if not os.path.exists(file_path):
                    missing_files.append(file)
            
            if missing_files:
                print(f"⚠️  向量資料庫不完整，缺少: {missing_files}")
                return False
            
            print(f"✅ 向量資料庫存在: {VECTOR_DB_PATH}")
            return True
        else:
            print(f"❌ 向量資料庫不存在: {VECTOR_DB_PATH}")
            return False
            
    except Exception as e:
        print(f"❌ 檢查向量資料庫失敗: {e}")
        return False

# =============================================================================
# 核心功能函數
# =============================================================================

def run_preprocessing(force: bool = False) -> bool:
    """執行PDF預處理"""
    if not CONFIG_LOADED:
        print("❌ 配置未載入")
        return False
    
    try:
        from preprocess import preprocess_documents
        
        # 如果強制重新處理，刪除現有資料庫
        if force and os.path.exists(VECTOR_DB_PATH):
            print("🗑️ 刪除舊的向量資料庫...")
            shutil.rmtree(VECTOR_DB_PATH)
        
        # 檢查是否需要預處理
        if not force and check_vector_database():
            print("ℹ️  向量資料庫已存在，跳過預處理")
            return True
        
        # 檢查PDF文件
        has_pdf, pdf_file = check_pdf_files()
        if not has_pdf:
            return False
        
        print("🔄 開始PDF預處理...")
        print("   這可能需要幾分鐘時間，請耐心等待...")
        
        # 執行預處理
        preprocess_documents(str(pdf_file))
        
        # 驗證結果
        if check_vector_database():
            print("✅ PDF預處理完成")
            return True
        else:
            print("❌ 預處理完成但向量資料庫驗證失敗")
            return False
            
    except Exception as e:
        print(f"❌ 預處理失敗: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_extraction(enable_llm: bool = True, max_docs: int = 200, enable_dedupe: bool = True, 
                  enable_strict_exclusion: bool = True) -> Optional[Tuple]:
    """執行資料提取（預設啟用強排除模式）"""
    try:
        from esg_extractor import ESGExtractor
        
        print("🚀 初始化ESG資料提取器...")
        print(f"📋 配置參數:")
        print(f"   LLM增強: {'✅ 已啟用' if enable_llm else '❌ 未啟用'}")
        print(f"   自動去重: {'✅ 已啟用' if enable_dedupe else '❌ 未啟用'}")
        print(f"   強排除模式: {'🔒 已啟用' if enable_strict_exclusion else '❌ 未啟用'}")
        
        extractor = ESGExtractor(
            enable_llm=enable_llm, 
            auto_dedupe=enable_dedupe,
            enable_strict_exclusion=enable_strict_exclusion
        )
        
        print("🔍 開始資料提取...")
        extractions, summary, excel_path = extractor.run_complete_extraction(max_docs)
        
        return extractions, summary, excel_path
        
    except Exception as e:
        print(f"❌ 資料提取失敗: {e}")
        import traceback
        traceback.print_exc()
        return None

def process_single_excel_file(file_path: str) -> Optional[str]:
    """處理單個Excel文件的去重"""
    try:
        from esg_extractor import ESGResultDeduplicator
        
        deduplicator = ESGResultDeduplicator()
        result_path = deduplicator.deduplicate_excel_file(file_path)
        
        return result_path
        
    except Exception as e:
        print(f"❌ 處理Excel文件失敗: {e}")
        return None

def find_latest_results_file() -> Optional[Path]:
    """找到最新的結果文件"""
    if not CONFIG_LOADED:
        return None
    
    try:
        results_dir = Path(RESULTS_PATH)
        current_dir = Path(".")
        
        search_dirs = [results_dir, current_dir]
        excel_files = []
        
        for search_dir in search_dirs:
            if search_dir.exists():
                excel_files.extend(search_dir.glob("esg_extraction_results_*.xlsx"))
        
        if not excel_files:
            return None
        
        # 返回最新的文件
        return max(excel_files, key=lambda x: x.stat().st_mtime)
        
    except Exception:
        return None

# =============================================================================
# 顯示和分析函數
# =============================================================================

def show_system_info():
    """顯示系統信息"""
    if not CONFIG_LOADED:
        print("❌ 配置未載入")
        return
    
    try:
        from esg_extractor import KeywordConfig, StrictExclusionConfig
        
        print("📋 系統配置信息")
        print("=" * 50)
        print(f"🤖 Gemini模型: {GEMINI_MODEL}")
        print(f"🧠 Embedding模型: {EMBEDDING_MODEL}")
        print(f"📚 向量資料庫: {VECTOR_DB_PATH}")
        print(f"📁 數據目錄: {DATA_PATH}")
        print(f"📊 結果目錄: {RESULTS_PATH}")
        print(f"🔢 文本塊大小: {CHUNK_SIZE}")
        print(f"🔍 搜索數量: {SEARCH_K}")
        print(f"📈 信心閾值: {CONFIDENCE_THRESHOLD}")
        
        print(f"\n🎯 關鍵字配置")
        print("=" * 50)
        keywords = KeywordConfig.get_all_keywords()
        continuous = [k for k in keywords if isinstance(k, str)]
        discontinuous = [k for k in keywords if isinstance(k, tuple)]
        
        print(f"總關鍵字數: {len(keywords)}")
        print(f"連續關鍵字: {len(continuous)}")
        print(f"不連續關鍵字: {len(discontinuous)}")
        
        print(f"\n連續關鍵字:")
        for keyword in continuous:
            print(f"   • {keyword}")
        
        print(f"\n不連續關鍵字:")
        for keyword in discontinuous:
            print(f"   • {' + '.join(keyword)}")
        
        print(f"\n🔒 強排除模式配置")
        print("=" * 50)
        exclusion_config = StrictExclusionConfig()
        print(f"排除關鍵字數量: {len(exclusion_config.EXCLUSION_KEYWORDS)}")
        print(f"確認關鍵字數量: {len(exclusion_config.CONFIRMATION_KEYWORDS)}")
        print(f"數值範圍: {exclusion_config.MIN_NUMERIC_VALUE} - {exclusion_config.MAX_NUMERIC_VALUE}")
        print(f"百分比範圍: {exclusion_config.MIN_PERCENTAGE_VALUE}% - {exclusion_config.MAX_PERCENTAGE_VALUE}%")
        print(f"段落長度範圍: {exclusion_config.MIN_PARAGRAPH_LENGTH} - {exclusion_config.MAX_PARAGRAPH_LENGTH} 字符")
        
        print(f"\n部分排除關鍵字:")
        for i, keyword in enumerate(exclusion_config.EXCLUSION_KEYWORDS[:10]):
            print(f"   • {keyword}")
        if len(exclusion_config.EXCLUSION_KEYWORDS) > 10:
            print(f"   ... 還有 {len(exclusion_config.EXCLUSION_KEYWORDS) - 10} 個")
            
    except Exception as e:
        print(f"❌ 顯示系統信息失敗: {e}")

def show_latest_results():
    """顯示最新結果"""
    if not CONFIG_LOADED:
        print("❌ 配置未載入")
        return
    
    try:
        import pandas as pd
        
        results_dir = Path(RESULTS_PATH)
        if not results_dir.exists():
            print("❌ 結果目錄不存在")
            return
        
        # 查找Excel文件
        excel_files = list(results_dir.glob("*.xlsx"))
        if not excel_files:
            print("❌ 沒有找到Excel結果文件")
            print("   請先執行資料提取")
            return
        
        # 找到最新的文件
        latest_file = max(excel_files, key=lambda x: x.stat().st_mtime)
        file_time = datetime.fromtimestamp(latest_file.stat().st_mtime)
        file_size = latest_file.stat().st_size / 1024
        
        print("📊 最新結果文件")
        print("=" * 50)
        print(f"📁 文件名: {latest_file.name}")
        print(f"🕒 修改時間: {file_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"📏 文件大小: {file_size:.1f} KB")
        print(f"🔗 完整路徑: {latest_file}")
        
        # 嘗試讀取並顯示摘要
        try:
            # 嘗試讀取不同的工作表名稱
            sheet_names = ['提取結果', 'extraction_results', 'results']
            df_results = None
            
            for sheet_name in sheet_names:
                try:
                    df_results = pd.read_excel(latest_file, sheet_name=sheet_name)
                    break
                except:
                    continue
            
            if df_results is None:
                df_results = pd.read_excel(latest_file)
            
            print(f"\n📈 內容摘要")
            print("=" * 50)
            print(f"總記錄數: {len(df_results)}")
            
            # 檢查關鍵字欄位
            keyword_cols = [col for col in df_results.columns if '關鍵字' in str(col) or 'keyword' in str(col).lower()]
            if keyword_cols:
                print(f"關鍵字種類: {df_results[keyword_cols[0]].nunique()}")
            
            # 檢查數據類型欄位
            type_cols = [col for col in df_results.columns if '類型' in str(col) or 'type' in str(col).lower()]
            if type_cols:
                type_counts = df_results[type_cols[0]].value_counts()
                print(f"\n數據類型分布:")
                for dtype, count in type_counts.items():
                    print(f"   {dtype}: {count} 個")
            
            # 檢查信心分數欄位
            conf_cols = [col for col in df_results.columns if '信心' in str(col) or 'confidence' in str(col).lower()]
            if conf_cols:
                avg_confidence = df_results[conf_cols[0]].mean()
                max_confidence = df_results[conf_cols[0]].max()
                min_confidence = df_results[conf_cols[0]].min()
                
                print(f"\n信心分數統計:")
                print(f"   平均: {avg_confidence:.3f}")
                print(f"   最高: {max_confidence:.3f}")
                print(f"   最低: {min_confidence:.3f}")
            
            # 嘗試讀取處理摘要工作表
            try:
                summary_df = pd.read_excel(latest_file, sheet_name='處理摘要')
                if not summary_df.empty:
                    print(f"\n📋 處理摘要:")
                    for _, row in summary_df.iterrows():
                        for col, val in row.items():
                            if pd.notna(val) and col != '項目':
                                print(f"   {col}: {val}")
            except:
                pass
            
            # 顯示樣例數據
            print(f"\n📋 樣例數據 (前3筆):")
            for i, (_, row) in enumerate(df_results.head(3).iterrows(), 1):
                print(f"\n{i}. ", end="")
                # 顯示主要欄位
                main_cols = ['關鍵字', 'keyword', '提取數值', 'value', '數據類型', 'data_type']
                for col_name in main_cols:
                    if col_name in df_results.columns:
                        print(f"{col_name}: {row[col_name]}", end="  ")
                print()
                
        except Exception as e:
            print(f"⚠️  無法讀取Excel內容: {e}")
            
    except Exception as e:
        print(f"❌ 查看結果失敗: {e}")

def test_system():
    """測試系統功能"""
    print("🧪 系統功能測試")
    print("=" * 50)
    
    tests = [
        ("Python版本", check_python_version),
        ("依賴包", check_dependencies),
        ("配置文件", lambda: CONFIG_LOADED),
        ("PDF文件", lambda: check_pdf_files()[0]),
        ("向量資料庫", check_vector_database),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n📋 測試: {test_name}")
        try:
            if test_func():
                print(f"✅ {test_name}: 通過")
                passed += 1
            else:
                print(f"❌ {test_name}: 失敗")
        except Exception as e:
            print(f"❌ {test_name}: 錯誤 - {e}")
    
    print(f"\n📊 測試結果: {passed}/{total} 通過")
    
    if passed == total:
        print("🎉 所有測試通過！系統可以正常運行")
        return True
    else:
        print("⚠️  部分測試失敗，請檢查相關配置")
        return False

# =============================================================================
# 去重功能
# =============================================================================

def deduplicate_existing_files():
    """去重現有的Excel文件"""
    if not CONFIG_LOADED:
        print("❌ 配置未載入")
        return
    
    try:
        print("🧹 Excel文件去重功能")
        print("=" * 50)
        
        # 搜索Excel文件
        results_dir = Path(RESULTS_PATH)
        current_dir = Path(".")
        
        search_dirs = [results_dir, current_dir]
        excel_files = []
        
        for search_dir in search_dirs:
            if search_dir.exists():
                files = list(search_dir.glob("esg_extraction_results_*.xlsx"))
                # 排除已去重的文件
                files = [f for f in files if "deduplicated" not in f.name]
                excel_files.extend(files)
        
        if not excel_files:
            print("❌ 未找到需要去重的Excel文件")
            print("   查找的文件模式: esg_extraction_results_*.xlsx")
            print("   查找的目錄:")
            for search_dir in search_dirs:
                print(f"     {search_dir}")
            return
        
        # 顯示找到的文件
        print(f"📁 找到 {len(excel_files)} 個文件:")
        for i, file in enumerate(excel_files, 1):
            file_time = datetime.fromtimestamp(file.stat().st_mtime)
            print(f"   {i}. {file.name} ({file_time.strftime('%Y-%m-%d %H:%M')})")
        
        # 選擇處理模式
        print(f"\n選擇處理模式:")
        print("1. 處理最新文件")
        print("2. 選擇特定文件")
        print("3. 批次處理所有文件")
        print("4. 返回主菜單")
        
        mode_choice = input("請選擇 (1-4): ").strip()
        
        if mode_choice == "1":
            # 處理最新文件
            latest_file = max(excel_files, key=lambda x: x.stat().st_mtime)
            print(f"\n🎯 處理最新文件: {latest_file.name}")
            result_path = process_single_excel_file(str(latest_file))
            if result_path:
                print(f"✅ 去重完成: {Path(result_path).name}")
        
        elif mode_choice == "2":
            # 選擇特定文件
            try:
                file_index = int(input(f"請選擇文件編號 (1-{len(excel_files)}): ")) - 1
                if 0 <= file_index < len(excel_files):
                    selected_file = excel_files[file_index]
                    print(f"\n🎯 處理選定文件: {selected_file.name}")
                    result_path = process_single_excel_file(str(selected_file))
                    if result_path:
                        print(f"✅ 去重完成: {Path(result_path).name}")
                else:
                    print("❌ 無效的文件編號")
            except ValueError:
                print("❌ 請輸入有效的數字")
        
        elif mode_choice == "3":
            # 批次處理
            confirm = input(f"確定要批次處理 {len(excel_files)} 個文件嗎？(y/n): ").strip().lower()
            if confirm == 'y':
                processed_count = 0
                for file in excel_files:
                    print(f"\n🔄 處理: {file.name}")
                    result_path = process_single_excel_file(str(file))
                    if result_path:
                        processed_count += 1
                        print(f"✅ 完成: {Path(result_path).name}")
                
                print(f"\n🎉 批次處理完成！成功處理 {processed_count}/{len(excel_files)} 個文件")
        
        elif mode_choice == "4":
            return
        
        else:
            print("❌ 無效選擇")
            
    except Exception as e:
        print(f"❌ 去重功能執行失敗: {e}")
        import traceback
        traceback.print_exc()

# =============================================================================
# 用戶界面
# =============================================================================

def show_usage_guide():
    """顯示使用說明"""
    print("\n💡 使用說明")
    print("=" * 60)
    print("""
📚 系統功能：
   專門提取ESG報告書中再生塑膠相關的數據
   
🎯 支援的關鍵字：
   • 再生塑膠、再生塑料、再生料、再生pp
   • PP回收、塑膠回收、PCR材料等不連續組合
   
🔍 提取內容：
   • 包含數值的段落（如：100KG、500噸）
   • 包含百分比的段落（如：30%、八成）
   
📋 基本流程：
   1. 將ESG報告書PDF放入 data/ 目錄
   2. 選擇功能1執行完整提取（含自動去重）
   3. 查看生成的Excel結果文件
   
🔒 強排除模式（新功能）：
   • 自動排除：計畫目標、財務數據、模糊表述
   • 確認實際：要求包含實際生產/使用的確認詞
   • 數值驗證：檢查數值和百分比的合理性範圍
   • 文本長度：過濾過短或過長的段落
   
🧹 去重功能：
   • 自動去重：提取過程中自動合併重複結果
   • 手動去重：選擇功能4處理現有Excel文件
   • 智能合併：相同數值+相似文本自動合併
   
📊 輸出結果：
   • 提取結果：清理後的主要數據
   • 關鍵字統計：各關鍵字的統計信息
   • 處理摘要：系統運行摘要（含強排除統計）
   
🔧 高級功能：
   • LLM增強：使用Gemini驗證提取準確性
   • 兩段式篩選：確保結果包含有意義的數值
   • 不連續關鍵字：支援"PP回收"等組合匹配
   • 強排除模式：預設啟用，提高結果精確度
   
⚡ 快速開始：
   1. 放入PDF到data目錄 → 2. 執行python main.py --auto → 3. 查看結果
   
❓ 常見問題：
   • 向量資料庫損壞：選擇功能2重新預處理
   • API錯誤：檢查.env文件中的API Key
   • 結果重複：使用功能4手動去重
   • 結果太少：可嘗試關閉強排除模式 --no-strict
""")

def interactive_menu():
    """互動式主選單"""
    while True:
        print("\n" + "🔷" * 20)
        print("🏢 ESG資料提取系統 v2.1")
        print("專注於再生塑膠相關關鍵字提取 + 智能去重 + 強排除模式")
        print("🔷" * 20)
        print("1. 📊 執行完整資料提取 (含自動去重 + 強排除模式)")
        print("2. 🔄 重新預處理PDF")
        print("3. 📋 查看最新結果")
        print("4. 🧹 去重現有Excel文件")
        print("5. ⚙️  顯示系統信息")
        print("6. 🧪 系統功能測試")
        print("7. 💡 使用說明")
        print("8. 🚪 退出系統")
        
        choice = input("\n請選擇功能 (1-8): ").strip()
        
        if choice == "1":
            # 執行完整資料提取
            print("\n🚀 準備執行資料提取...")
            
            if not check_environment():
                print("❌ 環境檢查失敗，無法執行提取")
                continue
            
            # 檢查向量資料庫
            if not check_vector_database():
                print("🔄 向量資料庫不存在，需要先預處理PDF")
                if run_preprocessing():
                    print("✅ 預處理完成，繼續提取...")
                else:
                    print("❌ 預處理失敗，無法執行提取")
                    continue
            
            # 詢問設定選項
            use_llm = input("是否啟用LLM增強？(y/n，預設y): ").strip().lower()
            enable_llm = use_llm != 'n'
            
            auto_dedupe = input("是否啟用自動去重？(y/n，預設y): ").strip().lower()
            enable_dedupe = auto_dedupe != 'n'
            
            strict_exclusion = input("是否啟用強排除模式？(y/n，預設y): ").strip().lower()
            enable_strict_exclusion = strict_exclusion != 'n'
            
            # 執行提取
            result = run_extraction(
                enable_llm=enable_llm, 
                enable_dedupe=enable_dedupe,
                enable_strict_exclusion=enable_strict_exclusion
            )
            if result:
                extractions, summary, excel_path = result
                print(f"\n🎉 提取完成！")
                print(f"📁 結果已保存: {excel_path}")
                
                # 詢問是否查看結果
                view_result = input("是否查看詳細結果？(y/n): ").strip().lower()
                if view_result == 'y':
                    show_latest_results()
            
        elif choice == "2":
            # 重新預處理PDF
            print("\n🔄 重新預處理PDF...")
            confirm = input("這將刪除現有向量資料庫，確定繼續？(y/n): ").strip().lower()
            if confirm == 'y':
                if run_preprocessing(force=True):
                    print("✅ 預處理完成，現在可以執行資料提取")
            
        elif choice == "3":
            # 查看最新結果
            show_latest_results()
            
        elif choice == "4":
            # 去重現有Excel文件
            deduplicate_existing_files()
            
        elif choice == "5":
            # 顯示系統信息
            show_system_info()
            
        elif choice == "6":
            # 系統功能測試
            test_system()
            
        elif choice == "7":
            # 使用說明
            show_usage_guide()
            
        elif choice == "8":
            # 退出
            print("👋 感謝使用ESG資料提取系統！")
            break
            
        else:
            print("❌ 無效選擇，請輸入1-8之間的數字")

def command_line_mode():
    """命令行模式"""
    parser = argparse.ArgumentParser(
        description="ESG資料提取系統 v2.1 - 專注於再生塑膠關鍵字 + 強排除模式",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用範例:
  python main.py                      # 互動模式
  python main.py --auto               # 自動執行完整流程（含強排除模式）
  python main.py --auto --no-strict   # 自動執行但關閉強排除模式
  python main.py --preprocess         # 僅預處理PDF
  python main.py --extract            # 僅執行提取
  python main.py --extract --no-strict # 提取但關閉強排除模式
  python main.py --dedupe [file]      # 去重Excel文件
  python main.py --test               # 系統測試
  python main.py --info               # 顯示系統信息
        """
    )
    
    parser.add_argument("--auto", action="store_true", help="自動執行完整流程")
    parser.add_argument("--preprocess", action="store_true", help="預處理PDF文件")
    parser.add_argument("--extract", action="store_true", help="執行資料提取")
    parser.add_argument("--dedupe", nargs="?", const="auto", help="去重Excel文件（可指定文件路徑）")
    parser.add_argument("--test", action="store_true", help="系統功能測試")
    parser.add_argument("--info", action="store_true", help="顯示系統信息")
    parser.add_argument("--results", action="store_true", help="查看最新結果")
    parser.add_argument("--no-llm", action="store_true", help="禁用LLM增強")
    parser.add_argument("--no-dedupe", action="store_true", help="禁用自動去重")
    parser.add_argument("--no-strict", action="store_true", help="禁用強排除模式")
    parser.add_argument("--max-docs", type=int, default=200, help="最大處理文檔數")
    
    args = parser.parse_args()
    
    # 根據參數執行對應功能
    if args.auto:
        # 自動執行完整流程
        print("🚀 自動執行模式")
        if not check_environment():
            sys.exit(1)
        
        if not check_vector_database():
            print("執行預處理...")
            if not run_preprocessing():
                sys.exit(1)
        
        print("執行資料提取...")
        enable_llm = not args.no_llm
        enable_dedupe = not args.no_dedupe
        enable_strict_exclusion = not args.no_strict  # 預設啟用強排除模式
        
        result = run_extraction(
            enable_llm=enable_llm, 
            max_docs=args.max_docs, 
            enable_dedupe=enable_dedupe,
            enable_strict_exclusion=enable_strict_exclusion
        )
        if result:
            print(f"✅ 完成！結果已保存")
        else:
            sys.exit(1)
            
    elif args.preprocess:
        if run_preprocessing(force=True):
            print("✅ 預處理完成")
        else:
            sys.exit(1)
            
    elif args.extract:
        enable_llm = not args.no_llm
        enable_dedupe = not args.no_dedupe
        enable_strict_exclusion = not args.no_strict  # 預設啟用強排除模式
        
        result = run_extraction(
            enable_llm=enable_llm, 
            max_docs=args.max_docs, 
            enable_dedupe=enable_dedupe,
            enable_strict_exclusion=enable_strict_exclusion
        )
        if not result:
            sys.exit(1)
    
    elif args.dedupe is not None:
        # 去重功能
        if args.dedupe == "auto":
            # 自動找到最新文件
            latest_file = find_latest_results_file()
            if latest_file:
                print(f"🎯 自動處理最新文件: {latest_file.name}")
                result_path = process_single_excel_file(str(latest_file))
                if result_path:
                    print(f"✅ 去重完成: {Path(result_path).name}")
                else:
                    sys.exit(1)
            else:
                print("❌ 未找到Excel結果文件")
                sys.exit(1)
        else:
            # 處理指定文件
            if os.path.exists(args.dedupe):
                result_path = process_single_excel_file(args.dedupe)
                if result_path:
                    print(f"✅ 去重完成: {Path(result_path).name}")
                else:
                    sys.exit(1)
            else:
                print(f"❌ 文件不存在: {args.dedupe}")
                sys.exit(1)
            
    elif args.test:
        if not test_system():
            sys.exit(1)
            
    elif args.info:
        show_system_info()
        
    elif args.results:
        show_latest_results()
        
    else:
        # 沒有參數，顯示幫助
        parser.print_help()

def main():
    """主函數"""
    print("🏢 ESG資料提取系統 v2.1")
    print("專注於再生塑膠關鍵字的智能提取")
    print("支援連續和不連續關鍵字匹配 + 智能去重 + 強排除模式")
    print("=" * 60)
    
    # 根據命令行參數決定運行模式
    if len(sys.argv) > 1:
        # 命令行模式
        command_line_mode()
    else:
        # 互動模式
        try:
            interactive_menu()
        except KeyboardInterrupt:
            print("\n👋 用戶中斷，系統退出")
        except Exception as e:
            print(f"\n❌ 系統錯誤: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()