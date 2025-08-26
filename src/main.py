#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ESG資料提取系統 - 平衡版主程式 v2.5
新增彙整功能，在提取準確度和覆蓋率之間取得平衡
"""

import os
import sys
import argparse
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple, Dict

# 添加當前目錄到路徑
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

# 配置載入
try:
    from config import (
        GOOGLE_API_KEY, GEMINI_MODEL, EMBEDDING_MODEL,
        VECTOR_DB_PATH, DATA_PATH, RESULTS_PATH, 
        CHUNK_SIZE, SEARCH_K, CONFIDENCE_THRESHOLD,
        MAX_DOCS_PER_RUN, ENABLE_LLM_ENHANCEMENT
    )
    CONFIG_LOADED = True
    print("✅ 配置載入成功")
except ImportError as e:
    print(f"❌ 配置載入失敗: {e}")
    print("請確保config.py文件存在且格式正確")
    CONFIG_LOADED = False

# =============================================================================
# 系統檢查函數
# =============================================================================

def check_environment():
    """檢查系統環境"""
    print("🔧 檢查系統環境...")
    
    if not CONFIG_LOADED:
        print("❌ 配置文件載入失敗")
        return False
    
    if not GOOGLE_API_KEY:
        print("❌ Google API Key未設置")
        print("請在.env文件中設置GOOGLE_API_KEY=your_api_key")
        return False
    
    print(f"✅ Google API Key: {GOOGLE_API_KEY[:10]}...")
    
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

def find_pdf_files() -> Tuple[bool, list]:
    """找到所有PDF文件"""
    if not CONFIG_LOADED:
        return False, []
    
    try:
        data_dir = Path(DATA_PATH)
        pdf_files = list(data_dir.glob("*.pdf"))
        
        if not pdf_files:
            print(f"❌ 在 {DATA_PATH} 目錄中找不到PDF文件")
            print("請將ESG報告PDF文件放入data目錄")
            return False, []
        
        print(f"✅ 找到 {len(pdf_files)} 個PDF文件:")
        for pdf_file in pdf_files:
            print(f"   - {pdf_file.name}")
        
        return True, pdf_files
        
    except Exception as e:
        print(f"❌ 查找PDF文件失敗: {e}")
        return False, []

# =============================================================================
# 核心功能函數
# =============================================================================

def run_preprocessing(pdf_files: list = None, force: bool = False) -> Optional[Dict]:
    """執行預處理，支援多文件"""
    try:
        from preprocess import preprocess_multiple_documents, DocumentMetadataExtractor
        
        if pdf_files is None:
            has_pdfs, pdf_files = find_pdf_files()
            if not has_pdfs:
                return None
        
        # 檢查是否需要預處理
        if not force:
            existing_dbs = []
            for pdf_file in pdf_files:
                pdf_name = pdf_file.stem
                db_path = os.path.join(
                    os.path.dirname(VECTOR_DB_PATH),
                    f"esg_db_{pdf_name}"
                )
                if os.path.exists(db_path):
                    existing_dbs.append(pdf_file.name)
            
            if existing_dbs and len(existing_dbs) == len(pdf_files):
                print("ℹ️  所有文件的向量資料庫已存在，跳過預處理")
                print("   如需重新處理，請使用 --force 參數")
                
                # 返回現有的文檔信息
                metadata_extractor = DocumentMetadataExtractor()
                docs_info = {}
                for pdf_file in pdf_files:
                    pdf_name = pdf_file.stem
                    metadata = metadata_extractor.extract_metadata(str(pdf_file))
                    docs_info[str(pdf_file)] = {
                        'db_path': os.path.join(os.path.dirname(VECTOR_DB_PATH), f"esg_db_{pdf_name}"),
                        'metadata': metadata,
                        'pdf_name': pdf_name
                    }
                return docs_info
        
        print("🔄 開始預處理...")
        print("   這可能需要幾分鐘時間，請耐心等待...")
        
        # 執行預處理
        results = preprocess_multiple_documents([str(f) for f in pdf_files])
        
        if results:
            print("✅ 預處理完成")
            return results
        else:
            print("❌ 預處理失敗")
            return None
            
    except Exception as e:
        print(f"❌ 預處理失敗: {e}")
        import traceback
        traceback.print_exc()
        return None

def run_balanced_extraction(docs_info: Dict, max_docs: int = None) -> Optional[Dict]:
    """執行平衡版資料提取"""
    try:
        from esg_extractor_balanced import BalancedMultiFileESGExtractor, DocumentInfo
        
        print("⚖️ 初始化平衡版ESG資料提取器...")
        extractor = BalancedMultiFileESGExtractor(enable_llm=ENABLE_LLM_ENHANCEMENT)
        
        # 使用配置中的最大文檔數
        if max_docs is None:
            max_docs = MAX_DOCS_PER_RUN
        
        # 轉換文檔信息格式
        document_infos = {}
        for pdf_path, info in docs_info.items():
            metadata = info['metadata']
            document_infos[pdf_path] = DocumentInfo(
                company_name=metadata['company_name'],
                report_year=metadata['report_year'],
                pdf_name=info['pdf_name'],
                db_path=info['db_path']
            )
        
        print("⚖️ 開始平衡版資料提取...")
        print(f"   最大處理文檔數: {max_docs}")
        print(f"   LLM增強: {'啟用' if ENABLE_LLM_ENHANCEMENT else '停用'}")
        print(f"   處理策略: 平衡準確度與覆蓋率")
        
        results = extractor.process_multiple_documents(document_infos, max_docs)
        
        return results
        
    except Exception as e:
        print(f"❌ 平衡版資料提取失敗: {e}")
        import traceback
        traceback.print_exc()
        return None

def run_consolidation() -> Optional[str]:
    """執行彙整功能，自動排除'無提取'檔案並統一公司名稱"""
    try:
        from consolidator import consolidate_esg_results
        
        print("\n📊 開始彙整ESG結果...")
        print("⚠️ 注意：檔名包含'無提取'的檔案將被自動排除")
        print("🏷️ 智能功能：同一公司的不同命名將自動統一")
        
        if not CONFIG_LOADED:
            print("❌ 配置未載入")
            return None
        
        # 檢查結果目錄是否存在
        if not os.path.exists(RESULTS_PATH):
            print(f"❌ 結果目錄不存在: {RESULTS_PATH}")
            return None
        
        # 檢查是否有Excel檔案
        results_dir = Path(RESULTS_PATH)
        excel_files = list(results_dir.glob("*.xlsx"))
        
        if not excel_files:
            print(f"❌ 在 {RESULTS_PATH} 目錄中找不到Excel結果檔案")
            print("請先執行資料提取功能生成結果檔案")
            return None
        
        # 統計有效檔案（排除'無提取'）
        valid_files = [f for f in excel_files if "無提取" not in f.name]
        excluded_files = [f for f in excel_files if "無提取" in f.name]
        
        print(f"📄 掃描到 {len(excel_files)} 個Excel檔案")
        if excluded_files:
            print(f"⊗ 將排除 {len(excluded_files)} 個'無提取'檔案:")
            for excluded_file in excluded_files[:3]:  # 只顯示前3個
                print(f"   - {excluded_file.name}")
            if len(excluded_files) > 3:
                print(f"   ... 還有 {len(excluded_files) - 3} 個")
        
        print(f"✅ 將處理 {len(valid_files)} 個有效檔案")
        for valid_file in valid_files[:5]:  # 顯示前5個
            print(f"   - {valid_file.name}")
        if len(valid_files) > 5:
            print(f"   ... 還有 {len(valid_files) - 5} 個檔案")
        
        if not valid_files:
            print("❌ 沒有有效的檔案可彙整（所有檔案都包含'無提取'）")
            return None
        
        # 執行彙整
        result_path = consolidate_esg_results(RESULTS_PATH)
        
        if result_path:
            print(f"✅ 彙整完成: {Path(result_path).name}")
            print(f"🏷️ 已完成公司名稱智能統一處理")
            return result_path
        else:
            print("❌ 彙整失敗")
            return None
            
    except ImportError as e:
        print(f"❌ 無法載入彙整模組: {e}")
        print("請確保consolidator.py文件存在")
        return None
    except Exception as e:
        print(f"❌ 彙整失敗: {e}")
        import traceback
        traceback.print_exc()
        return None

# =============================================================================
# 顯示和分析函數
# =============================================================================

def show_system_info():
    """顯示系統配置信息"""
    if not CONFIG_LOADED:
        print("❌ 配置未載入")
        return
    
    print("📋 平衡版系統配置信息")
    print("=" * 50)
    print(f"🤖 Gemini模型: {GEMINI_MODEL}")
    print(f"🧠 Embedding模型: {EMBEDDING_MODEL}")
    print(f"📚 向量資料庫: {VECTOR_DB_PATH}")
    print(f"📁 數據目錄: {DATA_PATH}")
    print(f"📊 結果目錄: {RESULTS_PATH}")
    print(f"🔢 文本塊大小: {CHUNK_SIZE}")
    print(f"🔍 搜索數量: {SEARCH_K}")
    print(f"📏 信心分數閾值: {CONFIDENCE_THRESHOLD}")
    print(f"📄 最大處理文檔數: {MAX_DOCS_PER_RUN}")
    print(f"🤖 LLM增強: {'啟用' if ENABLE_LLM_ENHANCEMENT else '停用'}")
    print(f"⚖️ 版本特色: 平衡準確度與覆蓋率")

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
        excel_files = list(results_dir.glob("ESG*.xlsx"))
        if not excel_files:
            print("❌ 沒有找到結果文件")
            return
        
        # 按修改時間排序
        excel_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        
        print("📊 最新結果文件")
        print("=" * 50)
        
        # 分類顯示
        consolidated_files = [f for f in excel_files if "彙整報告" in f.name]
        balanced_files = [f for f in excel_files if "平衡版" in f.name and "彙整報告" not in f.name]
        precision_files = [f for f in excel_files if "高精度" in f.name]
        other_files = [f for f in excel_files if all(keyword not in f.name for keyword in ["彙整報告", "平衡版", "高精度"])]
        
        if consolidated_files:
            print("\n📊 彙整報告:")
            for file in consolidated_files[:3]:
                file_time = datetime.fromtimestamp(file.stat().st_mtime)
                file_size = file.stat().st_size / 1024
                print(f"   📄 {file.name}")
                print(f"      🕒 {file_time.strftime('%Y-%m-%d %H:%M:%S')} | 📏 {file_size:.1f}KB")
        
        if balanced_files:
            print("\n⚖️ 平衡版結果:")
            for file in balanced_files[:3]:
                file_time = datetime.fromtimestamp(file.stat().st_mtime)
                file_size = file.stat().st_size / 1024
                print(f"   📄 {file.name}")
                print(f"      🕒 {file_time.strftime('%Y-%m-%d %H:%M:%S')} | 📏 {file_size:.1f}KB")
        
        if precision_files:
            print(f"\n🎯 高精度版結果:")
            for file in precision_files[:3]:
                file_time = datetime.fromtimestamp(file.stat().st_mtime)
                file_size = file.stat().st_size / 1024
                print(f"   📄 {file.name}")
                print(f"      🕒 {file_time.strftime('%Y-%m-%d %H:%M:%S')} | 📏 {file_size:.1f}KB")
        
        if other_files:
            print(f"\n📝 其他版本結果:")
            for file in other_files[:3]:
                file_time = datetime.fromtimestamp(file.stat().st_mtime)
                file_size = file.stat().st_size / 1024
                print(f"   📄 {file.name}")
                print(f"      🕒 {file_time.strftime('%Y-%m-%d %H:%M:%S')} | 📏 {file_size:.1f}KB")
        
        # 統計信息
        print(f"\n📈 統計摘要:")
        print(f"   總檔案數: {len(excel_files)}")
        print(f"   彙整報告: {len(consolidated_files)} 個")
        print(f"   平衡版結果: {len(balanced_files)} 個")
        print(f"   高精度版結果: {len(precision_files)} 個")
        print(f"   其他版本: {len(other_files)} 個")
            
    except Exception as e:
        print(f"❌ 查看結果失敗: {e}")

def compare_versions():
    """比較不同版本的結果"""
    if not CONFIG_LOADED:
        print("❌ 配置未載入")
        return
    
    try:
        import pandas as pd
        
        results_dir = Path(RESULTS_PATH)
        if not results_dir.exists():
            print("❌ 結果目錄不存在")
            return
        
        # 尋找不同版本的檔案
        balanced_files = list(results_dir.glob("*平衡版*.xlsx"))
        precision_files = list(results_dir.glob("*高精度*.xlsx"))
        consolidated_files = list(results_dir.glob("*彙整報告*.xlsx"))
        standard_files = [f for f in results_dir.glob("ESG提取結果_*.xlsx") 
                         if not any(keyword in f.name for keyword in ["平衡版", "高精度", "彙整報告"])]
        
        print("📊 版本比較分析")
        print("=" * 50)
        
        versions = [
            ("📊 彙整報告", consolidated_files),
            ("⚖️ 平衡版", balanced_files),
            ("🎯 高精度版", precision_files),
            ("📝 標準版", standard_files)
        ]
        
        for version_name, files in versions:
            if files:
                latest_file = max(files, key=lambda x: x.stat().st_mtime)
                try:
                    # 嘗試讀取不同的工作表
                    sheet_names_to_try = ['平衡版提取結果', '提取結果', 0]
                    df = None
                    
                    for sheet_name in sheet_names_to_try:
                        try:
                            df = pd.read_excel(latest_file, sheet_name=sheet_name)
                            break
                        except:
                            continue
                    
                    if df is not None:
                        # 跳過標題行計算數量
                        count = len(df) - 2 if len(df) > 2 else len(df)
                        print(f"\n{version_name}: {count} 個提取結果")
                    else:
                        print(f"\n{version_name}: 無法讀取內容")
                    
                    print(f"   📄 {latest_file.name}")
                    print(f"   📊 共 {len(files)} 個檔案")
                except Exception as e:
                    print(f"\n{version_name}: 讀取錯誤 - {e}")
            else:
                print(f"\n{version_name}: 無結果文件")
        
        print(f"\n💡 版本選擇建議:")
        print(f"   📊 彙整報告: 查看多公司多年度總覽")
        print(f"   🎯 高精度版: 要求高精確度，可接受少量遺漏")
        print(f"   ⚖️ 平衡版: 要求全面覆蓋，可接受適度人工篩選")
        print(f"   📝 標準版: 基礎功能，適合快速測試")
        
    except Exception as e:
        print(f"❌ 比較分析失敗: {e}")

def show_balanced_guide():
    """顯示平衡版使用說明"""
    print("\n⚖️ 平衡版提取系統 v2.5 使用說明")
    print("=" * 60)
    print("""
🎯 平衡版特色：
   • 確保基本覆蓋率 - 不會遺漏重要的再生塑膠相關內容
   • 適度過濾 - 只排除明確無關的內容（職業災害、賽事等）
   • 靈活匹配 - 降低門檻但保持質量
   • 多策略檢索 - 關鍵字+主題+數值三重檢索
   • 保留描述 - 即使沒有數值也保留重要描述

🆕 v2.5 新增功能：
   • 📊 智能彙整 - 自動整合多公司多年度結果
   • 📈 按年度分組 - 2024、2023、2022、2021年總覽
   • 🏢 按公司分組 - 每家公司的歷年資料總覽
   • 🏷️ 智能識別 - 自動統一同公司的不同命名方式
   • ✨ 美化格式 - 專業的Excel報表格式
   • 📋 統計摘要 - 完整的彙整統計信息
   • ⊗ 智能過濾 - 自動排除檔名包含'無提取'的檔案

🏷️ 公司名稱智能統一：
   • 自動識別：三芳、三芳化學、三芳化學工業 → 統一為最完整名稱
   • 靈活匹配：台灣塑膠工業、台灣塑膠工業股份 → 自動合併
   • 相似度分析：使用智能算法判斷是否為同一公司
   • 保留追踪：在摘要中顯示原始名稱到標準名稱的映射

⚖️ 設計理念：
   寧可多提取一些需要人工篩選的內容，
   也不要遺漏重要的再生塑膠相關數據
   
🎯 適用場景：
   ✅ 初次處理新的ESG報告
   ✅ 需要全面了解公司再生塑膠相關資訊
   ✅ 高精度版本提取結果過少時的備選方案
   ✅ 多公司多年度數據比較分析
   
📊 處理流程：
   1. 廣泛關鍵字檢索（包括中相關度詞彙）
   2. 多主題檢索（塑膠、回收、環保、永續）
   3. 數值導向檢索（億支、萬噸、產能等）
   4. 平衡門檻過濾（0.55信心分數）
   5. 強化去重和後處理
   6. 智能彙整多文件結果

🔧 關鍵參數：
   • 相關性門檻: 0.55（比高精度版稍寬鬆）
   • 排除策略: 僅排除明確無關內容
   • 檢索範圍: 擴大到中相關度關鍵字
   • 段落處理: 多種分割策略確保完整性
   • 去重邏輯: 精確去重+相似度去重

⚡ 快速開始：
   1. 將多個PDF放入data目錄
   2. 執行功能1進行平衡版提取
   3. 執行功能5生成彙整報告
   4. 查看results目錄中的彙整檔案
   
📈 預期效果：
   • 提取數量: 比高精度版多30-50%
   • 覆蓋率: 90%+的相關內容不遺漏
   • 精確度: 75-80%（需要適度人工篩選）
   • 彙整功能: 一鍵生成多維度總覽表
""")

# =============================================================================
# 用戶界面
# =============================================================================

def interactive_menu():
    """互動式主選單"""
    while True:
        print("\n" + "⚖️" * 20)
        print("🏢 ESG資料提取系統 v2.5 平衡版")
        print("在準確度與覆蓋率之間取得平衡，新增彙整功能")
        print("⚖️" * 20)
        print("1. ⚖️ 執行資料提取")
        print("2. 🔄 重新預處理PDF")
        print("3. 📊 查看最新結果")
        print("4. 📈 比較版本差異")
        print("5. 🔗 彙整多公司多年度結果（新功能！）")
        print("6. ⚙️  顯示系統信息")
        print("7. 💡 提取系統使用說明")
        print("8. 🚪 退出系統")
        
        choice = input("\n請選擇功能 (1-8): ").strip()
        
        if choice == "1":
            # 執行平衡版資料提取
            print("\n⚖️ 準備執行資料提取...")
            
            if not check_environment():
                print("❌ 環境檢查失敗，無法執行提取")
                continue
            
            # 找到所有PDF文件
            has_pdfs, pdf_files = find_pdf_files()
            if not has_pdfs:
                continue
            
            # 預處理（如果需要）
            docs_info = run_preprocessing(pdf_files)
            if not docs_info:
                print("❌ 預處理失敗，無法執行提取")
                continue
            
            # 執行平衡版提取
            results = run_balanced_extraction(docs_info)
            if results:
                print(f"\n🎉 提取完成！生成了 {len(results)} 個結果文件")
                for pdf_path, (extractions, summary, excel_path) in results.items():
                    print(f"⚖️ {summary.company_name} - {summary.report_year}: {len(extractions)} 個平衡結果")
                    print(f"   文件: {Path(excel_path).name}")
                
                # 詢問是否立即彙整
                consolidate_now = input("\n是否立即執行彙整功能？(y/n): ").strip().lower()
                if consolidate_now == 'y':
                    result_path = run_consolidation()
                    if result_path:
                        print(f"🔗 彙整完成: {Path(result_path).name}")
                
                # 詢問是否查看結果
                view_result = input("\n是否查看詳細結果？(y/n): ").strip().lower()
                if view_result == 'y':
                    show_latest_results()
            
        elif choice == "2":
            # 重新預處理PDF
            print("\n🔄 重新預處理PDF...")
            
            has_pdfs, pdf_files = find_pdf_files()
            if not has_pdfs:
                continue
            
            print(f"將處理 {len(pdf_files)} 個PDF文件：")
            for pdf_file in pdf_files:
                print(f"  - {pdf_file.name}")
            
            confirm = input("這將重新建立所有向量資料庫，確定繼續？(y/n): ").strip().lower()
            if confirm == 'y':
                docs_info = run_preprocessing(pdf_files, force=True)
                if docs_info:
                    print("✅ 預處理完成，現在可以執行平衡版提取")
            
        elif choice == "3":
            # 查看最新結果
            show_latest_results()
            
        elif choice == "4":
            # 比較版本差異
            compare_versions()
            
        elif choice == "5":
            # 🆕 新功能：彙整多公司多年度結果
            print("\n🔗 準備彙整多公司多年度結果...")
            print("⚠️ 注意：檔名包含'無提取'的檔案將自動排除")
            
            result_path = run_consolidation()
            if result_path:
                print(f"\n🎉 彙整功能執行完成！")
                print(f"📊 彙整檔案: {Path(result_path).name}")
                print(f"📁 存放位置: {RESULTS_PATH}")
                
                print(f"\n📋 彙整檔案包含以下工作表：")
                print(f"   📅 按年度分組: 2024年總覽、2023年總覽、2022年總覽、2021年總覽")
                print(f"   🏢 按公司分組: 每家公司的歷年資料總覽")
                print(f"   📊 彙整摘要: 統計信息和處理詳情")
                print(f"   ⊗ 已自動排除: 檔名包含'無提取'的檔案")
                
                # 詢問是否查看結果
                view_result = input("\n是否查看最新結果清單？(y/n): ").strip().lower()
                if view_result == 'y':
                    show_latest_results()
            else:
                print("❌ 彙整功能執行失敗")
                print("💡 請確保：")
                print("   1. results目錄中有Excel結果檔案")
                print("   2. 已執行過資料提取功能")
                print("   3. consolidator.py模組存在且正確")
                print("   4. 至少有一個檔案不包含'無提取'字樣")
            
        elif choice == "6":
            # 顯示系統信息
            show_system_info()
            
        elif choice == "7":
            # 平衡版使用說明
            show_balanced_guide()
            
        elif choice == "8":
            # 退出
            print("👋 感謝使用ESG平衡版資料提取系統！")
            break
            
        else:
            print("❌ 無效選擇，請輸入1-8之間的數字")

def command_line_mode():
    """命令行模式"""
    parser = argparse.ArgumentParser(
        description="ESG資料提取系統 v2.5 - 平衡版（新增彙整功能）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用範例:
  python main_balanced.py                      # 互動模式
  python main_balanced.py --auto               # 自動執行平衡版流程
  python main_balanced.py --preprocess         # 僅預處理
  python main_balanced.py --extract            # 僅平衡版提取
  python main_balanced.py --consolidate        # 僅彙整功能（新！）
  python main_balanced.py --force              # 強制重新預處理
  python main_balanced.py --results            # 查看結果
  python main_balanced.py --compare            # 比較版本
        """
    )
    
    parser.add_argument("--auto", action="store_true", help="自動執行平衡版完整流程")
    parser.add_argument("--preprocess", action="store_true", help="預處理所有PDF文件")
    parser.add_argument("--extract", action="store_true", help="執行平衡版資料提取")
    parser.add_argument("--consolidate", action="store_true", help="執行彙整功能（新！）")
    parser.add_argument("--force", action="store_true", help="強制重新預處理")
    parser.add_argument("--results", action="store_true", help="查看最新結果")
    parser.add_argument("--compare", action="store_true", help="比較版本差異")
    parser.add_argument("--max-docs", type=int, default=None, help="最大處理文檔數")
    
    args = parser.parse_args()
    
    # 根據參數執行對應功能
    if args.auto:
        # 自動執行平衡版完整流程
        print("⚖️ 自動平衡版執行模式")
        if not check_environment():
            sys.exit(1)
        
        has_pdfs, pdf_files = find_pdf_files()
        if not has_pdfs:
            sys.exit(1)
        
        print("執行預處理...")
        docs_info = run_preprocessing(pdf_files, force=args.force)
        if not docs_info:
            sys.exit(1)
        
        print("執行平衡版資料提取...")
        results = run_balanced_extraction(docs_info, args.max_docs)
        if results:
            print(f"✅ 平衡版提取完成！生成了 {len(results)} 個結果文件")
            for pdf_path, (extractions, summary, excel_path) in results.items():
                print(f"  ⚖️ {summary.company_name} - {summary.report_year}: {Path(excel_path).name}")
            
            # 自動執行彙整
            print("\n執行彙整功能...")
            result_path = run_consolidation()
            if result_path:
                print(f"🔗 彙整完成: {Path(result_path).name}")
        else:
            sys.exit(1)
            
    elif args.preprocess:
        has_pdfs, pdf_files = find_pdf_files()
        if not has_pdfs:
            sys.exit(1)
        
        docs_info = run_preprocessing(pdf_files, force=args.force)
        if docs_info:
            print("✅ 預處理完成")
        else:
            sys.exit(1)
            
    elif args.extract:
        has_pdfs, pdf_files = find_pdf_files()
        if not has_pdfs:
            sys.exit(1)
        
        docs_info = run_preprocessing(pdf_files, force=False)
        if not docs_info:
            print("❌ 需要先執行預處理")
            sys.exit(1)
        
        results = run_balanced_extraction(docs_info, args.max_docs)
        if not results:
            sys.exit(1)
    
    elif args.consolidate:
        # 🆕 新功能：僅執行彙整
        print("🔗 彙整功能模式")
        result_path = run_consolidation()
        if result_path:
            print(f"✅ 彙整完成: {Path(result_path).name}")
        else:
            sys.exit(1)
            
    elif args.results:
        show_latest_results()
        
    elif args.compare:
        compare_versions()
        
    else:
        # 沒有參數，顯示幫助
        parser.print_help()

def main():
    """主函數"""
    print("⚖️ ESG資料提取系統 v2.5 平衡版")
    print("在提取準確度和覆蓋率之間取得最佳平衡，新增彙整功能")
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