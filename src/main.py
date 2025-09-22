#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ESG報告書提取器 - 主程式 v2.0
支持增強關鍵字配置和Word文件輸出
"""

import os
import sys
import argparse
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict

# 添加當前目錄到路徑
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

# 配置載入
try:
    from config import (
        GOOGLE_API_KEY, DATA_PATH, RESULTS_PATH, 
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
        "結果目錄": RESULTS_PATH
    }
    
    for name, path in directories.items():
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
            print(f"✅ 創建{name}: {path}")
        else:
            print(f"✅ {name}: {path}")
    
    # 檢查Word處理依賴
    try:
        import docx
        print("✅ Word文檔處理依賴已安裝")
    except ImportError:
        print("⚠️ Word文檔處理依賴未安裝")
        print("請運行: pip install python-docx")
        return False
    
    return True

def find_pdf_files() -> tuple[bool, list]:
    """找到所有PDF文件"""
    if not CONFIG_LOADED:
        return False, []
    
    try:
        data_dir = Path(DATA_PATH)
        pdf_files = list(data_dir.glob("*.pdf"))
        
        if not pdf_files:
            print(f"❌ 在 {DATA_PATH} 目錄中找不到PDF文件")
            print("請將ESG報告PDF文件放入data目錄中")
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
    """執行預處理"""
    try:
        from preprocess import preprocess_multiple_documents, DocumentMetadataExtractor
        
        if pdf_files is None:
            has_pdfs, pdf_files = find_pdf_files()
            if not has_pdfs:
                return None
        
        # 檢查是否需要預處理
        if not force:
            from config import VECTOR_DB_PATH
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

def run_enhanced_extraction(docs_info: Dict, max_docs: int = None) -> Optional[Dict]:
    """執行增強版ESG數據提取"""
    try:
        from esg_extractor import EnhancedESGExtractor, DocumentInfo
        
        print("📊 初始化增強版ESG報告書提取器...")
        print("🆕 新功能:")
        print("   • 支持新的ESG關鍵字配置")
        print("   • 自動識別股票代號")
        print("   • 輸出Word文檔格式")
        print("   • 提升提取準確性")
        
        extractor = EnhancedESGExtractor(enable_llm=ENABLE_LLM_ENHANCEMENT)
        
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
                db_path=info['db_path'],
                stock_code=""  # 將在處理過程中自動提取
            )
        
        print("📊 開始增強版ESG數據提取...")
        print(f"   最大處理文檔數: {max_docs}")
        print(f"   LLM增強: {'啟用' if ENABLE_LLM_ENHANCEMENT else '停用'}")
        
        # 批量處理文檔
        results = {}
        
        for pdf_path, doc_info in document_infos.items():
            try:
                print(f"\n📄 處理: {doc_info.company_name} - {doc_info.report_year}")
                
                extractions, summary, excel_path, word_path = extractor.process_single_document(doc_info, max_docs)
                
                results[pdf_path] = (extractions, summary, excel_path, word_path)
                
                print(f"✅ 完成: 生成 {len(extractions)} 個結果")
                print(f"📊 Excel文件: {Path(excel_path).name}")
                if word_path:
                    print(f"📄 Word文件: {Path(word_path).name}")
                
            except Exception as e:
                print(f"❌ 處理失敗 {doc_info.company_name}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        return results
        
    except Exception as e:
        print(f"❌ 增強版ESG數據提取失敗: {e}")
        import traceback
        traceback.print_exc()
        return None

def run_consolidation() -> Optional[str]:
    """執行彙整功能"""
    try:
        from consolidator import consolidate_esg_results
        
        print("\n📊 開始彙整ESG結果...")
        print("⚠️ 注意：檔名包含'無提取'的檔案將被自動排除")
        
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
            print(f"⊗ 將排除 {len(excluded_files)} 個'無提取'檔案")
        
        print(f"✅ 將處理 {len(valid_files)} 個有效檔案")
        
        if not valid_files:
            print("❌ 沒有有效的檔案可彙整（所有檔案都包含'無提取'）")
            return None
        
        # 執行彙整
        result_path = consolidate_esg_results(RESULTS_PATH)
        
        if result_path:
            print(f"✅ 彙整完成: {Path(result_path).name}")
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
# 顯示函數
# =============================================================================

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
        
        # 查找Excel文件和Word文件
        excel_files = list(results_dir.glob("ESG*.xlsx"))
        word_files = list(results_dir.glob("*.docx"))
        
        if not excel_files and not word_files:
            print("❌ 沒有找到結果文件")
            return
        
        # 按修改時間排序
        all_files = excel_files + word_files
        all_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        
        print("📊 最新結果文件")
        print("=" * 50)
        
        # 分類顯示
        consolidated_files = [f for f in all_files if "彙整報告" in f.name]
        extraction_files = [f for f in all_files if "彙整報告" not in f.name]
        word_files_filtered = [f for f in extraction_files if f.suffix == '.docx']
        excel_files_filtered = [f for f in extraction_files if f.suffix == '.xlsx']
        
        if consolidated_files:
            print("\n📊 彙整報告:")
            for file in consolidated_files[:3]:
                file_time = datetime.fromtimestamp(file.stat().st_mtime)
                file_size = file.stat().st_size / 1024
                print(f"   📄 {file.name}")
                print(f"      🕒 {file_time.strftime('%Y-%m-%d %H:%M:%S')} | 📏 {file_size:.1f}KB")
        
        if word_files_filtered:
            print("\n📄 Word文檔:")
            for file in word_files_filtered[:5]:
                file_time = datetime.fromtimestamp(file.stat().st_mtime)
                file_size = file.stat().st_size / 1024
                print(f"   📄 {file.name}")
                print(f"      🕒 {file_time.strftime('%Y-%m-%d %H:%M:%S')} | 📏 {file_size:.1f}KB")
        
        if excel_files_filtered:
            print("\n📊 Excel結果:")
            for file in excel_files_filtered[:5]:
                file_time = datetime.fromtimestamp(file.stat().st_mtime)
                file_size = file.stat().st_size / 1024
                print(f"   📄 {file.name}")
                print(f"      🕒 {file_time.strftime('%Y-%m-%d %H:%M:%S')} | 📏 {file_size:.1f}KB")
        
        # 統計信息
        print(f"\n📈 統計摘要:")
        print(f"   總檔案數: {len(all_files)}")
        print(f"   彙整報告: {len(consolidated_files)} 個")
        print(f"   Excel結果: {len(excel_files_filtered)} 個")
        print(f"   Word文檔: {len(word_files_filtered)} 個")
            
    except Exception as e:
        print(f"❌ 查看結果失敗: {e}")

def show_system_info():
    """顯示系統配置信息"""
    if not CONFIG_LOADED:
        print("❌ 配置未載入")
        return
    
    from config import (
        GEMINI_MODEL, EMBEDDING_MODEL, VECTOR_DB_PATH,
        CHUNK_SIZE, SEARCH_K, CONFIDENCE_THRESHOLD
    )
    
    print("📋 ESG報告書提取器配置信息 v2.0")
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
    
    print(f"\n🆕 增強版新功能:")
    print(f"   • 新的關鍵字配置（材料循環率、再生能源使用率等）")
    print(f"   • 自動股票代號識別")
    print(f"   • Word文檔輸出（.docx格式）")
    print(f"   • 提升的提取準確性")

def show_usage_guide():
    """顯示使用說明"""
    print("\n📚 ESG報告書提取器使用說明 v2.0")
    print("=" * 60)
    print("""
🎯 主要功能：
   • 自動提取ESG報告中的再生材料和循環經濟相關數據
   • 支援新的關鍵字類型（比率類、數量類、技術類）
   • 自動識別公司股票代號
   • 批量處理多份報告
   • 智能識別關鍵數值和相關描述
   • 多公司多年度結果彙整分析

📋 處理流程：
   1. 將ESG報告PDF放入data目錄
   2. 執行功能1進行數據提取
   3. 執行功能3生成彙整報告
   4. 查看results目錄中的結果檔案

🆕 新增功能：
   • Word文檔輸出：格式為{股票代號}_{公司名稱}_{年度}_提取統整.docx
   • 新關鍵字支持：材料循環率、再生能源使用率、綠電憑證等
   • 自動股票代號識別：支援台股代號格式
   • 增強準確性：更精確的關鍵字-數值關聯性分析

🔧 核心特色：
   • 精確的關鍵字與數值關聯性分析
   • 智能排除無關內容（職業災害、賽事等）
   • 頁面級去重確保資料品質
   • 動態信心分數閾值調整
   • 自動公司名稱標準化
   • 雙格式輸出（Excel + Word）

📊 新增提取內容：
   • 材料循環率、材料可回收率
   • 再生能源使用率、綠電憑證數量
   • 再生材料使用量、碳排減量數據
   • 分選辨視技術、單一材料處理
   • 購電協議、太陽能電力相關數據

📄 Word文檔格式：
   頁碼：第X頁
   關鍵字：[關鍵字名稱]
   數值：[提取的數值和單位]
   信心分數：[0.000-1.000]
   整個段落內容：[完整段落內容]

⚡ 快速開始：
   1. 設置API Key（在.env檔案中）
   2. 安裝新依賴：pip install python-docx
   3. 放入PDF檔案到data目錄
   4. 執行功能1進行增強提取
   5. 執行功能3彙整結果
   6. 查看Excel和Word雙格式輸出
""")

# =============================================================================
# 用戶界面
# =============================================================================

def interactive_menu():
    """互動式主選單"""
    while True:
        print("\n" + "📊" * 20)
        print("🏢 ESG報告書提取器 v2.0 (增強版)")
        print("專業提取ESG報告中的再生材料和循環經濟數據")
        print("🆕 新增：股票代號識別 + Word文檔輸出")
        print("📊" * 20)
        print("1. 📊 執行增強版ESG數據提取（主要功能）")
        print("2. 🔄 重新預處理PDF")
        print("3. 🔗 彙整多公司結果")
        print("4. 📋 查看最新結果")
        print("5. ⚙️  顯示系統信息")
        print("6. 💡 使用說明")
        print("7. 🚪 退出系統")
        
        choice = input("\n請選擇功能 (1-7): ").strip()
        
        if choice == "1":
            # 執行增強版ESG數據提取
            print("\n📊 準備執行增強版ESG數據提取...")
            
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
            
            # 執行增強版提取
            results = run_enhanced_extraction(docs_info)
            if results:
                print(f"\n🎉 增強版提取完成！生成了 {len(results)} 個結果文件")
                
                excel_count = 0
                word_count = 0
                
                for pdf_path, (extractions, summary, excel_path, word_path) in results.items():
                    print(f"📊 {summary.company_name} ({summary.stock_code}) - {summary.report_year}: {len(extractions)} 個結果")
                    print(f"   📄 Excel: {Path(excel_path).name}")
                    if word_path:
                        print(f"   📄 Word: {Path(word_path).name}")
                        word_count += 1
                    excel_count += 1
                
                print(f"\n📈 總計輸出: {excel_count} 個Excel文件, {word_count} 個Word文件")
                
                # 詢問是否立即彙整
                if len(results) > 1:
                    consolidate_now = input("\n是否立即執行彙整功能？(y/n): ").strip().lower()
                    if consolidate_now == 'y':
                        result_path = run_consolidation()
                        if result_path:
                            print(f"🔗 彙整完成: {Path(result_path).name}")
            
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
                    print("✅ 預處理完成，現在可以執行數據提取")
            
        elif choice == "3":
            # 彙整多公司結果
            print("\n🔗 準備彙整多公司結果...")
            
            result_path = run_consolidation()
            if result_path:
                print(f"\n🎉 彙整功能執行完成！")
                print(f"📊 彙整檔案: {Path(result_path).name}")
                print(f"📁 存放位置: {RESULTS_PATH}")
            else:
                print("❌ 彙整功能執行失敗")
                print("💡 請確保已執行過資料提取功能")
            
        elif choice == "4":
            # 查看最新結果
            show_latest_results()
            
        elif choice == "5":
            # 顯示系統信息
            show_system_info()
            
        elif choice == "6":
            # 使用說明
            show_usage_guide()
            
        elif choice == "7":
            # 退出
            print("👋 感謝使用ESG報告書提取器 v2.0！")
            break
            
        else:
            print("❌ 無效選擇，請輸入1-7之間的數字")

def command_line_mode():
    """命令行模式"""
    parser = argparse.ArgumentParser(
        description="ESG報告書提取器 v2.0 - 增強版，支持新關鍵字和Word輸出",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用範例:
  python main.py                      # 互動模式
  python main.py --auto               # 自動執行完整流程
  python main.py --preprocess         # 僅預處理
  python main.py --extract            # 僅數據提取（增強版）
  python main.py --consolidate        # 僅彙整功能
  python main.py --force              # 強制重新預處理
  python main.py --results            # 查看結果
        """
    )
    
    parser.add_argument("--auto", action="store_true", help="自動執行完整流程")
    parser.add_argument("--preprocess", action="store_true", help="預處理所有PDF文件")
    parser.add_argument("--extract", action="store_true", help="執行增強版ESG數據提取")
    parser.add_argument("--consolidate", action="store_true", help="執行彙整功能")
    parser.add_argument("--force", action="store_true", help="強制重新預處理")
    parser.add_argument("--results", action="store_true", help="查看最新結果")
    parser.add_argument("--max-docs", type=int, default=None, help="最大處理文檔數")
    
    args = parser.parse_args()
    
    # 根據參數執行對應功能
    if args.auto:
        # 自動執行完整流程
        print("📊 增強版自動執行模式")
        if not check_environment():
            sys.exit(1)
        
        has_pdfs, pdf_files = find_pdf_files()
        if not has_pdfs:
            sys.exit(1)
        
        print("執行預處理...")
        docs_info = run_preprocessing(pdf_files, force=args.force)
        if not docs_info:
            sys.exit(1)
        
        print("執行增強版ESG數據提取...")
        results = run_enhanced_extraction(docs_info, args.max_docs)
        if results:
            excel_count = sum(1 for _ in results)
            word_count = sum(1 for _, (_, _, _, word_path) in results.items() if word_path)
            print(f"✅ 增強版提取完成！生成了 {excel_count} 個Excel文件和 {word_count} 個Word文件")
            
            # 自動執行彙整
            if len(results) > 1:
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
        
        results = run_enhanced_extraction(docs_info, args.max_docs)
        if not results:
            sys.exit(1)
    
    elif args.consolidate:
        # 僅執行彙整
        print("🔗 彙整功能模式")
        result_path = run_consolidation()
        if result_path:
            print(f"✅ 彙整完成: {Path(result_path).name}")
        else:
            sys.exit(1)
            
    elif args.results:
        show_latest_results()
        
    else:
        # 沒有參數，顯示幫助
        parser.print_help()

def main():
    """主函數"""
    print("📊 ESG報告書提取器 v2.0 (增強版)")
    print("專業提取ESG報告中的再生材料和循環經濟數據")
    print("🆕 新功能：新關鍵字配置 + 股票代號識別 + Word輸出")
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