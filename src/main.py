#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ESG資料提取系統 - 主程式 v2.2
修復：配置載入、多文件處理、過濾邏輯
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
        CHUNK_SIZE, SEARCH_K, CONFIDENCE_THRESHOLD
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

def run_extraction(docs_info: Dict, max_docs: int = 300) -> Optional[Dict]:
    """執行資料提取"""
    try:
        from esg_extractor import MultiFileESGExtractor, DocumentInfo
        
        print("🚀 初始化多文件ESG資料提取器...")
        extractor = MultiFileESGExtractor(enable_llm=True)
        
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
        
        print("🔍 開始資料提取...")
        results = extractor.process_multiple_documents(document_infos, max_docs)
        
        return results
        
    except Exception as e:
        print(f"❌ 資料提取失敗: {e}")
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
    
    print("📋 系統配置信息")
    print("=" * 50)
    print(f"🤖 Gemini模型: {GEMINI_MODEL}")
    print(f"🧠 Embedding模型: {EMBEDDING_MODEL}")
    print(f"📚 向量資料庫: {VECTOR_DB_PATH}")
    print(f"📁 數據目錄: {DATA_PATH}")
    print(f"📊 結果目錄: {RESULTS_PATH}")
    print(f"🔢 文本塊大小: {CHUNK_SIZE}")
    print(f"🔍 搜索數量: {SEARCH_K}")

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
        excel_files = list(results_dir.glob("ESG提取結果_*.xlsx"))
        if not excel_files:
            print("❌ 沒有找到結果文件")
            return
        
        # 按修改時間排序
        excel_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        
        print("📊 最新結果文件")
        print("=" * 50)
        
        for i, file in enumerate(excel_files[:5], 1):
            file_time = datetime.fromtimestamp(file.stat().st_mtime)
            file_size = file.stat().st_size / 1024
            
            print(f"\n{i}. {file.name}")
            print(f"   🕒 時間: {file_time.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"   📏 大小: {file_size:.1f} KB")
            
            # 嘗試讀取公司信息
            try:
                df = pd.read_excel(file, sheet_name='提取結果', nrows=1)
                if not df.empty:
                    first_row = df.iloc[0]
                    company_info = str(first_row.iloc[0]) if len(first_row) > 0 else ""
                    year_info = str(first_row.iloc[1]) if len(first_row) > 1 else ""
                    if "公司:" in company_info:
                        print(f"   🏢 {company_info}")
                    if "報告年度:" in year_info:
                        print(f"   📅 {year_info}")
            except:
                pass
            
    except Exception as e:
        print(f"❌ 查看結果失敗: {e}")

def show_usage_guide():
    """顯示使用說明"""
    print("\n💡 使用說明 v2.2")
    print("=" * 60)
    print("""
🆕 v2.2 更新：
   • 修復配置載入問題
   • 改進過濾邏輯，減少相關資訊被誤過濾
   • Excel第一行顯示公司名稱和報告年度
   • 增強多文件批量處理功能

📚 系統功能：
   專門提取ESG報告書中再生塑膠相關的數據
   
🎯 支援的關鍵字：
   • 再生塑膠、再生塑料、再生料、再生PP
   • 寶特瓶回收、循環經濟、廢料回收等
   
📋 基本流程：
   1. 將多個ESG報告書PDF放入 data/ 目錄
   2. 選擇功能1執行完整提取
   3. 每間公司各自生成獨立的Excel結果文件
   
📊 輸出結果：
   • 第一行顯示公司名稱和報告年度
   • 提取結果：清理後的主要數據
   • 關鍵字統計：各關鍵字的統計信息
   • 處理摘要：系統運行摘要
   
⚡ 快速開始：
   1. 放入多個PDF到data目錄
   2. 執行 python main_fixed.py --auto
   3. 查看results目錄中的多個結果文件
   
🔧 命令行選項：
   python main_fixed.py --auto              # 自動處理所有文件
   python main_fixed.py --preprocess        # 僅預處理
   python main_fixed.py --extract           # 僅提取（需先預處理）
   python main_fixed.py --force             # 強制重新預處理
""")

# =============================================================================
# 用戶界面
# =============================================================================

def interactive_menu():
    """互動式主選單"""
    while True:
        print("\n" + "🔷" * 20)
        print("🏢 ESG資料提取系統 v2.2")
        print("修復版：配置載入、過濾邏輯、多文件處理")
        print("🔷" * 20)
        print("1. 📊 執行完整資料提取（支援多文件）")
        print("2. 🔄 重新預處理PDF（支援多文件）")
        print("3. 📋 查看最新結果")
        print("4. ⚙️  顯示系統信息")
        print("5. 💡 使用說明")
        print("6. 🚪 退出系統")
        
        choice = input("\n請選擇功能 (1-6): ").strip()
        
        if choice == "1":
            # 執行完整資料提取
            print("\n🚀 準備執行資料提取...")
            
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
            
            # 執行提取
            results = run_extraction(docs_info)
            if results:
                print(f"\n🎉 提取完成！生成了 {len(results)} 個結果文件")
                for pdf_path, (extractions, summary, excel_path) in results.items():
                    print(f"📁 {summary.company_name} - {summary.report_year}: {len(extractions)} 個結果")
                    print(f"   文件: {Path(excel_path).name}")
                
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
                    print("✅ 預處理完成，現在可以執行資料提取")
            
        elif choice == "3":
            # 查看最新結果
            show_latest_results()
            
        elif choice == "4":
            # 顯示系統信息
            show_system_info()
            
        elif choice == "5":
            # 使用說明
            show_usage_guide()
            
        elif choice == "6":
            # 退出
            print("👋 感謝使用ESG資料提取系統！")
            break
            
        else:
            print("❌ 無效選擇，請輸入1-6之間的數字")

def command_line_mode():
    """命令行模式"""
    parser = argparse.ArgumentParser(
        description="ESG資料提取系統 v2.2 - 修復版",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用範例:
  python main_fixed.py                    # 互動模式
  python main_fixed.py --auto             # 自動執行完整流程（所有PDF）
  python main_fixed.py --preprocess       # 僅預處理所有PDF
  python main_fixed.py --extract          # 僅執行提取
  python main_fixed.py --force            # 強制重新預處理
  python main_fixed.py --results          # 查看最新結果
        """
    )
    
    parser.add_argument("--auto", action="store_true", help="自動執行完整流程（所有PDF文件）")
    parser.add_argument("--preprocess", action="store_true", help="預處理所有PDF文件")
    parser.add_argument("--extract", action="store_true", help="執行資料提取")
    parser.add_argument("--force", action="store_true", help="強制重新預處理")
    parser.add_argument("--results", action="store_true", help="查看最新結果")
    parser.add_argument("--max-docs", type=int, default=300, help="最大處理文檔數")
    
    args = parser.parse_args()
    
    # 根據參數執行對應功能
    if args.auto:
        # 自動執行完整流程
        print("🚀 自動執行模式（多文件）")
        if not check_environment():
            sys.exit(1)
        
        has_pdfs, pdf_files = find_pdf_files()
        if not has_pdfs:
            sys.exit(1)
        
        print("執行預處理...")
        docs_info = run_preprocessing(pdf_files, force=args.force)
        if not docs_info:
            sys.exit(1)
        
        print("執行資料提取...")
        results = run_extraction(docs_info, args.max_docs)
        if results:
            print(f"✅ 完成！生成了 {len(results)} 個結果文件")
            for pdf_path, (extractions, summary, excel_path) in results.items():
                print(f"  📁 {summary.company_name} - {summary.report_year}: {Path(excel_path).name}")
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
        # 需要先檢查是否已有預處理結果
        has_pdfs, pdf_files = find_pdf_files()
        if not has_pdfs:
            sys.exit(1)
        
        docs_info = run_preprocessing(pdf_files, force=False)
        if not docs_info:
            print("❌ 需要先執行預處理")
            sys.exit(1)
        
        results = run_extraction(docs_info, args.max_docs)
        if not results:
            sys.exit(1)
            
    elif args.results:
        show_latest_results()
        
    else:
        # 沒有參數，顯示幫助
        parser.print_help()

def main():
    """主函數"""
    print("🏢 ESG資料提取系統 v2.2")
    print("修復版：配置載入、過濾邏輯、多文件處理")
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