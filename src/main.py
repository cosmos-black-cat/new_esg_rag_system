#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ESG報告書提取器 - 主程式 v2.0 增強版
支持新關鍵字配置、提高準確度、Word文檔輸出
"""

import os
import sys
import argparse
import shutil
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List, Tuple

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
# ESG報告書標準化命名功能（保持不變）
# =============================================================================

class ESGFileNormalizer:
    """ESG報告書檔案名稱標準化處理器"""
    
    def __init__(self):
        self.standard_format = "{company_name}_{report_year}_ESG報告書.pdf"
        self.backup_suffix = "_backup"
        
    def scan_pdf_files(self) -> List[Path]:
        """掃描所有PDF文件"""
        if not CONFIG_LOADED:
            return []
            
        data_dir = Path(DATA_PATH)
        if not data_dir.exists():
            print(f"❌ 數據目錄不存在: {DATA_PATH}")
            return []
            
        pdf_files = list(data_dir.glob("*.pdf"))
        return pdf_files
    
    def analyze_filename(self, pdf_path: Path) -> Dict[str, str]:
        """分析檔案名稱，提取公司和年度信息"""
        try:
            from preprocess import DocumentMetadataExtractor
            
            # 使用現有的元數據提取器
            extractor = DocumentMetadataExtractor()
            metadata = extractor.extract_metadata(str(pdf_path))
            
            # 從檔名也嘗試提取信息作為備用
            filename_metadata = extractor._extract_from_filename(pdf_path.name)
            
            # 選擇最佳結果
            company_name = metadata.get('company_name', '')
            report_year = metadata.get('report_year', '')
            
            # 如果從內容提取失敗，使用檔名提取的結果
            if not company_name or company_name == "未知公司":
                company_name = filename_metadata.get('company_name', pdf_path.stem)
            
            if not report_year or report_year == "未知年度":
                report_year = filename_metadata.get('report_year', '未知年度')
            
            # 清理公司名稱（移除不適合檔名的字符）
            company_name = self._clean_filename_part(company_name)
            
            return {
                'original_name': pdf_path.name,
                'company_name': company_name,
                'report_year': report_year,
                'confidence': 'high' if metadata.get('company_name') != "未知公司" else 'medium'
            }
            
        except Exception as e:
            print(f"⚠️ 分析檔案失敗 {pdf_path.name}: {e}")
            return {
                'original_name': pdf_path.name,
                'company_name': pdf_path.stem,
                'report_year': '未知年度',
                'confidence': 'low'
            }
    
    def _clean_filename_part(self, text: str) -> str:
        """清理檔名部分，移除不適合檔名的字符"""
        if not text:
            return "未知"
        
        # 移除或替換不適合檔名的字符
        import re
        
        # 替換常見的問題字符
        replacements = {
            '/': '_',
            '\\': '_',
            ':': '_',
            '*': '_',
            '?': '_',
            '"': '_',
            '<': '_',
            '>': '_',
            '|': '_',
            '\n': '_',
            '\r': '_',
            '\t': '_'
        }
        
        cleaned = text
        for old, new in replacements.items():
            cleaned = cleaned.replace(old, new)
        
        # 移除多餘空白和下劃線
        cleaned = re.sub(r'[_\s]+', '_', cleaned)
        cleaned = cleaned.strip('_')
        
        # 限制長度
        if len(cleaned) > 30:
            cleaned = cleaned[:30]
        
        return cleaned if cleaned else "未知"
    
    def generate_standard_name(self, analysis: Dict[str, str]) -> str:
        """生成標準化檔名"""
        company = analysis['company_name']
        year = analysis['report_year']
        
        # 確保年度格式正確
        if year and year != "未知年度":
            # 只保留數字
            import re
            year_match = re.search(r'(20[12][0-9])', year)
            if year_match:
                year = year_match.group(1)
        
        return self.standard_format.format(
            company_name=company,
            report_year=year
        )
    
    def preview_renaming(self, pdf_files: List[Path]) -> List[Dict]:
        """預覽重命名計劃"""
        print("🔍 分析PDF檔案名稱...")
        
        renaming_plan = []
        
        for pdf_file in pdf_files:
            print(f"   分析: {pdf_file.name}")
            
            analysis = self.analyze_filename(pdf_file)
            new_name = self.generate_standard_name(analysis)
            
            # 檢查是否需要重命名
            needs_rename = pdf_file.name != new_name
            
            # 檢查新檔名是否會衝突
            new_path = pdf_file.parent / new_name
            has_conflict = new_path.exists() and new_path != pdf_file
            
            plan_item = {
                'original_path': pdf_file,
                'original_name': pdf_file.name,
                'new_name': new_name,
                'new_path': new_path,
                'needs_rename': needs_rename,
                'has_conflict': has_conflict,
                'analysis': analysis
            }
            
            renaming_plan.append(plan_item)
        
        return renaming_plan
    
    def execute_renaming(self, renaming_plan: List[Dict], create_backup: bool = True) -> bool:
        """執行重命名操作"""
        print("🔄 開始執行檔案重命名...")
        
        success_count = 0
        total_count = len([item for item in renaming_plan if item['needs_rename']])
        
        if total_count == 0:
            print("ℹ️  所有檔案名稱已符合標準，無需重命名")
            return True
        
        # 創建備份目錄（如果需要）
        backup_dir = None
        if create_backup:
            backup_dir = Path(DATA_PATH) / f"備份_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            backup_dir.mkdir(exist_ok=True)
            print(f"📁 創建備份目錄: {backup_dir.name}")
        
        for item in renaming_plan:
            if not item['needs_rename']:
                continue
                
            try:
                original_path = item['original_path']
                new_path = item['new_path']
                
                # 處理檔名衝突
                if item['has_conflict']:
                    print(f"⚠️ 檔名衝突: {item['new_name']}")
                    # 添加數字後綴
                    counter = 1
                    base_name = new_path.stem
                    while new_path.exists():
                        new_name_with_counter = f"{base_name}_{counter}.pdf"
                        new_path = new_path.parent / new_name_with_counter
                        counter += 1
                    
                    print(f"   解決衝突: 使用 {new_path.name}")
                    item['new_path'] = new_path
                    item['new_name'] = new_path.name
                
                # 創建備份
                if create_backup:
                    backup_path = backup_dir / original_path.name
                    shutil.copy2(original_path, backup_path)
                    print(f"   💾 備份: {original_path.name}")
                
                # 執行重命名
                original_path.rename(new_path)
                success_count += 1
                
                print(f"   ✅ 重命名: {original_path.name} → {new_path.name}")
                
            except Exception as e:
                print(f"   ❌ 重命名失敗 {item['original_name']}: {e}")
                continue
        
        print(f"\n📊 重命名完成: {success_count}/{total_count} 個檔案")
        
        if create_backup and backup_dir:
            print(f"📁 備份檔案已保存至: {backup_dir}")
        
        return success_count > 0

def run_filename_standardization() -> Optional[Dict[str, str]]:
    """執行PDF檔名標準化"""
    try:
        # 直接從 preprocess 模組導入標準化函數
        from preprocess import standardize_pdf_filenames
        
        print("\n📁 開始PDF檔名標準化...")
        print("🎯 使用智能檔名分析 + PDF內容提取雙重策略")
        print("📋 支援台灣上市櫃公司代號識別")
        
        if not CONFIG_LOADED:
            print("❌ 配置未載入")
            return None
        
        # 檢查數據目錄
        if not os.path.exists(DATA_PATH):
            print(f"❌ 數據目錄不存在: {DATA_PATH}")
            return None
        
        # 執行標準化（調用 preprocess.py 的函數）
        rename_mapping = standardize_pdf_filenames(DATA_PATH)
        
        if rename_mapping:
            print(f"✅ 檔名標準化完成，共重命名 {len(rename_mapping)} 個檔案")
            return rename_mapping
        else:
            print("ℹ️  所有檔案已符合標準格式，無需重命名")
            return {}
            
    except ImportError as e:
        print(f"❌ 無法載入標準化模組: {e}")
        print("💡 請確保 preprocess.py 文件存在且包含 standardize_pdf_filenames 函數")
        return None
    except Exception as e:
        print(f"❌ 檔名標準化失敗: {e}")
        import traceback
        traceback.print_exc()
        return None

# =============================================================================
# 系統檢查函數（保持不變）
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
# 核心功能函數 - 更新支持增強版提取器
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

def run_extraction(docs_info: Dict, max_docs: int = None) -> Optional[Dict]:
    """執行ESG數據提取 - 增強版"""
    try:
        # 使用增強版提取器
        from esg_extractor import EnhancedESGExtractor, DocumentInfo
        
        print("📊 初始化增強版ESG報告書提取器...")
        print("🔧 新功能：擴展關鍵字、提高準確度、Word文檔輸出")
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
                db_path=info['db_path']
            )
        
        print("📊 開始增強版ESG數據提取...")
        print(f"   最大處理文檔數: {max_docs}")
        print(f"   LLM增強: {'啟用' if ENABLE_LLM_ENHANCEMENT else '停用'}")
        print(f"   輸出格式: Excel + Word文檔")
        
        results = extractor.process_multiple_documents(document_infos, max_docs)
        
        return results
        
    except Exception as e:
        print(f"❌ ESG數據提取失敗: {e}")
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
# 顯示函數 - 更新支持Word文檔
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
        
        # 查找Excel和Word文件
        excel_files = list(results_dir.glob("*.xlsx"))
        word_files = list(results_dir.glob("*.docx"))
        
        if not excel_files and not word_files:
            print("❌ 沒有找到結果文件")
            return
        
        # 按修改時間排序
        excel_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        word_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        
        print("📊 最新結果文件")
        print("=" * 50)
        
        # 分類顯示Excel文件
        consolidated_files = [f for f in excel_files if "彙整報告" in f.name]
        extraction_files = [f for f in excel_files if "彙整報告" not in f.name]
        
        if consolidated_files:
            print("\n📊 彙整報告:")
            for file in consolidated_files[:3]:
                file_time = datetime.fromtimestamp(file.stat().st_mtime)
                file_size = file.stat().st_size / 1024
                print(f"   📄 {file.name}")
                print(f"      🕒 {file_time.strftime('%Y-%m-%d %H:%M:%S')} | 📏 {file_size:.1f}KB")
        
        if extraction_files:
            print("\n📊 提取結果 (Excel):")
            for file in extraction_files[:5]:
                file_time = datetime.fromtimestamp(file.stat().st_mtime)
                file_size = file.stat().st_size / 1024
                print(f"   📄 {file.name}")
                print(f"      🕒 {file_time.strftime('%Y-%m-%d %H:%M:%S')} | 📏 {file_size:.1f}KB")
        
        # 顯示Word文件
        if word_files:
            print("\n📝 提取統整 (Word):")
            for file in word_files[:5]:
                file_time = datetime.fromtimestamp(file.stat().st_mtime)
                file_size = file.stat().st_size / 1024
                print(f"   📄 {file.name}")
                print(f"      🕒 {file_time.strftime('%Y-%m-%d %H:%M:%S')} | 📏 {file_size:.1f}KB")
        
        # 統計信息
        print(f"\n📈 統計摘要:")
        print(f"   總Excel檔案: {len(excel_files)}")
        print(f"   總Word檔案: {len(word_files)}")
        print(f"   彙整報告: {len(consolidated_files)} 個")
        print(f"   提取結果: {len(extraction_files)} 個")
            
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
    print(f"📝 Word文檔輸出: ✅ 支持")
    print(f"🔧 提取器版本: v2.0 增強版")

def show_usage_guide():
    """顯示使用說明"""
    print("\n📚 ESG報告書提取器使用說明 v2.0")
    print("=" * 60)
    print("""
🎯 主要功能：
   • 自動提取ESG報告中的再生塑膠和永續材料相關數據
   • 支援批量處理多份報告
   • 智能識別關鍵數值和相關描述
   • 多公司多年度結果彙整分析
   • PDF檔案名稱標準化管理
   • 🆕 Word文檔統整報告輸出

🔧 v2.0 新功能：
   • 擴展關鍵字配置（材料循環率、再生能源使用率等）
   • 提高提取準確度（更嚴格的相關性檢查）
   • Word文檔輸出（格式化的提取統整報告）
   • 增強的排除規則（避免無關內容）

📋 處理流程：
   1. 將ESG報告PDF放入data目錄
   2. （推薦）執行功能2標準化檔案名稱
   3. 執行功能1進行數據提取
   4. 執行功能4生成彙整報告
   5. 查看results目錄中的結果檔案

🔧 核心特色：
   • 精確的關鍵字與數值關聯性分析
   • 智能排除無關內容（職業災害、賽事、訓練等）
   • 頁面級去重確保資料品質
   • 自動公司名稱和股票代號識別
   • 專業的Excel報表輸出
   • 🆕 格式化的Word文檔統整報告

📊 輸出內容：
   • 再生塑膠相關數值數據
   • 回收產能和使用量
   • 環保效益和減碳資料
   • 循環經濟相關指標
   • 🆕 材料循環率、再生能源使用率等新指標

📝 輸出格式：
   • Excel: 股票代號_公司簡稱_年度.xlsx
   • Word: 股票代號_公司簡稱_年度_提取統整.docx

⚡ 快速開始：
   1. 設置API Key（在.env檔案中）
   2. 安裝依賴：pip install -r requirements.txt
   3. 放入PDF檔案到data目錄
   4. 執行功能2標準化檔案名稱（建議）
   5. 執行功能1提取數據
   6. 執行功能4彙整結果
""")

# =============================================================================
# 更新後的用戶界面
# =============================================================================

def interactive_menu():
    """互動式主選單"""
    while True:
        print("\n" + "📊" * 20)
        print("🏢 ESG報告書提取器 v1.0")
        print("專業提取ESG報告中的再生塑膠相關數據")
        print("📊" * 20)
        print("1. 📊 執行ESG數據提取（主要功能）")
        print("2. 📁 標準化PDF檔名")  # 重命名功能
        print("3. 🔄 重新預處理PDF")
        print("4. 🔗 彙整多公司結果")
        print("5. 📋 查看最新結果")
        print("6. ⚙️  顯示系統信息")
        print("7. 💡 使用說明")
        print("8. 🚪 退出系統")
        
        choice = input("\n請選擇功能 (1-8): ").strip()
        
        if choice == "1":
            # 執行ESG數據提取 (保持原有邏輯)
            print("\n📊 準備執行ESG數據提取...")
            
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
                    print(f"📊 {summary.company_name} - {summary.report_year}: {len(extractions)} 個結果")
                    print(f"   文件: {Path(excel_path).name}")
                
                # 詢問是否立即彙整
                if len(results) > 1:
                    consolidate_now = input("\n是否立即執行彙整功能？(y/n): ").strip().lower()
                    if consolidate_now == 'y':
                        result_path = run_consolidation()
                        if result_path:
                            print(f"🔗 彙整完成: {Path(result_path).name}")
        
        elif choice == "2":
            # 📁 標準化PDF檔名 (新的重命名功能)
            print("\n📁 準備標準化PDF檔名...")
            
            rename_mapping = run_filename_standardization()
            if rename_mapping is not None:
                if rename_mapping:
                    print(f"\n🎉 檔名標準化完成！")
                    print(f"📁 重命名了 {len(rename_mapping)} 個檔案")
                    
                    # 詢問是否立即執行數據提取
                    extract_now = input("\n檔名已標準化，是否立即執行數據提取？(y/n): ").strip().lower()
                    if extract_now == 'y':
                        # 重新找到PDF文件（因為檔名已改變）
                        has_pdfs, pdf_files = find_pdf_files()
                        if has_pdfs:
                            docs_info = run_preprocessing(pdf_files)
                            if docs_info:
                                results = run_extraction(docs_info)
                                if results:
                                    print(f"🎉 提取完成！生成了 {len(results)} 個結果文件")
                else:
                    print("✅ 所有檔案檔名已符合標準")
        
        elif choice == "3":
            # 重新預處理PDF (原來的選項2)
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
            
        elif choice == "4":
            # 彙整多公司結果 (原來的選項3)
            print("\n🔗 準備彙整多公司結果...")
            
            result_path = run_consolidation()
            if result_path:
                print(f"\n🎉 彙整功能執行完成！")
                print(f"📊 彙整檔案: {Path(result_path).name}")
                print(f"📁 存放位置: {RESULTS_PATH}")
            else:
                print("❌ 彙整功能執行失敗")
                print("💡 請確保已執行過資料提取功能")
            
        elif choice == "5":
            # 查看最新結果 (原來的選項4)
            show_latest_results()
            
        elif choice == "6":
            # 顯示系統信息 (原來的選項5)
            show_system_info()
            
        elif choice == "7":
            # 使用說明 (原來的選項6)
            show_usage_guide()
            
        elif choice == "8":
            # 退出系統 (原來的選項7)
            print("👋 感謝使用ESG報告書提取器！")
            break
            
        else:
            print("❌ 無效選擇，請輸入1-8之間的數字")

def main():
    """主函數"""
    print("📊 ESG報告書提取器 v2.0 增強版")
    print("專業提取ESG報告中的再生塑膠和永續材料相關數據")
    print("🆕 新功能：擴展關鍵字、提高準確度、Word文檔輸出")
    print("=" * 70)
    
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