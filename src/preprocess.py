#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ESG報告書預處理模組 v1.0
處理PDF文件、建立向量資料庫、標準化檔名
"""

import os
import re
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent))
from config import *

# =============================================================================
# 台灣上市櫃公司代號映射表
# =============================================================================

COMPLETE_COMPANY_MAPPING = {
    # 13開頭塑膠工業
    "台塑": ("1301", "台塑"),
    "台灣塑膠": ("1301", "台塑"),
    "台灣塑膠工業": ("1301", "台塑"),
    "台塑工業": ("1301", "台塑"),
    "台塑集團": ("1301", "台塑"),
    
    "南亞": ("1303", "南亞"),
    "南亞塑膠": ("1303", "南亞"),
    "南亞塑膠工業": ("1303", "南亞"),
    "南亞公司": ("1303", "南亞"),
    
    "台聚": ("1304", "台聚"),
    "台灣聚合": ("1304", "台聚"),
    "台灣聚合化學": ("1304", "台聚"),
    
    "華夏": ("1305", "華夏"),
    "華夏海灣": ("1305", "華夏"),
    "華夏海灣塑膠": ("1305", "華夏"),
    
    "三芳": ("1307", "三芳"),
    "三芳化學": ("1307", "三芳"),
    "三芳化學工業": ("1307", "三芳"),
    "三芳化工": ("1307", "三芳"),
    
    "亞聚": ("1308", "亞聚"),
    "亞洲聚合": ("1308", "亞聚"),
    
    "台達化": ("1309", "台達化"),
    "台灣達化": ("1309", "台達化"),
    
    "台苯": ("1310", "台苯"),
    "台灣苯乙烯": ("1310", "台苯"),
    
    "國喬": ("1312", "國喬"),
    "國喬石化": ("1312", "國喬"),
    "國橋": ("1312", "國喬"),
    
    "國喬特": ("1312A", "國喬特"),
    
    "聯成": ("1313", "聯成"),
    "聯成化學": ("1313", "聯成"),
    "聯成化科": ("1313", "聯成"),
    
    "中石化": ("1314", "中石化"),
    "中國石油化學": ("1314", "中石化"),
    "鼎越": ("1314", "中石化"),
    "鼎越開發": ("1314", "中石化"),
    
    "達新": ("1315", "達新"),
    "達新工業": ("1315", "達新"),
    
    "上曜": ("1316", "上曜"),
    
    "東陽": ("1319", "東陽"),
    "東陽實業": ("1319", "東陽"),
    
    "大洋": ("1321", "大洋"),
    "大洋塑膠": ("1321", "大洋"),
    
    "永裕": ("1323", "永裕"),
    "永裕塑膠": ("1323", "永裕"),
    
    "地球": ("1324", "地球"),
    "地球化學": ("1324", "地球"),
    
    "恆大": ("1325", "恆大"),
    
    "台化": ("1326", "台化"),
    "台灣化學": ("1326", "台化"),
    "台灣化學纖維": ("1326", "台化"),
    
    "台翰": ("1336", "台翰"),
    
    "再生": ("1337", "再生-KY"),
    "再生-KY": ("1337", "再生-KY"),
    "再生資源": ("1337", "再生-KY"),
    
    "廣華": ("1338", "廣華-KY"),
    "廣華-KY": ("1338", "廣華-KY"),
    
    "昭輝": ("1339", "昭輝"),
    
    "勝悅": ("1340", "勝悅-KY"),
    "勝悅-KY": ("1340", "勝悅-KY"),
    
    "富林": ("1341", "富林-KY"),
    "富林-KY": ("1341", "富林-KY"),
    
    "八貫": ("1342", "八貫"),
    
    # 43開頭化學生技醫療
    "信立": ("4303", "信立"),
    "信立化學": ("4303", "信立"),
    "信立化": ("4303", "信立"),
    
    "勝昱": ("4304", "勝昱"),
    "勝昱精密": ("4304", "勝昱"),
    "勝昱精": ("4304", "勝昱"),
    
    "世坤": ("4305", "世坤"),
    "世坤科技": ("4305", "世坤"),
    "世坤科": ("4305", "世坤"),
    
    "炎洲": ("4306", "炎洲"),
    "炎洲科技": ("4306", "炎洲"),
    "炎洲科": ("4306", "炎洲"),
    
    # 99開頭其他
    "萬國通": ("9950", "萬國通"),
    "萬國通路": ("9950", "萬國通"),
}

# =============================================================================
# 文檔元數據提取器
# =============================================================================

class DocumentMetadataExtractor:
    """文檔元數據提取器"""
    
    def __init__(self):
        # 公司名稱匹配模式
        self.company_patterns = [
            # 高優先級：包含完整報告標題的模式
            r'([^,\n\d]{2,25}?)(?:股份)?有限公司\s*(202[0-9])\s*年(?:度)?(?:永續|ESG|企業社會責任)報告',
            r'([^,\n\d]{2,25}?)(?:股份)?有限公司.*?(202[0-9]).*?(?:永續|ESG|企業社會責任)報告',
            
            # 中優先級：公司名稱+報告類型
            r'([^,\n\d]{2,25}?)(?:股份)?有限公司.*?(?:永續|ESG|企業社會責任)報告',
            r'([^,\n\d]{2,25}?)公司.*?(?:202[0-9]).*?(?:永續|ESG|企業社會責任)報告',
            
            # 低優先級：僅公司名稱
            r'([\u4e00-\u9fff]{2,15})(?:股份)?有限公司',
            r'([\u4e00-\u9fff]{2,15})公司(?:[^\u4e00-\u9fff]|$)',
        ]
        
        # 年度匹配模式
        self.year_patterns = [
            # 高精確度：明確的報告年度表達
            r'(202[0-9])\s*年(?:度)?(?:永續|ESG|企業社會責任)報告(?:書)?',
            r'(?:永續|ESG|企業社會責任)報告(?:書)?.*?(202[0-9])\s*年(?:度)?',
            
            # 中精確度：報告標題中的年度
            r'(?:永續|ESG|企業社會責任)報告(?:書)?.*?(202[0-9])',
            r'(202[0-9]).*?(?:永續|ESG|企業社會責任)報告(?:書)?',
            
            # 包含"年報"的模式
            r'(202[0-9])\s*年(?:度)?年報',
            r'年報.*?(202[0-9])',
            
            # 其他可能的年度表達
            r'報告(?:書)?期間.*?(202[0-9])',
            r'財政年度.*?(202[0-9])',
            r'會計年度.*?(202[0-9])',
            
            # 最低優先級：任何四位數年份
            r'(202[0-9])',
        ]
    
    def extract_metadata(self, pdf_path: str) -> Dict[str, str]:
        """
        提取文檔元數據
        
        Returns:
            Dict包含 'company_name' 和 'report_year'
        """
        print(f"📋 提取文檔元數據: {Path(pdf_path).name}")
        
        try:
            loader = PyPDFLoader(pdf_path)
            pages = loader.load()
            
            # 檢查前8頁
            text_for_extraction = ""
            pages_to_check = min(8, len(pages))
            
            for page in pages[:pages_to_check]:
                text_for_extraction += page.page_content + "\n"
            
            # 先嘗試從文件名提取作為參考
            filename_metadata = self._extract_from_filename(Path(pdf_path).name)
            
            # 提取公司名稱和報告年度
            company_name = self._extract_company_name(text_for_extraction, filename_metadata.get('company_name', ''))
            report_year = self._extract_report_year(text_for_extraction, filename_metadata.get('report_year', ''))
            
            # 如果仍無法提取到有效信息，使用文件名作為備用
            if not company_name or company_name == "未知公司":
                company_name = filename_metadata.get('company_name', Path(pdf_path).stem)
            
            if not report_year or report_year == "未知年度":
                report_year = filename_metadata.get('report_year', '未知年度')
            
            result = {
                'company_name': company_name,
                'report_year': report_year
            }
            
            print(f"✅ 提取到：{company_name} - {report_year}")
            return result
            
        except Exception as e:
            print(f"⚠️ 元數據提取失敗: {e}")
            return {
                'company_name': Path(pdf_path).stem,
                'report_year': '未知年度'
            }
    
    def _extract_company_name(self, text: str, filename_hint: str = "") -> str:
        """提取公司名稱"""
        text_clean = re.sub(r'\s+', ' ', text[:3000])
        
        best_match = ""
        best_confidence = 0
        
        for i, pattern in enumerate(self.company_patterns):
            matches = re.findall(pattern, text_clean, re.IGNORECASE | re.MULTILINE)
            
            if matches:
                for match in matches:
                    if isinstance(match, tuple):
                        company_candidate = match[0].strip()
                    else:
                        company_candidate = match.strip()
                    
                    # 清理公司名稱
                    company_candidate = self._clean_company_name(company_candidate)
                    
                    if self._is_valid_company_name(company_candidate):
                        # 計算信心度
                        confidence = (len(self.company_patterns) - i) / len(self.company_patterns)
                        
                        # 如果與檔名提示匹配，加分
                        if filename_hint and filename_hint in company_candidate:
                            confidence += 0.2
                        
                        if confidence > best_confidence:
                            best_match = company_candidate
                            best_confidence = confidence
        
        return best_match if best_match else ""
    
    def _extract_report_year(self, text: str, filename_hint: str = "") -> str:
        """提取報告年度"""
        text_clean = re.sub(r'\s+', ' ', text[:3000])
        
        best_year = ""
        best_confidence = 0
        
        for i, pattern in enumerate(self.year_patterns):
            matches = re.findall(pattern, text_clean, re.IGNORECASE | re.MULTILINE)
            
            if matches:
                for match in matches:
                    year_candidate = match if isinstance(match, str) else match[0]
                    
                    # 驗證年份合理性
                    if self._is_valid_year(year_candidate):
                        # 計算信心度
                        confidence = (len(self.year_patterns) - i) / len(self.year_patterns)
                        
                        # 特別加分：如果是明確的報告書年度表達
                        if i < 4:
                            confidence += 0.3
                        
                        # 如果與檔名提示匹配，加分
                        if filename_hint and year_candidate == filename_hint:
                            confidence += 0.2
                        
                        # 優先選擇較新的年度
                        if confidence > best_confidence or (confidence == best_confidence and int(year_candidate) > int(best_year or "0")):
                            best_year = year_candidate
                            best_confidence = confidence
        
        return best_year if best_year else ""
    
    def _clean_company_name(self, raw_name: str) -> str:
        """清理公司名稱"""
        if not raw_name:
            return ""
        
        # 去除前後的空白、數字、特殊符號
        cleaned = re.sub(r'^[\s\d\-\.。，,\(\)（）【】]+', '', raw_name)
        cleaned = re.sub(r'[\s\-\.。，,\(\)（）【】]+$', '', cleaned)
        
        # 去除常見的無關詞彙
        noise_words = [
            '報告', '書', '永續', 'ESG', '企業社會責任', 
            '第', '章', '節', '頁', '附錄', '目錄'
        ]
        for word in noise_words:
            cleaned = re.sub(f'^{word}', '', cleaned)
            cleaned = re.sub(f'{word}$', '', cleaned)
        
        return cleaned.strip()
    
    def _is_valid_company_name(self, name: str) -> bool:
        """驗證公司名稱的有效性"""
        if not name or len(name) < 2 or len(name) > 25:
            return False
        
        # 排除明顯不是公司名稱的詞彙
        invalid_patterns = [
            r'^[0-9\.\-\s]+$',  # 純數字或符號
            r'^[a-zA-Z\s]+$',   # 純英文
            r'第.*?章|第.*?節|頁.*?碼',  # 章節頁碼
        ]
        
        for pattern in invalid_patterns:
            if re.match(pattern, name):
                return False
        
        return True
    
    def _is_valid_year(self, year: str) -> bool:
        """驗證年份的有效性"""
        try:
            year_int = int(year)
            return 2015 <= year_int <= 2030
        except ValueError:
            return False
    
    def _extract_from_filename(self, filename: str) -> Dict[str, str]:
        """從文件名提取元數據作為輔助信息"""
        result = {'company_name': '', 'report_year': ''}
        
        # 提取年份
        year_patterns = [
            r'(202[0-9])',
            r'(20[12][0-9])',
        ]
        
        for pattern in year_patterns:
            year_match = re.search(pattern, filename)
            if year_match:
                result['report_year'] = year_match.group(1)
                break
        
        # 提取公司名稱
        company_part = filename
        
        # 去除副檔名
        company_part = re.sub(r'\.pdf$', '', company_part, flags=re.IGNORECASE)
        
        # 去除年份
        company_part = re.sub(r'202[0-9]', '', company_part)
        company_part = re.sub(r'20[12][0-9]', '', company_part)
        
        # 去除常見關鍵詞
        keywords_to_remove = [
            'ESG', 'esg', '永續', '報告', '書', '企業社會責任', 
            '_', '-', '提取', '結果'
        ]
        for keyword in keywords_to_remove:
            company_part = re.sub(keyword, '', company_part, flags=re.IGNORECASE)
        
        # 清理剩餘的符號和空白
        company_part = re.sub(r'[_\-\s]+', ' ', company_part).strip()
        
        if company_part and len(company_part) >= 2:
            result['company_name'] = company_part
        
        return result

# =============================================================================
# 檔名標準化功能
# =============================================================================

def smart_extract_company_from_filename(filename: str) -> List[str]:
    """智能從檔名提取公司名稱候選"""
    candidates = []
    
    # 移除副檔名和常見後綴
    name = filename.replace('.pdf', '').replace('.PDF', '')
    name = re.sub(r'_esg報告書.*$', '', name, flags=re.IGNORECASE)
    name = re.sub(r'_2024.*$', '', name)
    name = re.sub(r'_2023.*$', '', name)
    name = re.sub(r'_2022.*$', '', name)
    
    # 策略1：檢查是否包含已知公司名稱
    for company_name in COMPLETE_COMPANY_MAPPING.keys():
        if company_name in filename:
            candidates.append(company_name)
    
    # 策略2：分割檔名並檢查每個部分
    separators = ['_', '-', ' ', '年', 'esg', 'ESG', '報告', '書']
    parts = [name]
    
    for sep in separators:
        new_parts = []
        for part in parts:
            new_parts.extend(part.split(sep))
        parts = new_parts
    
    # 清理和過濾部分
    clean_parts = []
    for part in parts:
        part = part.strip()
        if len(part) >= 2 and not part.isdigit() and not re.match(r'^202[0-9]$', part):
            clean_parts.append(part)
    
    # 檢查每個部分是否為已知公司
    for part in clean_parts:
        for company_name in COMPLETE_COMPANY_MAPPING.keys():
            if company_name in part or part in company_name:
                if len(company_name) >= 2:
                    candidates.append(company_name)
    
    # 策略3：檢查檔名中是否包含股票代號
    for company_name, (stock_code, standard_name) in COMPLETE_COMPANY_MAPPING.items():
        if stock_code and stock_code in filename:
            candidates.append(company_name)
    
    return list(set(candidates))  # 去重

def extract_year_from_filename(filename: str) -> str:
    """從檔名提取年度"""
    year_match = re.search(r'(202[0-9])', filename)
    return year_match.group(1) if year_match else ""

def standardize_pdf_filenames(data_path: str = None) -> Dict[str, str]:
    """
    標準化PDF檔名為：代號_公司名_年度_esg報告書.pdf
    
    Returns:
        Dict: {原檔名: 新檔名} 的映射
    """
    if data_path is None:
        data_path = DATA_PATH
    
    data_dir = Path(data_path)
    pdf_files = list(data_dir.glob("*.pdf"))
    
    if not pdf_files:
        print(f"❌ 在 {data_path} 目錄中找不到PDF文件")
        return {}
    
    print(f"📁 開始標準化 {len(pdf_files)} 個PDF檔名...")
    print("🎯 使用智能檔名分析 + PDF內容提取雙重策略")
    print("📋 支援台灣上市櫃公司代號識別")
    print("=" * 60)
    
    metadata_extractor = DocumentMetadataExtractor()
    rename_mapping = {}
    
    for pdf_file in pdf_files:
        try:
            print(f"\n📄 處理: {pdf_file.name}")
            
            # 策略1：從檔名智能提取公司
            filename_candidates = smart_extract_company_from_filename(pdf_file.name)
            print(f"   📝 檔名分析候選: {filename_candidates}")
            
            # 策略2：從PDF內容提取
            try:
                metadata = metadata_extractor.extract_metadata(str(pdf_file))
                pdf_company = metadata['company_name']
                pdf_year = metadata['report_year']
                print(f"   📄 PDF提取: 公司={pdf_company}, 年度={pdf_year}")
            except Exception as e:
                print(f"   ⚠️ PDF提取失敗: {e}")
                pdf_company = "未知公司"
                pdf_year = ""
            
            # 策略3：從檔名提取年度
            filename_year = extract_year_from_filename(pdf_file.name)
            if filename_year:
                print(f"   📅 檔名提取年度: {filename_year}")
            
            # 決定最佳公司名稱和代號
            stock_code = ""
            standard_name = ""
            
            # 優先使用檔名候選（通常更準確）
            if filename_candidates:
                for candidate in filename_candidates:
                    if candidate in COMPLETE_COMPANY_MAPPING:
                        stock_code, standard_name = COMPLETE_COMPANY_MAPPING[candidate]
                        print(f"   ✅ 檔名匹配: {stock_code} {standard_name}")
                        break
            
            # 如果檔名匹配失敗，嘗試PDF提取的名稱
            if not stock_code and pdf_company != "未知公司":
                pdf_clean = pdf_company.strip()
                # 移除常見後綴
                for suffix in ["股份有限公司", "有限公司", "公司", "工業", "化學", "塑膠"]:
                    pdf_clean = pdf_clean.replace(suffix, "").strip()
                
                for company_name, (code, std_name) in COMPLETE_COMPANY_MAPPING.items():
                    if company_name in pdf_clean or pdf_clean in company_name:
                        stock_code = code
                        standard_name = std_name
                        print(f"   ✅ PDF匹配: {stock_code} {standard_name}")
                        break
            
            # 決定最終使用的年度
            final_year = filename_year or pdf_year
            if final_year == "未知年度":
                final_year = ""
            
            # 生成新檔名
            if stock_code and standard_name:
                if final_year:
                    new_filename = f"{stock_code}_{standard_name}_{final_year}_esg報告書.pdf"
                else:
                    new_filename = f"{stock_code}_{standard_name}_esg報告書.pdf"
                print(f"   🎯 生成標準檔名: {new_filename}")
            else:
                # 無法識別的情況，使用清理後的原檔名
                if filename_candidates:
                    clean_name = filename_candidates[0]
                elif pdf_company != "未知公司":
                    clean_name = re.sub(r'[^\w\s-]', '', pdf_company)
                    clean_name = re.sub(r'\s+', '', clean_name)
                else:
                    # 從原檔名提取合理名稱
                    base_name = pdf_file.stem
                    base_name = re.sub(r'_esg.*$', '', base_name, flags=re.IGNORECASE)
                    base_name = re.sub(r'_202[0-9].*$', '', base_name)
                    clean_name = base_name[:20]  # 限制長度
                
                if final_year:
                    new_filename = f"{clean_name}_{final_year}_esg報告書.pdf"
                else:
                    new_filename = f"{clean_name}_esg報告書.pdf"
                
                print(f"   ⚠️ 未識別公司，使用: {new_filename}")
            
            # 檢查檔名是否需要更改
            if pdf_file.name == new_filename:
                print(f"   ✓ 檔名已標準化")
                continue
            
            # 處理檔名衝突
            new_path = data_dir / new_filename
            counter = 1
            original_new_filename = new_filename
            
            while new_path.exists():
                name_without_ext = original_new_filename.replace('.pdf', '')
                new_filename = f"{name_without_ext}_{counter}.pdf"
                new_path = data_dir / new_filename
                counter += 1
            
            if counter > 1:
                print(f"   📝 避免檔名衝突，使用: {new_filename}")
            
            # 執行重命名
            pdf_file.rename(new_path)
            rename_mapping[pdf_file.name] = new_filename
            
            print(f"   ✅ 重命名成功: {new_filename}")
            
        except Exception as e:
            print(f"   ❌ 處理失敗: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\n🎉 檔名標準化完成！")
    print(f"✅ 成功重命名: {len(rename_mapping)} 個檔案")
    
    if rename_mapping:
        print(f"\n📋 重命名清單:")
        for old_name, new_name in rename_mapping.items():
            print(f"   • {old_name}")
            print(f"     → {new_name}")
    
    return rename_mapping

# =============================================================================
# 原有的預處理功能
# =============================================================================

def preprocess_documents(pdf_path: str, output_db_path: str = None, metadata: Dict[str, str] = None):
    """預處理PDF文檔並建立向量資料庫"""
    
    if output_db_path is None:
        output_db_path = VECTOR_DB_PATH
    
    print(f"開始處理PDF: {pdf_path}")
    
    # 1. 載入PDF
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"找不到PDF文件: {pdf_path}")
    
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    print(f"成功載入 {len(pages)} 頁")
    
    # 2. 為每個文檔添加元數據
    if metadata:
        for page in pages:
            page.metadata.update(metadata)
            page.metadata['source_file'] = Path(pdf_path).name
            
            # 添加頁碼信息（如果缺失）
            if 'page' not in page.metadata:
                page.metadata['page'] = pages.index(page) + 1
    
    # 3. 文本分割
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=180,
        separators=["\n\n", "\n", ".", "。", "，", " ", ""]
    )
    
    print("正在分割文本...")
    chunks = text_splitter.split_documents(pages)
    print(f"分割成 {len(chunks)} 個文本塊")
    
    # 4. 初始化embedding模型
    print(f"載入embedding模型: {EMBEDDING_MODEL}")
    embedding_model = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={'device': 'cpu'}
    )
    
    # 5. 建立向量資料庫
    print("建立向量資料庫...")
    db = FAISS.from_documents(chunks, embedding_model)
    
    # 6. 保存資料庫
    os.makedirs(os.path.dirname(output_db_path), exist_ok=True)
    db.save_local(output_db_path)
    print(f"向量資料庫已保存到: {output_db_path}")
    
    return db

def preprocess_multiple_documents(pdf_paths: List[str]) -> Dict[str, Dict]:
    """
    批量預處理多個PDF文檔
    
    Returns:
        Dict: {pdf_path: {'db_path': str, 'metadata': dict}}
    """
    print(f"🚀 開始批量預處理 {len(pdf_paths)} 個PDF文件")
    print("=" * 60)
    
    metadata_extractor = DocumentMetadataExtractor()
    results = {}
    
    for pdf_path in pdf_paths:
        try:
            print(f"\n📄 處理文件: {Path(pdf_path).name}")
            
            # 1. 元數據提取
            metadata = metadata_extractor.extract_metadata(pdf_path)
            
            # 2. 為每個文件創建獨立的向量資料庫
            pdf_name = Path(pdf_path).stem
            db_path = os.path.join(
                os.path.dirname(VECTOR_DB_PATH),
                f"esg_db_{pdf_name}"
            )
            
            # 3. 預處理文檔
            preprocess_documents(pdf_path, db_path, metadata)
            
            results[pdf_path] = {
                'db_path': db_path,
                'metadata': metadata,
                'pdf_name': pdf_name
            }
            
            print(f"✅ 完成: {metadata['company_name']} - {metadata['report_year']}")
            
        except Exception as e:
            print(f"❌ 處理失敗 {Path(pdf_path).name}: {e}")
            continue
    
    print(f"\n🎉 批量預處理完成！成功處理 {len(results)}/{len(pdf_paths)} 個文件")
    
    # 顯示提取結果摘要
    print(f"\n📋 提取摘要:")
    companies_years = {}
    for pdf_path, result in results.items():
        metadata = result['metadata']
        company = metadata['company_name']
        year = metadata['report_year']
        
        if company not in companies_years:
            companies_years[company] = []
        companies_years[company].append(year)
    
    for company, years in companies_years.items():
        years_str = ', '.join(sorted(set(years), reverse=True))
        print(f"   🏢 {company}: {years_str}")
    
    return results

# =============================================================================
# 測試和診斷功能
# =============================================================================

def test_company_mapping():
    """測試公司映射表"""
    target_companies = ["信立", "勝昱", "世坤", "炎洲", "萬國通", "南亞", "台塑", "三芳"]
    
    print("🧪 測試公司映射表")
    print("=" * 40)
    
    for company in target_companies:
        if company in COMPLETE_COMPANY_MAPPING:
            code, name = COMPLETE_COMPANY_MAPPING[company]
            print(f"✅ {company} → {code}_{name}_2024_esg報告書.pdf")
        else:
            print(f"❌ {company} → 未找到映射")
    
    print(f"\n📊 映射表統計:")
    print(f"   總公司數: {len(COMPLETE_COMPANY_MAPPING)}")
    print(f"   13開頭塑膠工業: {len([k for k, v in COMPLETE_COMPANY_MAPPING.items() if v[0].startswith('13')])}")
    print(f"   43開頭化學生技: {len([k for k, v in COMPLETE_COMPANY_MAPPING.items() if v[0].startswith('43')])}")
    print(f"   其他公司: {len([k for k, v in COMPLETE_COMPANY_MAPPING.items() if not v[0].startswith(('13', '43'))])}")

def diagnose_filename_issues(data_path: str = None):
    """診斷檔名問題"""
    if data_path is None:
        data_path = DATA_PATH
    
    data_dir = Path(data_path)
    pdf_files = list(data_dir.glob("*.pdf"))
    
    if not pdf_files:
        print(f"❌ 在 {data_path} 目錄中找不到PDF文件")
        return
    
    print("🔍 診斷PDF檔名問題")
    print("=" * 50)
    
    for pdf_file in pdf_files:
        print(f"\n📄 診斷: {pdf_file.name}")
        
        # 檔名分析
        candidates = smart_extract_company_from_filename(pdf_file.name)
        year = extract_year_from_filename(pdf_file.name)
        
        print(f"   📝 檔名候選公司: {candidates}")
        print(f"   📅 檔名提取年度: {year}")
        
        # 檢查映射
        matched = False
        for candidate in candidates:
            if candidate in COMPLETE_COMPANY_MAPPING:
                code, name = COMPLETE_COMPANY_MAPPING[candidate]
                print(f"   ✅ 匹配: {candidate} → {code} {name}")
                matched = True
                break
        
        if not matched:
            print(f"   ❌ 無法自動匹配")
            print(f"   💡 建議手動檢查公司名稱是否在映射表中")

def main():
    """主函數"""
    # 檢查data目錄中的PDF文件
    data_dir = Path(DATA_PATH)
    pdf_files = list(data_dir.glob("*.pdf"))
    
    if not pdf_files:
        print(f"錯誤: 在 {DATA_PATH} 目錄中找不到PDF文件")
        print("請將ESG報告PDF文件放入data目錄中")
        return
    
    print(f"找到 {len(pdf_files)} 個PDF文件:")
    for pdf_file in pdf_files:
        print(f"  - {pdf_file.name}")
    
    # 執行預處理
    if len(pdf_files) == 1:
        # 單文件模式
        pdf_path = pdf_files[0]
        print(f"\n單文件模式：處理 {pdf_path.name}")
        
        try:
            metadata_extractor = DocumentMetadataExtractor()
            metadata = metadata_extractor.extract_metadata(str(pdf_path))
            preprocess_documents(str(pdf_path), metadata=metadata)
            print("✅ 預處理完成！")
        except Exception as e:
            print(f"❌ 預處理失敗: {e}")
    else:
        # 多文件模式
        print(f"\n多文件模式：處理 {len(pdf_files)} 個文件")
        confirm = input("確定要批量處理所有文件嗎？(y/n): ").strip().lower()
        
        if confirm == 'y':
            try:
                results = preprocess_multiple_documents([str(f) for f in pdf_files])
                print(f"✅ 批量預處理完成！")
                
            except Exception as e:
                print(f"❌ 批量預處理失敗: {e}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "--test":
            test_company_mapping()
        elif sys.argv[1] == "--diagnose":
            diagnose_filename_issues()
        elif sys.argv[1] == "--standardize":
            result = standardize_pdf_filenames()
            if result:
                print(f"✅ 標準化完成，處理了 {len(result)} 個檔案")
        else:
            print("用法:")
            print("  python preprocess.py --test        # 測試公司映射表")
            print("  python preprocess.py --diagnose    # 診斷檔名問題")
            print("  python preprocess.py --standardize # 執行檔名標準化")
            print("  python preprocess.py              # 執行預處理")
    else:
        # 預設執行測試
        test_company_mapping()