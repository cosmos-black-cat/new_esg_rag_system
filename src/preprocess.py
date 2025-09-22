#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ESG報告書預處理模組 v1.0
處理PDF文件並建立向量資料庫
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
# 在 preprocess.py 文件末尾（def main(): 之前）添加以下代碼
# =============================================================================

# 台灣上市櫃塑膠工業公司代號映射表
# 更新的完整公司映射表
TAIWAN_COMPANIES_EXTENDED = {
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
    "台灣氯乙烯工業": ("1305", "華夏"),
    
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
    "國橋": ("1312", "國喬"),  # 常見寫法
    
    "國喬特": ("1312A", "國喬特"),
    
    "聯成": ("1313", "聯成"),
    "聯成化學": ("1313", "聯成"),
    "聯成化科": ("1313", "聯成"),
    "德國萊因技術監護顧問": ("1313", "聯成"),
    
    "中石化": ("1314", "中石化"),
    "中國石油化學": ("1314", "中石化"),
    
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
    "地球綜合": ("1324", "地球"),
    "地球化學": ("1324", "地球"),
    
    "恆大": ("1325", "恆大"),
    
    "台化": ("1326", "台化"),
    "台灣化學": ("1326", "台化"),
    "台灣化學纖維": ("1326", "台化"),
    
    "台翰": ("1336", "台翰"),
    
    "再生": ("1337", "再生-KY"),
    "再生-KY": ("1337", "再生-KY"),
    "亞洲塑膠再生": ("1337", "再生-KY"),
    
    "廣華": ("1338", "廣華-KY"),
    "廣華-KY": ("1338", "廣華-KY"),
    
    "昭輝": ("1339", "昭輝"),
    
    "勝悅": ("1340", "勝悅-KY"),
    "勝悅新材料": ("1340", "勝悅-KY"),
    "勝悅-KY": ("1340", "勝悅-KY"),
    
    "富林": ("1341", "富林-KY"),
    "富林-KY": ("1341", "富林-KY"),
    
    "八貫": ("1342", "八貫"),
    
    # 43開頭化學生技醫療
    "信立": ("4303", "信立"),
    "信立化學": ("4303", "信立"),
    
    "勝昱": ("4304", "勝昱"),
    "勝昱精密": ("4304", "勝昱"),
    
    "世堃": ("4305", "世坤"),
    "世堃塑膠": ("4305", "世坤"),
    
    "炎洲": ("4306", "炎洲"),  # 更正代號
    "炎洲股份": ("4306", "炎洲"),  # 更正代號
    "炎洲科技": ("4306", "炎洲"),
    
    # 99開頭其他
    "萬國通": ("9950", "萬國通"),
    "萬國通路": ("9950", "萬國通"),
    
    # 添加常見的簡稱和變體
    "信立化": ("4303", "信立"),
    "勝昱精": ("4304", "勝昱"),
    "世坤科": ("4305", "世坤"),
    "炎洲科": ("4306", "炎洲"),
    "萬國通路股份有限公司": ("9950", "萬國通"),
}

def extract_company_from_filename(filename: str) -> str:
    """從檔名中提取公司名稱"""
    # 移除副檔名
    name_without_ext = filename.replace('.pdf', '').replace('.PDF', '')
    
    # 常見的分隔符
    separators = ['_', '-', ' ', '年', '2024', '2023', '2022', 'esg', 'ESG', '報告', '書']
    
    # 嘗試不同的分割方式
    possible_names = [name_without_ext]
    
    for sep in separators:
        parts = name_without_ext.split(sep)
        for part in parts:
            clean_part = part.strip()
            if len(clean_part) >= 2 and not clean_part.isdigit():
                possible_names.append(clean_part)
    
    # 返回最長的可能名稱
    valid_names = [name for name in possible_names if len(name) >= 2 and not name.isdigit()]
    return max(valid_names, key=len) if valid_names else ""

def enhanced_find_company_code_and_name(extracted_company_name: str, filename: str = "") -> Tuple[str, str, str]:
    """
    增強版公司代號和名稱查找
    
    Returns:
        Tuple: (股票代號, 標準公司名, 來源)
    """
    
    def try_match_company(name: str) -> Tuple[str, str]:
        if not name:
            return "", ""
        
        # 清理名稱
        clean_name = name.strip()
        
        # 移除常見後綴進行匹配
        suffixes_to_remove = [
            "股份有限公司", "有限公司", "股份", "公司",
            "工業股份有限公司", "工業有限公司", "工業股份", "工業",
            "化學工業股份有限公司", "化學工業有限公司", "化學工業股份", "化學工業", "化學",
            "塑膠工業股份有限公司", "塑膠工業有限公司", "塑膠工業股份", "塑膠工業", "塑膠",
        ]
        
        # 生成名稱變體
        name_variants = [clean_name]
        
        for suffix in suffixes_to_remove:
            if clean_name.endswith(suffix):
                core_name = clean_name[:-len(suffix)].strip()
                if core_name:
                    name_variants.append(core_name)
        
        # 精確匹配
        for variant in name_variants:
            if variant in TAIWAN_COMPANIES_EXTENDED:
                code, standard_name = TAIWAN_COMPANIES_EXTENDED[variant]
                return code, standard_name
        
        # 模糊匹配
        for variant in name_variants:
            for known_name, (code, standard_name) in TAIWAN_COMPANIES_EXTENDED.items():
                if len(known_name) >= 2:
                    # 檢查包含關係
                    if (known_name in variant or variant in known_name) and len(known_name) >= 3:
                        return code, standard_name
                    # 檢查開頭匹配
                    if variant.startswith(known_name) or known_name.startswith(variant):
                        if len(known_name) >= 2:
                            return code, standard_name
        
        return "", ""
    
    # 策略1：使用PDF提取的公司名稱
    if extracted_company_name and extracted_company_name != "未知公司":
        code, name = try_match_company(extracted_company_name)
        if code and name:
            return code, name, "PDF內容"
    
    # 策略2：使用檔名提取的公司名稱
    if filename:
        filename_company = extract_company_from_filename(filename)
        if filename_company:
            code, name = try_match_company(filename_company)
            if code and name:
                return code, name, "檔名分析"
    
    # 策略3：直接分析檔名中的關鍵字
    if filename:
        filename_lower = filename.lower()
        for known_name, (code, standard_name) in TAIWAN_COMPANIES_EXTENDED.items():
            if known_name.lower() in filename_lower and len(known_name) >= 2:
                return code, standard_name, "檔名關鍵字"
    
    return "", "", "未識別"

def standardize_pdf_filenames_enhanced(data_path: str = None) -> Dict[str, str]:
    """
    增強版PDF檔名標準化
    """
    if data_path is None:
        data_path = DATA_PATH
    
    data_dir = Path(data_path)
    pdf_files = list(data_dir.glob("*.pdf"))
    
    if not pdf_files:
        print(f"❌ 在 {data_path} 目錄中找不到PDF文件")
        return {}
    
    print(f"📁 開始標準化 {len(pdf_files)} 個PDF檔名...")
    print("📋 支援台灣上市櫃公司代號識別（增強版）")
    print("🔍 多策略匹配：PDF內容 + 檔名分析 + 關鍵字匹配")
    print("=" * 70)
    
    metadata_extractor = DocumentMetadataExtractor()
    rename_mapping = {}
    
    for pdf_file in pdf_files:
        try:
            print(f"📄 處理: {pdf_file.name}")
            
            # 提取元數據
            metadata = metadata_extractor.extract_metadata(str(pdf_file))
            extracted_company = metadata['company_name']
            report_year = metadata['report_year']
            
            print(f"   🏢 PDF提取公司名稱: {extracted_company}")
            print(f"   📅 PDF提取年度: {report_year}")
            
            # 從檔名提取公司名稱作為備選
            filename_company = extract_company_from_filename(pdf_file.name)
            print(f"   📝 檔名分析公司名稱: {filename_company}")
            
            # 增強版公司識別
            stock_code, standard_company_name, source = enhanced_find_company_code_and_name(
                extracted_company, pdf_file.name
            )
            
            # 決定使用的公司名稱和前綴
            if stock_code and standard_company_name:
                print(f"   ✅ 識別為: {stock_code} {standard_company_name} (來源: {source})")
                company_for_filename = standard_company_name
                prefix = stock_code
            else:
                print(f"   ⚠️ 未識別為已知公司，使用原名稱")
                # 選擇最佳的公司名稱
                if filename_company and len(filename_company) > len(extracted_company):
                    company_for_filename = filename_company
                    print(f"   📝 使用檔名提取的名稱: {filename_company}")
                else:
                    safe_company_name = re.sub(r'[^\w\s-]', '', extracted_company)
                    safe_company_name = re.sub(r'\s+', '', safe_company_name)
                    company_for_filename = safe_company_name
                    print(f"   📄 使用PDF提取的名稱: {safe_company_name}")
                prefix = ""
            
            # 提取年度（優先使用PDF提取的，其次從檔名提取）
            final_year = report_year
            if not final_year or final_year == "未知年度":
                # 從檔名提取年度
                year_match = re.search(r'(202[0-9])', pdf_file.name)
                if year_match:
                    final_year = year_match.group(1)
                    print(f"   📅 從檔名提取年度: {final_year}")
            
            # 生成新檔名
            if prefix:
                if final_year and final_year != "未知年度":
                    new_filename = f"{prefix}_{company_for_filename}_{final_year}_esg報告書.pdf"
                else:
                    new_filename = f"{prefix}_{company_for_filename}_esg報告書.pdf"
            else:
                if final_year and final_year != "未知年度":
                    new_filename = f"{company_for_filename}_{final_year}_esg報告書.pdf"
                else:
                    new_filename = f"{company_for_filename}_esg報告書.pdf"
            
            # 檢查是否需要重命名
            if pdf_file.name == new_filename:
                print(f"   ✓ 檔名已標準化: {new_filename}")
                continue
            
            # 檢查新檔名是否已存在
            new_path = data_dir / new_filename
            counter = 1
            original_new_filename = new_filename
            
            while new_path.exists():
                print(f"   ⚠️ 目標檔名已存在: {new_filename}")
                name_without_ext = original_new_filename.replace('.pdf', '')
                new_filename = f"{name_without_ext}_{counter}.pdf"
                new_path = data_dir / new_filename
                counter += 1
            
            if counter > 1:
                print(f"   📝 使用替代檔名: {new_filename}")
            
            # 執行重命名
            pdf_file.rename(new_path)
            rename_mapping[pdf_file.name] = new_filename
            
            print(f"   ✅ 重命名成功:")
            print(f"      {pdf_file.name}")
            print(f"      → {new_filename}")
            
        except Exception as e:
            print(f"   ❌ 處理失敗: {e}")
            continue
        
        print()  # 空行分隔
    
    print(f"🎉 檔名標準化完成！")
    print(f"✅ 成功重命名: {len(rename_mapping)} 個檔案")
    
    if rename_mapping:
        print(f"\n📋 重命名清單:")
        for old_name, new_name in rename_mapping.items():
            print(f"   • {old_name}")
            print(f"     → {new_name}")
    
    return rename_mapping

# 為了保持向後兼容，保留原函數名
def standardize_pdf_filenames(data_path: str = None) -> Dict[str, str]:
    """向後兼容的函數名"""
    return standardize_pdf_filenames_enhanced(data_path)

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
    main()