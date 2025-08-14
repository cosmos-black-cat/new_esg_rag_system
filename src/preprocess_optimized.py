#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
優化的預處理器 v2.3
專為高精度提取設計，特別優化表格數據處理
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
    """文檔元數據提取器，用於提取公司名稱和報告年度"""
    
    def __init__(self):
        # 公司名稱匹配模式
        self.company_patterns = [
            r'([^,\n]+?)(?:股份)?有限公司.*?(?:年|年度).*?(?:永續|ESG|企業社會責任)報告',
            r'([^,\n]+?)(?:股份)?有限公司.*?(?:永續|ESG|企業社會責任)報告',
            r'([^,\n\d]+?)公司.*?(?:永續|ESG|企業社會責任)報告',
            r'([\u4e00-\u9fff]{2,10})(?:股份)?有限公司'
        ]
        
        # 年度匹配模式
        self.year_patterns = [
            r'(20\d{2})\s*年(?:度)?.*?(?:永續|ESG|企業社會責任)報告',
            r'(?:永續|ESG|企業社會責任)報告.*?(20\d{2})',
            r'(20\d{2})\s*年(?:度)?報告',
            r'報告.*?期間.*?(20\d{2})',
            r'(20\d{2})',  # 最後備用：任何四位數年份
        ]
    
    def extract_metadata(self, pdf_path: str) -> Dict[str, str]:
        """
        從PDF文件中提取公司名稱和報告年度
        
        Returns:
            Dict包含 'company_name' 和 'report_year'
        """
        print(f"📋 提取文檔元數據: {Path(pdf_path).name}")
        
        try:
            loader = PyPDFLoader(pdf_path)
            pages = loader.load()
            
            # 主要從前幾頁提取信息
            text_for_extraction = ""
            for page in pages[:5]:  # 只檢查前5頁
                text_for_extraction += page.page_content + "\n"
            
            # 提取公司名稱
            company_name = self._extract_company_name(text_for_extraction)
            
            # 提取報告年度
            report_year = self._extract_report_year(text_for_extraction)
            
            # 如果無法從文檔中提取，嘗試從文件名提取
            if not company_name or not report_year:
                filename_metadata = self._extract_from_filename(Path(pdf_path).name)
                if not company_name:
                    company_name = filename_metadata.get('company_name', '未知公司')
                if not report_year:
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
    
    def _extract_company_name(self, text: str) -> str:
        """提取公司名稱"""
        text_clean = re.sub(r'\s+', ' ', text[:2000])  # 只檢查前2000字符
        
        for pattern in self.company_patterns:
            matches = re.findall(pattern, text_clean, re.IGNORECASE)
            if matches:
                company_name = matches[0].strip()
                # 清理公司名稱
                company_name = re.sub(r'^[\s\d\-\.]+', '', company_name)
                company_name = re.sub(r'[\s\-\.]+$', '', company_name)
                if len(company_name) >= 2 and len(company_name) <= 20:
                    return company_name
        
        return ""
    
    def _extract_report_year(self, text: str) -> str:
        """提取報告年度"""
        text_clean = re.sub(r'\s+', ' ', text[:2000])
        
        for pattern in self.year_patterns:
            matches = re.findall(pattern, text_clean)
            if matches:
                year = matches[0]
                # 驗證年份合理性
                if 2015 <= int(year) <= 2030:
                    return year
        
        return ""
    
    def _extract_from_filename(self, filename: str) -> Dict[str, str]:
        """從文件名提取元數據作為備用"""
        result = {'company_name': '', 'report_year': ''}
        
        # 提取年份
        year_match = re.search(r'(20\d{2})', filename)
        if year_match:
            result['report_year'] = year_match.group(1)
        
        # 簡單的公司名稱提取（去除年份、副檔名等）
        company_part = re.sub(r'(20\d{2}|ESG|永續|報告|\.pdf)', '', filename, flags=re.IGNORECASE)
        company_part = re.sub(r'[_\-\s]+', '', company_part).strip()
        if company_part:
            result['company_name'] = company_part
        
        return result

class OptimizedTextSplitter:
    """優化的文本分割器，專為表格數據設計"""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # 標準分割器
        self.standard_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ".", "。", "，", " ", ""]
        )
        
        # 表格專用分割器（更大的塊）
        self.table_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size * 2,  # 表格使用更大的塊
            chunk_overlap=chunk_overlap,
            separators=["\n\n\n", "\n\n", "。。", "\n", ""]
        )
    
    def split_documents_optimized(self, documents: List) -> List:
        """優化的文檔分割"""
        all_chunks = []
        
        for doc in tqdm(documents, desc="優化分割文檔"):
            content = doc.page_content
            
            # 檢測是否為表格內容
            if self._is_table_content(content):
                # 使用表格專用分割器
                chunks = self.table_splitter.split_documents([doc])
                # 為表格塊添加特殊標記
                for chunk in chunks:
                    chunk.metadata['content_type'] = 'table'
                    chunk.metadata['is_table'] = True
            else:
                # 使用標準分割器
                chunks = self.standard_splitter.split_documents([doc])
                for chunk in chunks:
                    chunk.metadata['content_type'] = 'standard'
                    chunk.metadata['is_table'] = False
            
            all_chunks.extend(chunks)
        
        print(f"📊 分割統計: 總共 {len(all_chunks)} 個文本塊")
        table_chunks = sum(1 for chunk in all_chunks if chunk.metadata.get('is_table', False))
        print(f"   - 表格塊: {table_chunks} 個")
        print(f"   - 標準塊: {len(all_chunks) - table_chunks} 個")
        
        return all_chunks
    
    def _is_table_content(self, content: str) -> bool:
        """檢測是否為表格內容"""
        # 表格特徵指標
        table_indicators = [
            # 數值排列
            re.search(r'\d+(?:,\d{3})*(?:\.\d+)?\s+\d+(?:,\d{3})*(?:\.\d+)?\s+\d+(?:,\d{3})*(?:\.\d+)?', content),
            
            # 年份序列
            re.search(r'20\d{2}\s+20\d{2}\s+20\d{2}', content),
            
            # 表格關鍵詞
            any(keyword in content.lower() for keyword in [
                '歷年', '年份', '回收數量', '碳排減少量', '效益',
                '回收量可繞行', '大安森林', '吸碳量'
            ]),
            
            # 多列數據模式
            len(re.findall(r'\d+(?:,\d{3})*(?:\.\d+)?', content)) >= 6,
            
            # 特定表格標記詞
            any(phrase in content for phrase in [
                '億支寶特瓶', '萬噸', '噸/年', '噸/月',
                '座大安森林公園', '可繞行地球'
            ])
        ]
        
        # 如果有2個以上指標匹配，認為是表格內容
        matched_indicators = sum(1 for indicator in table_indicators if indicator)
        return matched_indicators >= 2

def preprocess_documents_optimized(pdf_path: str, output_db_path: str = None, metadata: Dict[str, str] = None):
    """優化的文檔預處理"""
    
    if output_db_path is None:
        output_db_path = VECTOR_DB_PATH
    
    print(f"🔄 開始優化預處理PDF: {pdf_path}")
    
    # 1. 載入PDF
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"找不到PDF文件: {pdf_path}")
    
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    print(f"📄 成功載入 {len(pages)} 頁")
    
    # 2. 為每個文檔添加元數據
    if metadata:
        for page in pages:
            page.metadata.update(metadata)
            page.metadata['source_file'] = Path(pdf_path).name
    
    # 3. 優化文本分割
    print("🔧 正在進行優化分割...")
    text_splitter = OptimizedTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    
    chunks = text_splitter.split_documents_optimized(pages)
    print(f"✅ 優化分割完成，生成 {len(chunks)} 個智能文本塊")
    
    # 4. 初始化embedding模型
    print(f"🧠 載入embedding模型: {EMBEDDING_MODEL}")
    embedding_model = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={'device': 'cpu'}
    )
    
    # 5. 建立向量資料庫
    print("🔗 建立優化向量資料庫...")
    db = FAISS.from_documents(chunks, embedding_model)
    
    # 6. 保存資料庫
    os.makedirs(os.path.dirname(output_db_path), exist_ok=True)
    db.save_local(output_db_path)
    print(f"💾 向量資料庫已保存到: {output_db_path}")
    
    return db

def preprocess_multiple_documents_optimized(pdf_paths: List[str]) -> Dict[str, Dict]:
    """
    優化的批量預處理多個PDF文檔
    
    Returns:
        Dict: {pdf_path: {'db_path': str, 'metadata': dict}}
    """
    print(f"🚀 開始優化批量預處理 {len(pdf_paths)} 個PDF文件")
    print("=" * 60)
    
    metadata_extractor = DocumentMetadataExtractor()
    results = {}
    
    for pdf_path in pdf_paths:
        try:
            print(f"\n📄 優化處理文件: {Path(pdf_path).name}")
            
            # 1. 提取元數據
            metadata = metadata_extractor.extract_metadata(pdf_path)
            
            # 2. 為每個文件創建獨立的向量資料庫
            pdf_name = Path(pdf_path).stem
            db_path = os.path.join(
                os.path.dirname(VECTOR_DB_PATH),
                f"esg_db_optimized_{pdf_name}"
            )
            
            # 3. 優化預處理文檔
            preprocess_documents_optimized(pdf_path, db_path, metadata)
            
            results[pdf_path] = {
                'db_path': db_path,
                'metadata': metadata,
                'pdf_name': pdf_name
            }
            
            print(f"✅ 優化完成: {metadata['company_name']} - {metadata['report_year']}")
            
        except Exception as e:
            print(f"❌ 優化處理失敗 {Path(pdf_path).name}: {e}")
            continue
    
    print(f"\n🎉 優化批量預處理完成！成功處理 {len(results)}/{len(pdf_paths)} 個文件")
    return results

def test_table_detection():
    """測試表格檢測功能"""
    print("🧪 測試表格檢測功能")
    print("=" * 50)
    
    # 測試用例
    test_cases = [
        {
            "name": "寶特瓶回收表格",
            "content": """歷年回收數量與碳排減少量
年份 2021 2022 2023
回收數量 碳排減少量 回收數量 碳排減少量 回收數量 碳排減少量
效益
109,008 噸
87 億支寶特瓶 188,039 噸
88,140 噸
70 億支寶特瓶 152,042 噸
80,904 噸
64 億支寶特瓶 139,559 噸""",
            "expected": True
        },
        {
            "name": "標準文本段落",
            "content": """南亞公司秉持保護地球、永續發展的經營理念，自2007年開始即致力於回收、再生消費者使用後的寶特瓶等聚酯製品，全力發展環保永續產品。""",
            "expected": False
        },
        {
            "name": "數值密集段落",
            "content": """2023年回收寶特瓶64億支，減碳排放13.9萬噸/年，寶特瓶回收造粒後取代原生聚酯粒較原製程生產之碳排放量可減少72%。""",
            "expected": True
        }
    ]
    
    splitter = OptimizedTextSplitter()
    
    for i, case in enumerate(test_cases, 1):
        print(f"\n測試案例 {i}: {case['name']}")
        result = splitter._is_table_content(case['content'])
        status = "✅ 正確" if result == case['expected'] else "❌ 錯誤"
        print(f"預期: {case['expected']}, 實際: {result} - {status}")

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
    
    # 詢問處理模式
    if len(pdf_files) == 1:
        # 單文件模式
        pdf_path = pdf_files[0]
        print(f"\n單文件優化模式：處理 {pdf_path.name}")
        
        try:
            metadata_extractor = DocumentMetadataExtractor()
            metadata = metadata_extractor.extract_metadata(str(pdf_path))
            preprocess_documents_optimized(str(pdf_path), metadata=metadata)
            print("✅ 優化預處理完成！")
        except Exception as e:
            print(f"❌ 優化預處理失敗: {e}")
    else:
        # 多文件模式
        print(f"\n多文件優化模式：處理 {len(pdf_files)} 個文件")
        
        print("選項:")
        print("1. 執行優化預處理")
        print("2. 測試表格檢測功能")
        
        choice = input("請選擇 (1-2): ").strip()
        
        if choice == "1":
            confirm = input("確定要批量優化處理所有文件嗎？(y/n): ").strip().lower()
            
            if confirm == 'y':
                try:
                    results = preprocess_multiple_documents_optimized([str(f) for f in pdf_files])
                    print(f"✅ 優化批量預處理完成！")
                    
                    # 顯示處理結果摘要
                    print("\n📋 優化處理摘要:")
                    for pdf_path, result in results.items():
                        metadata = result['metadata']
                        print(f"  ✓ {Path(pdf_path).name}: {metadata['company_name']} - {metadata['report_year']}")
                        
                except Exception as e:
                    print(f"❌ 優化批量預處理失敗: {e}")
        
        elif choice == "2":
            test_table_detection()

if __name__ == "__main__":
    main()