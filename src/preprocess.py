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
    
    # 3. 文本分割
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
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
            
            # 1. 提取元數據
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
    return results

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
                
                # 顯示處理結果摘要
                print("\n📋 處理摘要:")
                for pdf_path, result in results.items():
                    metadata = result['metadata']
                    print(f"  ✓ {Path(pdf_path).name}: {metadata['company_name']} - {metadata['report_year']}")
                    
            except Exception as e:
                print(f"❌ 批量預處理失敗: {e}")

if __name__ == "__main__":
    main()