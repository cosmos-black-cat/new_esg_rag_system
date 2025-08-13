import os
import sys
import re
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from tqdm import tqdm
from typing import List, Tuple, Dict
sys.path.append(str(Path(__file__).parent))
from config import *

def extract_company_info_from_filename(pdf_path: str) -> Tuple[str, str]:
    """從PDF文件名提取公司名稱和年度"""
    filename = Path(pdf_path).stem
    
    # 嘗試匹配年度 (2020-2030)
    year_match = re.search(r'(202[0-9])', filename)
    year = year_match.group(1) if year_match else "未知年度"
    
    # 提取公司名稱 (移除年度後的剩餘部分)
    company_name = filename
    if year_match:
        company_name = re.sub(r'[_\-\s]*202[0-9][_\-\s]*', '', company_name)
    
    # 清理公司名稱
    company_name = re.sub(r'[_\-\s]+', ' ', company_name).strip()
    if not company_name:
        company_name = "未知公司"
    
    return company_name, year

def extract_company_info_from_content(pdf_path: str) -> Tuple[str, str]:
    """從PDF內容提取公司名稱和年度（備用方法）"""
    try:
        loader = PyPDFLoader(pdf_path)
        pages = loader.load()
        
        if not pages:
            return "未知公司", "未知年度"
        
        # 檢查前幾頁內容
        first_pages_content = " ".join([page.page_content for page in pages[:3]])
        
        # 尋找年度
        year_patterns = [
            r'(202[0-9])\s*年.*報告',
            r'(202[0-9])\s*年.*永續',
            r'(202[0-9])\s*Annual\s*Report',
            r'(202[0-9])'
        ]
        
        year = "未知年度"
        for pattern in year_patterns:
            match = re.search(pattern, first_pages_content)
            if match:
                year = match.group(1)
                break
        
        # 尋找公司名稱
        company_patterns = [
            r'([^。\n]{2,20}(?:股份有限公司|有限公司|公司))',
            r'([^。\n]{2,20}(?:Corporation|Corp|Company|Ltd))',
        ]
        
        company = "未知公司"
        for pattern in company_patterns:
            matches = re.findall(pattern, first_pages_content)
            if matches:
                # 選擇最短的匹配（通常是公司簡稱）
                company = min(matches, key=len).strip()
                break
        
        return company, year
        
    except Exception as e:
        print(f"⚠️ 無法從內容提取公司信息: {e}")
        return "未知公司", "未知年度"

def get_pdf_company_info(pdf_path: str) -> Dict[str, str]:
    """獲取PDF的公司信息"""
    # 首先嘗試從文件名提取
    company_from_filename, year_from_filename = extract_company_info_from_filename(pdf_path)
    
    # 如果文件名提取失敗，嘗試從內容提取
    if company_from_filename == "未知公司" or year_from_filename == "未知年度":
        company_from_content, year_from_content = extract_company_info_from_content(pdf_path)
        
        # 使用更好的結果
        company = company_from_content if company_from_filename == "未知公司" else company_from_filename
        year = year_from_content if year_from_filename == "未知年度" else year_from_filename
    else:
        company = company_from_filename
        year = year_from_filename
    
    return {
        'company_name': company,
        'report_year': year,
        'pdf_filename': Path(pdf_path).name
    }

def preprocess_single_document(pdf_path: str, output_db_path: str = None) -> Tuple[FAISS, Dict[str, str]]:
    """預處理單個PDF文檔並建立向量資料庫"""
    
    if output_db_path is None:
        # 為每個PDF創建獨立的資料庫路徑
        pdf_stem = Path(pdf_path).stem
        output_db_path = f"./vector_db/{pdf_stem}_db"
    
    print(f"📄 開始處理PDF: {Path(pdf_path).name}")
    
    # 提取公司信息
    company_info = get_pdf_company_info(pdf_path)
    print(f"📊 識別公司: {company_info['company_name']} ({company_info['report_year']})")
    
    # 1. 載入PDF
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"找不到PDF文件: {pdf_path}")
    
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    print(f"✅ 成功載入 {len(pages)} 頁")
    
    # 在文檔metadata中加入公司信息
    for page in pages:
        page.metadata.update(company_info)
    
    # 2. 文本分割
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,  # 增加chunk size以減少信息丟失
        chunk_overlap=200,  # 增加overlap
        separators=["\n\n", "\n", "。", "，", " ", ""]
    )
    
    print("🔄 正在分割文本...")
    chunks = text_splitter.split_documents(pages)
    print(f"✅ 分割成 {len(chunks)} 個文本塊")
    
    # 3. 初始化embedding模型
    print(f"🧠 載入embedding模型: {EMBEDDING_MODEL}")
    embedding_model = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={'device': 'cpu'}
    )
    
    # 4. 建立向量資料庫
    print("🔍 建立向量資料庫...")
    db = FAISS.from_documents(chunks, embedding_model)
    
    # 5. 保存資料庫
    os.makedirs(os.path.dirname(output_db_path), exist_ok=True)
    db.save_local(output_db_path)
    print(f"💾 向量資料庫已保存到: {output_db_path}")
    
    return db, company_info

def preprocess_multiple_documents(data_dir: str = None) -> List[Tuple[str, Dict[str, str]]]:
    """預處理多個PDF文檔"""
    if data_dir is None:
        data_dir = DATA_PATH
    
    data_path = Path(data_dir)
    pdf_files = list(data_path.glob("*.pdf"))
    
    if not pdf_files:
        print(f"❌ 在 {data_dir} 目錄中找不到PDF文件")
        return []
    
    print(f"📚 找到 {len(pdf_files)} 個PDF文件")
    
    processed_files = []
    
    for pdf_path in pdf_files:
        try:
            print(f"\n{'='*60}")
            db, company_info = preprocess_single_document(str(pdf_path))
            
            # 記錄處理結果
            db_path = f"./vector_db/{pdf_path.stem}_db"
            processed_files.append((db_path, company_info))
            
            print(f"✅ {pdf_path.name} 處理完成")
            
        except Exception as e:
            print(f"❌ 處理 {pdf_path.name} 失敗: {e}")
            continue
    
    print(f"\n🎉 批量預處理完成！成功處理 {len(processed_files)}/{len(pdf_files)} 個文件")
    return processed_files

def preprocess_documents(pdf_path: str, output_db_path: str = None):
    """預處理PDF文檔並建立向量資料庫（保持向後兼容）"""
    db, company_info = preprocess_single_document(pdf_path, output_db_path)
    return db

def main():
    """主函數"""
    # 檢查data目錄中的PDF文件
    data_dir = Path(DATA_PATH)
    pdf_files = list(data_dir.glob("*.pdf"))
    
    if not pdf_files:
        print(f"❌ 在 {DATA_PATH} 目錄中找不到PDF文件")
        print("請將ESG報告PDF文件放入data目錄中")
        return
    
    if len(pdf_files) == 1:
        # 單個文件處理
        pdf_path = pdf_files[0]
        print(f"🎯 處理單個文件: {pdf_path}")
        
        try:
            preprocess_documents(str(pdf_path))
            print("✅ 預處理完成！")
        except Exception as e:
            print(f"❌ 預處理失敗: {e}")
    else:
        # 多個文件批量處理
        print(f"🎯 批量處理 {len(pdf_files)} 個文件")
        
        try:
            processed_files = preprocess_multiple_documents()
            if processed_files:
                print("✅ 批量預處理完成！")
                print("\n📋 處理結果:")
                for db_path, company_info in processed_files:
                    print(f"   📁 {company_info['company_name']} ({company_info['report_year']})")
            else:
                print("❌ 沒有成功處理任何文件")
        except Exception as e:
            print(f"❌ 批量預處理失敗: {e}")

if __name__ == "__main__":
    main()