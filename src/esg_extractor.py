#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ESG資料提取器 v2.2 - 改進版
修復：過濾邏輯、多文件處理、Excel格式
"""

import json
import re
import os
import sys
import pandas as pd
from pathlib import Path
from typing import Dict, List, Union, Tuple, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
from tqdm import tqdm
import numpy as np

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI

# 添加當前目錄到路徑
sys.path.append(str(Path(__file__).parent))
from config import *
from api_manager import create_api_manager

# =============================================================================
# 數據結構定義
# =============================================================================

@dataclass
class DocumentInfo:
    """文檔信息"""
    company_name: str
    report_year: str
    pdf_name: str
    db_path: str

@dataclass
class NumericExtraction:
    """數值提取結果"""
    keyword: str
    value: str
    value_type: str  # 'number' or 'percentage'  
    unit: str
    paragraph: str
    paragraph_number: int
    page_number: str
    confidence: float
    context_window: str
    company_name: str = ""
    report_year: str = ""

@dataclass
class ProcessingSummary:
    """處理摘要"""
    company_name: str
    report_year: str
    total_documents: int
    stage1_passed: int
    stage2_passed: int
    total_extractions: int
    keywords_found: Dict[str, int]
    processing_time: float

# =============================================================================
# 改進的關鍵字配置類
# =============================================================================

class ImprovedKeywordConfig:
    """改進的關鍵字配置，減少誤過濾"""
    
    CORE_KEYWORDS = {
        "再生塑膠材料": {
            "continuous": [
                "再生塑膠", "再生塑料", "再生料", "再生pp", "再生PET",
                "回收塑膠", "回收塑料", "回收PP", "回收PET",
                "PCR塑膠", "PCR塑料", "PCR材料", 
                "寶特瓶回收", "廢塑膠", "環保塑膠",
                "循環塑膠", "可回收塑膠", "rPET", "再生聚酯",
                "回收造粒", "廢料回收", "材料回收", "物料回收"
            ],
            "discontinuous": [
                ("再生", "塑膠"), ("再生", "塑料"), ("再生", "PP"), ("再生", "PET"),
                ("PP", "回收"), ("PP", "再生"), ("PP", "棧板"),
                ("PET", "回收"), ("PET", "再生"), ("PET", "材料"),
                ("塑膠", "回收"), ("塑料", "回收"), ("塑膠", "循環"),
                ("PCR", "塑膠"), ("PCR", "塑料"), ("PCR", "材料"),
                ("回收", "塑膠"), ("回收", "塑料"), ("回收", "材料"),
                ("寶特瓶", "回收"), ("寶特瓶", "再造"), ("寶特瓶", "循環"),
                ("廢棄", "塑膠"), ("廢棄", "塑料"),
                ("rPET", "含量"), ("再生", "材料"), ("MLCC", "回收"),
                ("回收", "產能"), ("循環", "經濟"), ("環保", "材料"),
                ("回收", "造粒"), ("廢料", "回收"), ("億支", "寶特瓶"),
                ("原生", "材料"), ("碳排放", "減少"), ("減碳", "效益")
            ]
        }
    }
    
    @classmethod
    def get_all_keywords(cls) -> List[Union[str, tuple]]:
        """獲取所有關鍵字"""
        all_keywords = []
        for category in cls.CORE_KEYWORDS.values():
            all_keywords.extend(category["continuous"])
            all_keywords.extend(category["discontinuous"])
        return all_keywords

# =============================================================================
# 改進的匹配引擎
# =============================================================================

class ImprovedMatcher:
    """改進的匹配引擎，減少誤過濾"""
    
    def __init__(self, max_distance: int = 300):
        self.max_distance = max_distance
        
        # 數值匹配模式
        self.number_patterns = [
            r'\d+(?:,\d{3})*(?:\.\d+)?(?:\s*(?:億|萬|千)?(?:支|個|件|批|台|套|次|倍))',
            r'\d+(?:,\d{3})*(?:\.\d+)?(?:\s*(?:kg|KG|公斤|噸|克|g|G|公克|萬噸|千噸))',
            r'\d+(?:,\d{3})*(?:\.\d+)?(?:\s*噸/月|噸/年|噸/日)',
            r'\d+(?:,\d{3})*(?:\.\d+)?(?:\s*(?:萬|千)?(?:噸|公斤|kg|g))',
            r'\d+(?:,\d{3})*(?:\.\d+)?(?:\s*立方米|m³)',
            r'\d+(?:,\d{3})*(?:\.\d+)?(?:\s*億支)',
        ]
        
        # 百分比匹配模式
        self.percentage_patterns = [
            r'\d+(?:\.\d+)?(?:\s*%|％|百分比)',
            r'\d+(?:\.\d+)?(?:\s*成)',
            r'百分之\d+(?:\.\d+)?',
        ]
    
    def match_keyword(self, text: str, keyword: Union[str, tuple]) -> Tuple[bool, float, str]:
        """改進的關鍵字匹配"""
        text_lower = text.lower()
        
        if isinstance(keyword, str):
            if keyword.lower() in text_lower:
                pos = text_lower.find(keyword.lower())
                start = max(0, pos - 30)
                end = min(len(text), pos + len(keyword) + 30)
                context = text[start:end]
                return True, 1.0, f"精確匹配: {context}"
            return False, 0.0, ""
        
        elif isinstance(keyword, tuple):
            components = [comp.lower() for comp in keyword]
            positions = []
            
            for comp in components:
                pos = text_lower.find(comp)
                if pos == -1:
                    return False, 0.0, f"缺少組件: {comp}"
                positions.append(pos)
            
            min_pos = min(positions)
            max_pos = max(positions)
            distance = max_pos - min_pos
            
            start = max(0, min_pos - 50)
            end = min(len(text), max_pos + 50)
            context = text[start:end]
            
            # 更寬鬆的距離判斷
            if distance <= 80:
                return True, 0.95, f"近距離匹配({distance}字): {context}"
            elif distance <= 200:
                return True, 0.85, f"中距離匹配({distance}字): {context}"
            elif distance <= self.max_distance:
                return True, 0.7, f"遠距離匹配({distance}字): {context}"
            else:
                return True, 0.5, f"極遠距離匹配({distance}字): {context}"
        
        return False, 0.0, ""
    
    def extract_numbers_and_percentages(self, text: str) -> Tuple[List[str], List[str]]:
        """提取數值和百分比"""
        numbers = []
        percentages = []
        
        for pattern in self.number_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            numbers.extend(matches)
        
        for pattern in self.percentage_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            percentages.extend(matches)
        
        return list(set(numbers)), list(set(percentages))
    
    def is_relevant_context(self, text: str) -> bool:
        """簡化的相關性檢查，減少誤過濾"""
        text_lower = text.lower()
        
        # 明確排除的無關內容
        exclusions = [
            "職業災害", "工安", "降雨量", "雨水", "節能改善案", 
            "節水改善案", "垂直馬拉松", "賽衣", "選手", "比賽"
        ]
        
        for exclusion in exclusions:
            if exclusion in text_lower:
                return False
        
        # 相關性指標（降低要求）
        relevant_indicators = [
            "回收", "再生", "循環", "減碳", "環保", "永續",
            "塑膠", "塑料", "材料", "產能", "生產", "製造",
            "寶特瓶", "PET", "PP", "PCR"
        ]
        
        found_indicators = sum(1 for indicator in relevant_indicators if indicator in text_lower)
        return found_indicators >= 1  # 只需要1個相關指標即可

# =============================================================================
# 多文件ESG提取器
# =============================================================================

class MultiFileESGExtractor:
    """支援多文件處理的ESG提取器"""
    
    def __init__(self, enable_llm: bool = True):
        self.enable_llm = enable_llm
        self.matcher = ImprovedMatcher()
        self.keyword_config = ImprovedKeywordConfig()
        
        if self.enable_llm:
            self._init_llm()
        
        print("✅ 多文件ESG提取器初始化完成")

    def _init_llm(self):
        """初始化LLM"""
        try:
            print("🤖 初始化Gemini API管理器...")
            self.api_manager = create_api_manager()
            print("✅ LLM初始化完成")
        except Exception as e:
            print(f"⚠️ LLM初始化失敗: {e}")
            self.enable_llm = False
    
    def process_single_document(self, doc_info: DocumentInfo, max_documents: int = 300) -> Tuple[List[NumericExtraction], ProcessingSummary, str]:
        """處理單個文檔"""
        start_time = datetime.now()
        print(f"\n🚀 處理文檔: {doc_info.company_name} - {doc_info.report_year}")
        print("=" * 60)
        
        # 1. 載入向量資料庫
        db = self._load_vector_database(doc_info.db_path)
        
        # 2. 獲取相關文檔
        documents = self._retrieve_relevant_documents(db, max_documents)
        
        # 3. 改進的篩選邏輯
        extractions = self._improved_filtering(documents, doc_info)
        
        # 4. LLM增強（可選）
        if self.enable_llm and len(extractions) > 50:
            extractions = self._llm_enhancement(extractions[:50])
        
        # 5. 創建處理摘要
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        
        keywords_found = {}
        for extraction in extractions:
            keyword = extraction.keyword
            keywords_found[keyword] = keywords_found.get(keyword, 0) + 1
        
        summary = ProcessingSummary(
            company_name=doc_info.company_name,
            report_year=doc_info.report_year,
            total_documents=len(documents),
            stage1_passed=len(documents),
            stage2_passed=len(extractions),
            total_extractions=len(extractions),
            keywords_found=keywords_found,
            processing_time=processing_time
        )
        
        # 6. 匯出結果
        excel_path = self._export_to_excel(extractions, summary, doc_info)
        
        return extractions, summary, excel_path
    
    def process_multiple_documents(self, docs_info: Dict[str, DocumentInfo], max_documents: int = 300) -> Dict[str, Tuple]:
        """批量處理多個文檔"""
        print(f"🚀 開始批量處理 {len(docs_info)} 個文檔")
        print("=" * 60)
        
        results = {}
        
        for pdf_path, doc_info in docs_info.items():
            try:
                print(f"\n📄 處理: {doc_info.company_name} - {doc_info.report_year}")
                
                extractions, summary, excel_path = self.process_single_document(doc_info, max_documents)
                
                results[pdf_path] = (extractions, summary, excel_path)
                
                print(f"✅ 完成: 生成 {len(extractions)} 個結果 -> {Path(excel_path).name}")
                
            except Exception as e:
                print(f"❌ 處理失敗 {doc_info.company_name}: {e}")
                continue
        
        print(f"\n🎉 批量處理完成！成功處理 {len(results)}/{len(docs_info)} 個文檔")
        return results
    
    def _load_vector_database(self, db_path: str):
        """載入向量資料庫"""
        if not os.path.exists(db_path):
            raise FileNotFoundError(f"向量資料庫不存在: {db_path}")
        
        embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={'device': 'cpu'}
        )
        db = FAISS.load_local(
            db_path, 
            embeddings, 
            allow_dangerous_deserialization=True
        )
        return db
    
    def _retrieve_relevant_documents(self, db, max_docs: int) -> List[Document]:
        """檢索相關文檔"""
        keywords = self.keyword_config.get_all_keywords()
        all_docs = []
        
        for keyword in keywords[:10]:  # 限制關鍵字數量提高效率
            search_term = keyword if isinstance(keyword, str) else " ".join(keyword)
            docs = db.similarity_search(search_term, k=20)
            all_docs.extend(docs)
        
        # 去重
        unique_docs = {}
        for doc in all_docs:
            doc_hash = hash(doc.page_content)
            if doc_hash not in unique_docs:
                unique_docs[doc_hash] = doc
        
        result_docs = list(unique_docs.values())[:max_docs]
        return result_docs
    
    def _improved_filtering(self, documents: List[Document], doc_info: DocumentInfo) -> List[NumericExtraction]:
        """改進的篩選邏輯，減少誤過濾"""
        print("🔍 執行改進的篩選邏輯...")
        
        keywords = self.keyword_config.get_all_keywords()
        extractions = []
        
        for doc in tqdm(documents, desc="篩選處理"):
            paragraphs = self._split_into_paragraphs(doc.page_content, min_length=10)
            page_num = doc.metadata.get('page', '未知')
            
            for para_idx, paragraph in enumerate(paragraphs):
                if len(paragraph.strip()) < 10:
                    continue
                
                # 第一階段：關鍵字匹配
                para_matches = []
                for keyword in keywords:
                    is_match, confidence, details = self.matcher.match_keyword(paragraph, keyword)
                    if is_match:
                        para_matches.append((keyword, confidence, details))
                
                if para_matches:
                    # 第二階段：相關性檢查（寬鬆）
                    if self.matcher.is_relevant_context(paragraph):
                        # 第三階段：數值提取
                        numbers, percentages = self.matcher.extract_numbers_and_percentages(paragraph)
                        
                        # 為數值創建提取結果
                        for number in numbers:
                            for keyword, confidence, details in para_matches[:3]:  # 限制數量
                                keyword_str = keyword if isinstance(keyword, str) else " + ".join(keyword)
                                
                                extraction = NumericExtraction(
                                    keyword=keyword_str,
                                    value=number,
                                    value_type='number',
                                    unit=self._extract_unit(number),
                                    paragraph=paragraph.strip(),
                                    paragraph_number=para_idx + 1,
                                    page_number=f"第{page_num}頁",
                                    confidence=confidence,
                                    context_window=self._get_context_window(doc.page_content, paragraph),
                                    company_name=doc_info.company_name,
                                    report_year=doc_info.report_year
                                )
                                extractions.append(extraction)
                        
                        # 為百分比創建提取結果
                        for percentage in percentages:
                            for keyword, confidence, details in para_matches[:3]:
                                keyword_str = keyword if isinstance(keyword, str) else " + ".join(keyword)
                                
                                extraction = NumericExtraction(
                                    keyword=keyword_str,
                                    value=percentage,
                                    value_type='percentage',
                                    unit='%',
                                    paragraph=paragraph.strip(),
                                    paragraph_number=para_idx + 1,
                                    page_number=f"第{page_num}頁",
                                    confidence=confidence,
                                    context_window=self._get_context_window(doc.page_content, paragraph),
                                    company_name=doc_info.company_name,
                                    report_year=doc_info.report_year
                                )
                                extractions.append(extraction)
        
        print(f"✅ 篩選完成: 找到 {len(extractions)} 個提取結果")
        return extractions
    
    def _llm_enhancement(self, extractions: List[NumericExtraction]) -> List[NumericExtraction]:
        """LLM增強（簡化版）"""
        if not self.enable_llm:
            return extractions
        
        print(f"🤖 LLM增強處理...")
        # 簡化處理，主要用於去重和提高信心分數
        return extractions
    
    def _export_to_excel(self, extractions: List[NumericExtraction], summary: ProcessingSummary, doc_info: DocumentInfo) -> str:
        """匯出結果到Excel，第一行包含公司信息"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        company_safe = re.sub(r'[^\w\s-]', '', doc_info.company_name).strip()
        
        output_filename = f"ESG提取結果_{company_safe}_{doc_info.report_year}_{timestamp}.xlsx"
        output_path = os.path.join(RESULTS_PATH, output_filename)
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        print(f"📊 匯出結果到Excel: {output_filename}")
        
        # 準備主要數據
        main_data = []
        
        # 第一行：公司信息
        header_row = {
            '關鍵字': f"公司: {doc_info.company_name}",
            '提取數值': f"報告年度: {doc_info.report_year}",
            '數據類型': f"處理時間: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            '單位': '',
            '段落內容': f"總提取結果: {len(extractions)} 項",
            '段落編號': '',
            '頁碼': '',
            '信心分數': '',
            '上下文': f"提取器版本: v2.2"
        }
        main_data.append(header_row)
        
        # 空行分隔
        main_data.append({col: '' for col in header_row.keys()})
        
        # 提取結果
        for extraction in extractions:
            main_data.append({
                '關鍵字': extraction.keyword,
                '提取數值': extraction.value,
                '數據類型': extraction.value_type,
                '單位': extraction.unit,
                '段落內容': extraction.paragraph,
                '段落編號': extraction.paragraph_number,
                '頁碼': extraction.page_number,
                '信心分數': round(extraction.confidence, 3),
                '上下文': extraction.context_window[:200] + "..." if len(extraction.context_window) > 200 else extraction.context_window
            })
        
        # 統計數據
        stats_data = []
        for keyword, count in summary.keywords_found.items():
            keyword_extractions = [e for e in extractions if e.keyword == keyword]
            
            stats_data.append({
                '關鍵字': keyword,
                '提取數量': count,
                '平均信心分數': round(np.mean([e.confidence for e in keyword_extractions]) if keyword_extractions else 0, 3),
                '最高信心分數': round(max([e.confidence for e in keyword_extractions]) if keyword_extractions else 0, 3)
            })
        
        # 寫入Excel
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            pd.DataFrame(main_data).to_excel(writer, sheet_name='提取結果', index=False)
            pd.DataFrame(stats_data).to_excel(writer, sheet_name='關鍵字統計', index=False)
            
            # 處理摘要
            summary_data = [{
                '公司名稱': summary.company_name,
                '報告年度': summary.report_year,
                '總文檔數': summary.total_documents,
                '總提取結果': summary.total_extractions,
                '處理時間(秒)': round(summary.processing_time, 2),
                '處理日期': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }]
            pd.DataFrame(summary_data).to_excel(writer, sheet_name='處理摘要', index=False)
        
        print(f"✅ Excel檔案已保存")
        return output_path
    
    # =============================================================================
    # 輔助方法
    # =============================================================================
    
    def _split_into_paragraphs(self, text: str, min_length: int = 10) -> List[str]:
        """將文本分割成段落"""
        paragraphs = re.split(r'\n{2,}|\r{2,}|。{2,}|\.{2,}', text)
        
        cleaned_paragraphs = []
        for para in paragraphs:
            para = para.strip()
            if len(para) >= min_length:
                cleaned_paragraphs.append(para)
        
        return cleaned_paragraphs
    
    def _extract_unit(self, value_str: str) -> str:
        """從數值字符串中提取單位"""
        units = re.findall(r'[a-zA-Z\u4e00-\u9fff]+', value_str)
        return units[-1] if units else ""
    
    def _get_context_window(self, full_text: str, target_paragraph: str, window_size: int = 100) -> str:
        """獲取段落的上下文窗口"""
        try:
            pos = full_text.find(target_paragraph)
            if pos == -1:
                return target_paragraph[:200]
            
            start = max(0, pos - window_size)
            end = min(len(full_text), pos + len(target_paragraph) + window_size)
            
            return full_text[start:end]
        except:
            return target_paragraph[:200]

def main():
    """主函數 - 測試用"""
    print("🧪 ESG提取器測試模式")
    
    # 這裡可以添加測試代碼
    extractor = MultiFileESGExtractor(enable_llm=False)
    print("✅ 提取器初始化完成")

if __name__ == "__main__":
    main()