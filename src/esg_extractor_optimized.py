#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ESG資料提取器 v2.1
支援多文件處理、改進過濾邏輯、添加公司信息
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
from difflib import SequenceMatcher

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
# 關鍵字配置類（改進版）
# =============================================================================

class KeywordConfig:
    """關鍵字配置管理類"""
    
    # 擴展的關鍵字配置，增加更多相關詞彙
    CORE_KEYWORDS = {
        "再生塑膠材料": {
            "continuous": [
                "再生塑膠", "再生塑料", "再生料", "再生pp", "再生PET",
                "回收塑膠", "回收塑料", "回收PP", "回收PET",
                "PCR塑膠", "PCR塑料", "PCR材料", 
                "寶特瓶回收", "廢塑膠", "環保塑膠",
                "循環塑膠", "可回收塑膠"
            ],
            "discontinuous": [
                ("再生", "塑膠"), ("再生", "塑料"), ("再生", "PP"), ("再生", "PET"),
                ("PP", "回收"), ("PP", "再生"), ("PP", "棧板", "回收"),
                ("PET", "回收"), ("PET", "再生"),
                ("塑膠", "回收"), ("塑料", "回收"), ("塑膠", "循環"),
                ("PCR", "塑膠"), ("PCR", "塑料"), ("PCR", "材料"),
                ("回收", "塑膠"), ("回收", "塑料"), ("回收", "材料"),
                ("寶特瓶", "回收"), ("寶特瓶", "再造"), ("寶特瓶", "循環"),
                ("廢棄", "塑膠"), ("廢棄", "塑料"),
                ("rPET", "含量"), ("再生", "材料"), ("MLCC", "回收"),
                ("回收", "產能"), ("循環", "經濟"), ("環保", "材料"),
                ("回收", "造粒"), ("廢料", "回收")
            ]
        }
    }
    
    @classmethod
    def get_all_keywords(cls) -> List[Union[str, tuple]]:
        """獲取所有關鍵字（連續+不連續）"""
        all_keywords = []
        for category in cls.CORE_KEYWORDS.values():
            all_keywords.extend(category["continuous"])
            all_keywords.extend(category["discontinuous"])
        return all_keywords

# =============================================================================
# 增強匹配引擎（改進版）
# =============================================================================

class EnhancedMatcher:
    """增強的關鍵字匹配引擎"""
    
    def __init__(self, max_distance: int = 200):  # 增加最大距離
        self.max_distance = max_distance
        
        # 擴展的數值匹配模式
        self.number_patterns = [
            r'\d+(?:,\d{3})*(?:\.\d+)?(?:\s*(?:億|萬|千)?(?:支|個|件|批|台|套|次|倍))',
            r'\d+(?:,\d{3})*(?:\.\d+)?(?:\s*(?:kg|KG|公斤|噸|克|g|G|公克|萬噸|千噸))',
            r'\d+(?:,\d{3})*(?:\.\d+)?(?:\s*噸/月|噸/年|噸/日)',
            r'\d+(?:,\d{3})*(?:\.\d+)?(?:\s*(?:萬|千)?(?:噸|公斤|kg|g))',
            r'\d+(?:,\d{3})*(?:\.\d+)?(?:\s*立方米|m³)',
            r'\d+(?:,\d{3})*(?:\.\d+)?(?:\s*億支)',  # 新增：億支（如寶特瓶）
        ]
        
        # 百分比匹配模式
        self.percentage_patterns = [
            r'\d+(?:\.\d+)?(?:\s*%|％|百分比)',
            r'\d+(?:\.\d+)?(?:\s*成)',
            r'百分之\d+(?:\.\d+)?',
        ]
    
    def match_keyword(self, text: str, keyword: Union[str, tuple]) -> Tuple[bool, float, str]:
        """改進的關鍵字匹配，降低門檻但提高精確度"""
        text_lower = text.lower()
        
        if isinstance(keyword, str):
            # 連續關鍵字匹配
            if keyword.lower() in text_lower:
                pos = text_lower.find(keyword.lower())
                start = max(0, pos - 30)
                end = min(len(text), pos + len(keyword) + 30)
                context = text[start:end]
                return True, 1.0, f"精確匹配: {context}"
            return False, 0.0, ""
        
        elif isinstance(keyword, tuple):
            # 不連續關鍵字匹配 - 改進邏輯
            components = [comp.lower() for comp in keyword]
            positions = []
            
            # 找到每個組件的位置
            for comp in components:
                pos = text_lower.find(comp)
                if pos == -1:
                    return False, 0.0, f"缺少組件: {comp}"
                positions.append(pos)
            
            # 計算距離和信心分數
            min_pos = min(positions)
            max_pos = max(positions)
            distance = max_pos - min_pos
            
            # 提供匹配上下文
            start = max(0, min_pos - 50)
            end = min(len(text), max_pos + 50)
            context = text[start:end]
            
            # 調整距離判斷標準
            if distance <= 50:
                return True, 0.95, f"近距離匹配({distance}字): {context}"
            elif distance <= 120:
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
        
        # 提取數值
        for pattern in self.number_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            numbers.extend(matches)
        
        # 提取百分比
        for pattern in self.percentage_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            percentages.extend(matches)
        
        # 去重並排序
        numbers = list(set(numbers))
        percentages = list(set(percentages))
        
        return numbers, percentages

# =============================================================================
# 多文件ESG提取器
# =============================================================================

class MultiFileESGExtractor:
    """支援多文件處理的ESG提取器"""
    
    def __init__(self, enable_llm: bool = True):
        self.enable_llm = enable_llm
        self.matcher = EnhancedMatcher()
        self.keyword_config = KeywordConfig()
        
        # 初始化LLM（如果啟用）
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
        
        # 3. 改進的兩階段篩選
        stage2_extractions = self._improved_two_stage_filtering(documents, doc_info)
        
        # 4. LLM增強（如果啟用）
        enhanced_extractions = self._llm_enhancement(stage2_extractions)
        
        # 5. 創建處理摘要
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        
        keywords_found = {}
        for extraction in enhanced_extractions:
            keyword = extraction.keyword
            keywords_found[keyword] = keywords_found.get(keyword, 0) + 1
        
        summary = ProcessingSummary(
            company_name=doc_info.company_name,
            report_year=doc_info.report_year,
            total_documents=len(documents),
            stage1_passed=len(documents),  # 簡化統計
            stage2_passed=len(enhanced_extractions),
            total_extractions=len(enhanced_extractions),
            keywords_found=keywords_found,
            processing_time=processing_time
        )
        
        # 6. 匯出結果
        excel_path = self._export_to_excel(enhanced_extractions, summary, doc_info)
        
        return enhanced_extractions, summary, excel_path
    
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
                
                print(f"✅ 完成: {Path(excel_path).name}")
                
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
        
        # 對每個關鍵字進行檢索
        for keyword in keywords:
            search_term = keyword if isinstance(keyword, str) else " ".join(keyword)
            docs = db.similarity_search(search_term, k=30)
            all_docs.extend(docs)
        
        # 去重
        unique_docs = {}
        for doc in all_docs:
            doc_hash = hash(doc.page_content)
            if doc_hash not in unique_docs:
                unique_docs[doc_hash] = doc
        
        result_docs = list(unique_docs.values())[:max_docs]
        return result_docs
    
    def _improved_two_stage_filtering(self, documents: List[Document], doc_info: DocumentInfo) -> List[NumericExtraction]:
        """改進的兩階段篩選，避免過度過濾"""
        print("🔍 執行改進的兩階段篩選...")
        
        keywords = self.keyword_config.get_all_keywords()
        extractions = []
        
        for doc in tqdm(documents, desc="改進篩選"):
            # 改進：降低段落長度要求，分割更細
            paragraphs = self._split_into_paragraphs(doc.page_content, min_length=5)
            page_num = doc.metadata.get('page', '未知')
            
            for para_idx, paragraph in enumerate(paragraphs):
                if len(paragraph.strip()) < 5:  # 降低最小長度要求
                    continue
                
                # 第一階段：關鍵字匹配
                para_matches = []
                for keyword in keywords:
                    is_match, confidence, details = self.matcher.match_keyword(paragraph, keyword)
                    if is_match:
                        para_matches.append((keyword, confidence, details))
                
                if para_matches:
                    # 第二階段：數值提取（改進邏輯）
                    numbers, percentages = self.matcher.extract_numbers_and_percentages(paragraph)
                    
                    # 改進：即使沒有明確數值，也保留包含重要關鍵詞的段落
                    has_important_content = self._check_important_content(paragraph)
                    
                    if numbers or percentages or has_important_content:
                        # 為每個找到的數值創建提取結果
                        for number in numbers:
                            for keyword, confidence, details in para_matches:
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
                        
                        # 為每個找到的百分比創建提取結果
                        for percentage in percentages:
                            for keyword, confidence, details in para_matches:
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
                        
                        # 如果沒有具體數值但有重要內容，創建描述性結果
                        if not numbers and not percentages and has_important_content:
                            for keyword, confidence, details in para_matches:
                                keyword_str = keyword if isinstance(keyword, str) else " + ".join(keyword)
                                
                                extraction = NumericExtraction(
                                    keyword=keyword_str,
                                    value="[重要描述]",
                                    value_type='description',
                                    unit='',
                                    paragraph=paragraph.strip(),
                                    paragraph_number=para_idx + 1,
                                    page_number=f"第{page_num}頁",
                                    confidence=confidence * 0.8,  # 稍微降低信心分數
                                    context_window=self._get_context_window(doc.page_content, paragraph),
                                    company_name=doc_info.company_name,
                                    report_year=doc_info.report_year
                                )
                                extractions.append(extraction)
        
        print(f"✅ 改進篩選完成: 找到 {len(extractions)} 個提取結果")
        return extractions
    
    def _check_important_content(self, paragraph: str) -> bool:
        """檢查段落是否包含重要內容（即使沒有具體數值）"""
        important_indicators = [
            "回收", "再生", "循環", "減碳", "減排", "效益", "成果",
            "目標", "策略", "技術", "應用", "開發", "生產",
            "推動", "實施", "建置", "建立", "發展"
        ]
        
        paragraph_lower = paragraph.lower()
        matched_indicators = sum(1 for indicator in important_indicators if indicator in paragraph_lower)
        
        # 如果包含2個以上重要指標詞，且段落長度合理，就認為是重要內容
        return matched_indicators >= 2 and len(paragraph) >= 30
    
    def _llm_enhancement(self, extractions: List[NumericExtraction]) -> List[NumericExtraction]:
        """LLM增強（簡化版）"""
        if not self.enable_llm or not extractions:
            return extractions
        
        print(f"🤖 執行LLM增強 ({len(extractions)} 個結果)...")
        
        # 為了保持簡潔，這裡僅對信心分數低於0.7的結果進行LLM驗證
        enhanced_extractions = []
        
        for extraction in tqdm(extractions, desc="LLM增強"):
            if extraction.confidence >= 0.7:
                enhanced_extractions.append(extraction)
            else:
                # 對低信心結果進行LLM驗證
                try:
                    enhanced_extraction = self._llm_verify_extraction(extraction)
                    enhanced_extractions.append(enhanced_extraction)
                except:
                    enhanced_extractions.append(extraction)  # 失敗時保留原始結果
        
        return enhanced_extractions
    
    def _llm_verify_extraction(self, extraction: NumericExtraction) -> NumericExtraction:
        """LLM驗證單個提取結果"""
        prompt = f"""
請驗證以下數據提取是否合理：

關鍵字: {extraction.keyword}
提取值: {extraction.value}
段落: {extraction.paragraph[:200]}...

這個提取是否與再生塑膠/循環經濟相關？請回答 "相關" 或 "不相關"，並給出1-10的信心分數。
格式：[相關/不相關] [分數]
"""
        
        try:
            response = self.api_manager.invoke(prompt)
            
            # 簡單解析回應
            if "相關" in response and extraction.confidence < 0.9:
                extraction.confidence = min(extraction.confidence + 0.1, 0.9)
            elif "不相關" in response:
                extraction.confidence = max(extraction.confidence - 0.2, 0.3)
                
        except:
            pass  # LLM失敗時不修改
        
        return extraction
    
    def _export_to_excel(self, extractions: List[NumericExtraction], summary: ProcessingSummary, doc_info: DocumentInfo) -> str:
        """匯出結果到Excel，包含公司信息"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        company_safe = re.sub(r'[^\w\s-]', '', doc_info.company_name).strip()
        
        output_filename = f"ESG提取結果_{company_safe}_{doc_info.report_year}_{timestamp}.xlsx"
        output_path = os.path.join(RESULTS_PATH, output_filename)
        
        # 確保輸出目錄存在
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
            '上下文': f"提取器版本: v2.1"
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
        
        # 準備統計數據
        stats_data = []
        for keyword, count in summary.keywords_found.items():
            keyword_extractions = [e for e in extractions if e.keyword == keyword]
            numbers = [e for e in keyword_extractions if e.value_type == 'number']
            percentages = [e for e in keyword_extractions if e.value_type == 'percentage']
            descriptions = [e for e in keyword_extractions if e.value_type == 'description']
            
            stats_data.append({
                '關鍵字': keyword,
                '總提取數': len(keyword_extractions),
                '數值類型': len(numbers),
                '百分比類型': len(percentages),
                '描述類型': len(descriptions),
                '平均信心分數': round(np.mean([e.confidence for e in keyword_extractions]) if keyword_extractions else 0, 3),
                '最高信心分數': round(max([e.confidence for e in keyword_extractions]) if keyword_extractions else 0, 3)
            })
        
        # 寫入Excel
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            # 主要結果
            pd.DataFrame(main_data).to_excel(writer, sheet_name='提取結果', index=False)
            
            # 統計摘要
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
    
    def _split_into_paragraphs(self, text: str, min_length: int = 5) -> List[str]:
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
    """主函數 - 獨立運行測試"""
    try:
        print("🚀 多文件ESG資料提取器 - 測試模式")
        print("=" * 50)
        
        # 模擬文檔信息（實際使用時由預處理器提供）
        sample_docs = {
            "南亞_2023.pdf": DocumentInfo(
                company_name="南亞塑膠",
                report_year="2023",
                pdf_name="南亞_2023",
                db_path="./vector_db/esg_db_南亞_2023"
            )
        }
        
        # 初始化提取器
        extractor = MultiFileESGExtractor(enable_llm=True)
        
        # 處理文檔
        results = extractor.process_multiple_documents(sample_docs)
        
        if results:
            print(f"\n🎉 處理完成！")
            for pdf_path, (extractions, summary, excel_path) in results.items():
                print(f"📁 {summary.company_name} - {summary.report_year}: {len(extractions)} 個結果")
                print(f"   文件: {Path(excel_path).name}")
        
    except Exception as e:
        print(f"❌ 執行錯誤: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()