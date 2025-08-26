#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ESG報告書提取器核心模組 v1.0
專門提取ESG報告中的再生塑膠相關數據
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
    keyword_distance: int = 0

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
# 關鍵字配置
# =============================================================================

class KeywordConfig:
    """ESG報告書關鍵字配置"""
    
    RECYCLED_PLASTIC_KEYWORDS = {
        "high_relevance_continuous": [
            "再生塑膠", "再生塑料", "再生料", "再生PET", "再生PP",
            "回收塑膠", "回收塑料", "回收PP", "回收PET", 
            "rPET", "PCR塑膠", "PCR塑料", "PCR材料",
            "寶特瓶回收", "廢塑膠回收", "塑膠循環",
            "回收造粒", "再生聚酯", "回收聚酯",
            "循環經濟", "物料回收", "材料回收"
        ],
        
        "medium_relevance_continuous": [
            "環保塑膠", "綠色材料", "永續材料",
            "廢料回收", "資源回收", "循環利用"
        ],
        
        "high_relevance_discontinuous": [
            ("寶特瓶", "回收"), ("寶特瓶", "再造"), ("寶特瓶", "循環"),
            ("億支", "寶特瓶"), ("萬支", "寶特瓶"),
            ("PET", "回收"), ("PET", "再生"), ("PP", "回收"), ("PP", "再生"),
            ("塑膠", "回收"), ("塑料", "回收"), ("塑膠", "循環"),
            ("回收", "造粒"), ("回收", "產能"), ("回收", "材料"),
            ("再生", "材料"), ("廢料", "回收"), ("MLCC", "回收"),
            ("原生", "材料"), ("碳排放", "減少"), ("減碳", "效益"),
            ("歷年", "回收"), ("回收", "數量"), ("回收", "效益"),
            ("循環", "經濟"), ("永續", "發展"), ("環保", "產品")
        ]
    }
    
    EXCLUSION_RULES = {
        "exclude_topics": [
            "職業災害", "工安", "安全事故", "職災",
            "馬拉松", "賽事", "選手", "比賽", "賽衣", "運動",
            "雨水回收", "廢水處理", "水質監測",
            "改善案", "改善專案", "案例選拔",
            "能源轉型", "燃油改燃", "鍋爐改善", "天然氣燃燒",
            "節能產品", "隔熱漆", "節能窗", "隔熱紙", "酷樂漆",
            "氣密窗", "隔熱產品", "保溫材料", "建材產品",
            "太陽能", "風電", "綠能", "光電", "電池材料"
        ],
        
        "exclude_contexts": [
            "垂直馬拉松", "史上最環保賽衣", "各界好手",
            "職業災害比率", "工安統計", 
            "節能改善案", "節水改善案", "優良案例",
            "雨水回收量減少", "降雨量減少",
            "燃油改燃汽鍋爐", "天然氣燃燒機", "鍋爐改造",
            "酷樂漆", "隔熱漆", "節能氣密窗", "冰酷隔熱紙"
        ]
    }
    
    PLASTIC_INDICATORS = {
        "plastic_materials": [
            "塑膠", "塑料", "聚酯", "PET", "PP", "聚合物",
            "樹脂", "粒子", "顆粒", "材料", "塑膠粒", "聚酯粒",
            "寶特瓶", "瓶片", "容器", "包裝", "膜材", "纖維"
        ],
        
        "recycling_specific": [
            "回收", "再生", "循環", "再利用", "回收利用",
            "造粒", "再製", "轉換", "處理", "循環經濟",
            "廢料", "廢棄", "回收料", "再生料", "PCR"
        ]
    }
    
    @classmethod
    def get_all_keywords(cls) -> List[Union[str, tuple]]:
        """獲取所有關鍵字"""
        all_keywords = []
        all_keywords.extend(cls.RECYCLED_PLASTIC_KEYWORDS["high_relevance_continuous"])
        all_keywords.extend(cls.RECYCLED_PLASTIC_KEYWORDS["medium_relevance_continuous"])
        all_keywords.extend(cls.RECYCLED_PLASTIC_KEYWORDS["high_relevance_discontinuous"])
        return all_keywords

# =============================================================================
# 匹配引擎
# =============================================================================

class ESGMatcher:
    """ESG數據匹配引擎"""
    
    def __init__(self):
        self.config = KeywordConfig()
        self.max_distance = 300
        
        # 數值匹配模式
        self.number_patterns = [
            r'\d+(?:\.\d+)?\s*億支',
            r'\d+(?:\.\d+)?\s*萬支',
            r'\d+(?:,\d{3})*(?:\.\d+)?\s*(?:萬|千)?噸',
            r'\d+(?:,\d{3})*(?:\.\d+)?\s*(?:kg|KG|公斤)',
            r'\d+(?:,\d{3})*(?:\.\d+)?\s*噸/月',
            r'\d+(?:,\d{3})*(?:\.\d+)?\s*噸/年',
            r'\d+(?:,\d{3})*(?:\.\d+)?\s*噸/日',
            r'\d+(?:,\d{3})*(?:\.\d+)?\s*(?:件|個|批|台|套)',
        ]
        
        # 百分比匹配模式
        self.percentage_patterns = [
            r'\d+(?:\.\d+)?\s*%',
            r'\d+(?:\.\d+)?\s*％',
            r'百分之\d+(?:\.\d+)?',
        ]
    
    def extract_keyword_value_pairs(self, text: str, keyword: Union[str, tuple]) -> List[Tuple[str, str, float, int]]:
        """
        提取關鍵字與數值的配對
        返回: [(數值, 數值類型, 關聯度分數, 距離)]
        """
        text_lower = text.lower()
        
        # 1. 檢查關鍵字是否存在
        keyword_match, keyword_confidence, _ = self._match_keyword(text, keyword)
        if not keyword_match:
            return []
        
        # 2. 找到關鍵字位置
        keyword_positions = self._get_keyword_positions(text_lower, keyword)
        if not keyword_positions:
            return []
        
        # 3. 在關鍵字附近尋找數值
        valid_pairs = []
        
        for kw_start, kw_end in keyword_positions:
            # 搜索範圍
            search_start = max(0, kw_start - 100)
            search_end = min(len(text), kw_end + 100)
            search_window = text[search_start:search_end]
            
            # 提取數值
            numbers = self._extract_numbers_in_window(search_window)
            percentages = self._extract_percentages_in_window(search_window)
            
            # 驗證數值關聯性
            for number in numbers:
                number_pos = search_window.find(number)
                if number_pos != -1:
                    actual_number_pos = search_start + number_pos
                    distance = min(abs(actual_number_pos - kw_start), abs(actual_number_pos - kw_end))
                    
                    association_score = self._calculate_association(
                        text, keyword, number, kw_start, kw_end, actual_number_pos
                    )
                    
                    if association_score > 0.5 and distance <= 80:
                        valid_pairs.append((number, 'number', association_score, distance))
            
            for percentage in percentages:
                percentage_pos = search_window.find(percentage)
                if percentage_pos != -1:
                    actual_percentage_pos = search_start + percentage_pos
                    distance = min(abs(actual_percentage_pos - kw_start), abs(actual_percentage_pos - kw_end))
                    
                    association_score = self._calculate_association(
                        text, keyword, percentage, kw_start, kw_end, actual_percentage_pos
                    )
                    
                    if association_score > 0.5 and distance <= 80:
                        valid_pairs.append((percentage, 'percentage', association_score, distance))
        
        # 去重並排序
        unique_pairs = []
        seen_values = set()
        
        for value, value_type, score, distance in sorted(valid_pairs, key=lambda x: x[2], reverse=True):
            if value not in seen_values:
                seen_values.add(value)
                unique_pairs.append((value, value_type, score, distance))
        
        return unique_pairs[:3]
    
    def comprehensive_relevance_check(self, text: str, keyword: Union[str, tuple]) -> Tuple[bool, float, str]:
        """綜合相關性檢查"""
        text_lower = text.lower()
        
        # 1. 排除檢查
        if self._is_excluded_content(text_lower):
            return False, 0.0, "明確無關內容"
        
        # 2. 關鍵字匹配
        keyword_match, keyword_confidence, keyword_details = self._match_keyword(text, keyword)
        if not keyword_match:
            return False, 0.0, "關鍵字不匹配"
        
        # 3. 塑膠相關性檢查
        plastic_relevance = self._check_plastic_relevance(text_lower)
        if plastic_relevance < 0.3:
            return False, 0.0, f"非塑膠相關內容: {plastic_relevance:.2f}"
        
        # 4. 計算綜合分數
        relevance_score = self._calculate_relevance_score(text_lower)
        
        final_score = (
            keyword_confidence * 0.3 + 
            plastic_relevance * 0.3 + 
            relevance_score * 0.4
        )
        
        is_relevant = final_score > 0.55
        
        details = f"關鍵字:{keyword_confidence:.2f}, 塑膠相關:{plastic_relevance:.2f}, 相關性:{relevance_score:.2f}"
        
        return is_relevant, final_score, details
    
    # 輔助方法
    def _match_keyword(self, text: str, keyword: Union[str, tuple]) -> Tuple[bool, float, str]:
        """關鍵字匹配"""
        text_lower = text.lower()
        
        if isinstance(keyword, str):
            if keyword.lower() in text_lower:
                return True, 1.0, f"精確匹配: {keyword}"
            return False, 0.0, ""
        
        elif isinstance(keyword, tuple):
            components = [comp.lower() for comp in keyword]
            positions = []
            
            for comp in components:
                pos = text_lower.find(comp)
                if pos == -1:
                    return False, 0.0, f"缺少組件: {comp}"
                positions.append(pos)
            
            distance = max(positions) - min(positions)
            
            if distance <= 80:
                return True, 0.9, f"近距離匹配({distance}字)"
            elif distance <= 200:
                return True, 0.8, f"中距離匹配({distance}字)"
            elif distance <= self.max_distance:
                return True, 0.6, f"遠距離匹配({distance}字)"
            else:
                return True, 0.4, f"極遠距離匹配({distance}字)"
        
        return False, 0.0, ""
    
    def _get_keyword_positions(self, text: str, keyword: Union[str, tuple]) -> List[Tuple[int, int]]:
        """獲取關鍵字位置"""
        positions = []
        
        if isinstance(keyword, str):
            keyword_lower = keyword.lower()
            start = 0
            while True:
                pos = text.find(keyword_lower, start)
                if pos == -1:
                    break
                positions.append((pos, pos + len(keyword_lower)))
                start = pos + 1
        
        elif isinstance(keyword, tuple):
            components = [comp.lower() for comp in keyword]
            component_positions = {}
            
            for comp in components:
                comp_positions = []
                start = 0
                while True:
                    pos = text.find(comp, start)
                    if pos == -1:
                        break
                    comp_positions.append((pos, pos + len(comp)))
                    start = pos + 1
                component_positions[comp] = comp_positions
            
            # 找到所有組件都在合理距離內的組合
            for comp1_pos in component_positions.get(components[0], []):
                for comp2_pos in component_positions.get(components[1], []):
                    distance = abs(comp1_pos[0] - comp2_pos[0])
                    if distance <= self.max_distance:
                        start_pos = min(comp1_pos[0], comp2_pos[0])
                        end_pos = max(comp1_pos[1], comp2_pos[1])
                        positions.append((start_pos, end_pos))
        
        return positions
    
    def _extract_numbers_in_window(self, window_text: str) -> List[str]:
        """提取數值"""
        numbers = []
        for pattern in self.number_patterns:
            matches = re.findall(pattern, window_text, re.IGNORECASE)
            numbers.extend(matches)
        return list(set(numbers))
    
    def _extract_percentages_in_window(self, window_text: str) -> List[str]:
        """提取百分比"""
        percentages = []
        for pattern in self.percentage_patterns:
            matches = re.findall(pattern, window_text, re.IGNORECASE)
            percentages.extend(matches)
        return list(set(percentages))
    
    def _calculate_association(self, text: str, keyword: Union[str, tuple], 
                             value: str, kw_start: int, kw_end: int, value_pos: int) -> float:
        """計算關鍵字與數值的關聯度"""
        
        # 距離因子
        distance = min(abs(value_pos - kw_start), abs(value_pos - kw_end))
        if distance <= 20:
            distance_score = 1.0
        elif distance <= 50:
            distance_score = 0.8
        elif distance <= 80:
            distance_score = 0.6
        else:
            distance_score = 0.3
        
        # 上下文相關性
        context_start = min(kw_start, value_pos) - 30
        context_end = max(kw_end, value_pos + len(value)) + 30
        context_start = max(0, context_start)
        context_end = min(len(text), context_end)
        context = text[context_start:context_end].lower()
        
        context_score = self._calculate_context_score(context)
        
        # 數值合理性
        value_score = self._calculate_value_score(value, context)
        
        final_score = (
            distance_score * 0.4 +
            context_score * 0.35 + 
            value_score * 0.25
        )
        
        return final_score
    
    def _is_excluded_content(self, text: str) -> bool:
        """檢查是否為排除內容"""
        for topic in self.config.EXCLUSION_RULES["exclude_topics"]:
            if topic in text:
                return True
        
        for context in self.config.EXCLUSION_RULES["exclude_contexts"]:
            if context in text:
                return True
        
        return False
    
    def _check_plastic_relevance(self, text: str) -> float:
        """檢查塑膠相關性"""
        plastic_score = 0.0
        recycling_score = 0.0
        
        plastic_count = 0
        for indicator in self.config.PLASTIC_INDICATORS["plastic_materials"]:
            if indicator in text:
                plastic_count += 1
        
        plastic_score = min(plastic_count / 3.0, 1.0)
        
        recycling_count = 0
        for indicator in self.config.PLASTIC_INDICATORS["recycling_specific"]:
            if indicator in text:
                recycling_count += 1
        
        recycling_score = min(recycling_count / 2.0, 1.0)
        
        if plastic_score > 0 and recycling_score > 0:
            return (plastic_score + recycling_score) / 2.0
        else:
            return 0.0
    
    def _calculate_relevance_score(self, text: str) -> float:
        """計算相關性分數"""
        relevance_indicators = {
            "plastic_materials": [
                "塑膠", "塑料", "聚酯", "PET", "PP", "聚合物",
                "樹脂", "粒子", "顆粒", "材料", "聚合物", "塑膠粒"
            ],
            "recycling_process": [
                "回收", "再生", "循環", "再利用", "回收利用",
                "造粒", "再製", "轉換", "處理", "循環經濟"
            ],
            "production_application": [
                "生產", "製造", "產能", "產量", "使用", "應用",
                "製成", "加工", "生產線", "工廠", "產品"
            ],
            "environmental_benefit": [
                "減碳", "碳排放", "環保", "永續", "節能", "減排",
                "碳足跡", "綠色", "低碳", "效益", "環境"
            ]
        }
        
        total_score = 0.0
        
        for category, indicators in relevance_indicators.items():
            category_score = 0.0
            for indicator in indicators:
                if indicator in text:
                    category_score += 1
            
            normalized_score = min(category_score / len(indicators), 1.0)
            total_score += normalized_score * 0.25
        
        return total_score
    
    def _calculate_context_score(self, context: str) -> float:
        """計算上下文分數"""
        high_relevance_words = [
            "回收", "再生", "循環", "製造", "生產", "產能", "使用",
            "塑膠", "塑料", "聚酯", "材料", "寶特瓶", "減碳", "效益"
        ]
        
        negative_words = [
            "災害", "事故", "馬拉松", "賽事", "改善案", "案例",
            "雨水", "節能", "隔熱", "鍋爐", "燃油"
        ]
        
        score = 0.0
        
        for word in high_relevance_words:
            if word in context:
                score += 0.15
        
        for word in negative_words:
            if word in context:
                score -= 0.2
        
        return max(0.0, min(1.0, score))
    
    def _calculate_value_score(self, value: str, context: str) -> float:
        """計算數值合理性分數"""
        number_match = re.search(r'\d+(?:,\d{3})*(?:\.\d+)?', value)
        if not number_match:
            return 0.0
        
        try:
            number_str = number_match.group().replace(',', '')
            number = float(number_str)
        except ValueError:
            return 0.0
        
        if "億支" in value:
            if 1 <= number <= 100:
                return 1.0
            elif 0.1 <= number <= 500:
                return 0.7
            else:
                return 0.3
        
        elif "萬噸" in value or "千噸" in value:
            if 0.1 <= number <= 50:
                return 1.0
            elif 0.01 <= number <= 100:
                return 0.7
            else:
                return 0.3
        
        elif "噸" in value:
            if 1 <= number <= 10000:
                return 1.0
            elif 0.1 <= number <= 50000:
                return 0.7
            else:
                return 0.3
        
        elif "%" in value or "％" in value:
            if 0 <= number <= 100:
                return 1.0
            else:
                return 0.2
        
        elif "件" in value:
            if 1 <= number <= 10000:
                return 1.0
            elif 1 <= number <= 100000:
                return 0.7
            else:
                return 0.3
        
        return 0.5

# =============================================================================
# ESG提取器主類
# =============================================================================

class ESGExtractor:
    """ESG報告書提取器主類"""
    
    def __init__(self, enable_llm: bool = True):
        self.enable_llm = enable_llm
        self.matcher = ESGMatcher()
        self.keyword_config = KeywordConfig()
        
        if self.enable_llm:
            self._init_llm()
        
        print("✅ ESG報告書提取器初始化完成")

    def _init_llm(self):
        """初始化LLM"""
        try:
            print("🤖 初始化Gemini API管理器...")
            self.api_manager = create_api_manager()
            print("✅ LLM初始化完成")
        except Exception as e:
            print(f"⚠️ LLM初始化失敗: {e}")
            self.enable_llm = False
    
    def process_single_document(self, doc_info: DocumentInfo, max_documents: int = 400) -> Tuple[List[NumericExtraction], ProcessingSummary, str]:
        """處理單個文檔"""
        start_time = datetime.now()
        print(f"\n📊 處理文檔: {doc_info.company_name} - {doc_info.report_year}")
        print("=" * 60)
        
        # 1. 載入向量資料庫
        db = self._load_vector_database(doc_info.db_path)
        
        # 2. 文檔檢索
        documents = self._document_retrieval(db, max_documents)
        
        # 3. 數據提取
        extractions = self._extract_data(documents, doc_info)
        
        # 4. 後處理
        extractions = self._post_process_extractions(extractions)
        
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
    
    def process_multiple_documents(self, docs_info: Dict[str, DocumentInfo], max_documents: int = 400) -> Dict[str, Tuple]:
        """批量處理多個文檔"""
        print(f"📊 開始批量處理 {len(docs_info)} 個文檔")
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
                import traceback
                traceback.print_exc()
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
    
    def _document_retrieval(self, db, max_docs: int) -> List[Document]:
        """文檔檢索"""
        keywords = self.keyword_config.get_all_keywords()
        all_docs = []
        
        # 關鍵字檢索
        print("   🔍 執行關鍵字檢索...")
        for keyword in keywords[:20]:
            search_term = keyword if isinstance(keyword, str) else " ".join(keyword)
            docs = db.similarity_search(search_term, k=15)
            all_docs.extend(docs)
        
        # 主題檢索
        print("   🔍 執行主題檢索...")
        topic_queries = [
            "塑膠 回收 材料",
            "寶特瓶 循環 經濟",
            "再生 環保 永續",
            "廢料 處理 利用"
        ]
        
        for query in topic_queries:
            docs = db.similarity_search(query, k=20)
            all_docs.extend(docs)
        
        # 數值檢索
        print("   🔍 執行數值檢索...")
        number_queries = [
            "億支", "萬噸", "千噸", "產能", "回收量",
            "減碳", "百分比", "效益", "數量"
        ]
        
        for query in number_queries:
            docs = db.similarity_search(query, k=10)
            all_docs.extend(docs)
        
        # 去重
        unique_docs = {}
        for doc in all_docs:
            doc_hash = hash(doc.page_content)
            if doc_hash not in unique_docs:
                unique_docs[doc_hash] = doc
        
        result_docs = list(unique_docs.values())[:max_docs]
        print(f"📚 檢索到 {len(result_docs)} 個候選文檔")
        return result_docs
    
    def _extract_data(self, documents: List[Document], doc_info: DocumentInfo) -> List[NumericExtraction]:
        """數據提取"""
        print("🎯 執行數據提取...")
        
        keywords = self.keyword_config.get_all_keywords()
        extractions = []
        
        for doc in tqdm(documents, desc="數據提取"):
            # 段落分割
            paragraphs = self._split_paragraphs(doc.page_content)
            page_num = doc.metadata.get('page', '未知')
            
            for para_idx, paragraph in enumerate(paragraphs):
                if len(paragraph.strip()) < 15:
                    continue
                
                # 對每個關鍵字進行匹配
                for keyword in keywords:
                    # 檢查相關性
                    is_relevant, relevance_score, details = self.matcher.comprehensive_relevance_check(paragraph, keyword)
                    
                    if is_relevant and relevance_score > 0.55:
                        # 提取數值配對
                        value_pairs = self.matcher.extract_keyword_value_pairs(paragraph, keyword)
                        
                        # 如果沒有找到數值但相關性很高，保留作為描述
                        if not value_pairs and relevance_score > 0.75:
                            keyword_str = keyword if isinstance(keyword, str) else " + ".join(keyword)
                            
                            extraction = NumericExtraction(
                                keyword=keyword_str,
                                value="[相關描述]",
                                value_type='description',
                                unit='',
                                paragraph=paragraph.strip(),
                                paragraph_number=para_idx + 1,
                                page_number=f"第{page_num}頁",
                                confidence=relevance_score,
                                context_window=self._get_context_window(doc.page_content, paragraph),
                                company_name=doc_info.company_name,
                                report_year=doc_info.report_year,
                                keyword_distance=0
                            )
                            extractions.append(extraction)
                        
                        # 處理找到的數值配對
                        for value, value_type, association_score, distance in value_pairs:
                            keyword_str = keyword if isinstance(keyword, str) else " + ".join(keyword)
                            
                            final_confidence = (relevance_score * 0.4 + association_score * 0.6)
                            
                            extraction = NumericExtraction(
                                keyword=keyword_str,
                                value=value,
                                value_type=value_type,
                                unit=self._extract_unit(value) if value_type == 'number' else '%',
                                paragraph=paragraph.strip(),
                                paragraph_number=para_idx + 1,
                                page_number=f"第{page_num}頁",
                                confidence=final_confidence,
                                context_window=self._get_context_window(doc.page_content, paragraph),
                                company_name=doc_info.company_name,
                                report_year=doc_info.report_year,
                                keyword_distance=distance
                            )
                            extractions.append(extraction)
        
        print(f"✅ 數據提取完成: 找到 {len(extractions)} 個結果")
        return extractions
    
    def _post_process_extractions(self, extractions: List[NumericExtraction]) -> List[NumericExtraction]:
        """後處理和去重"""
        if not extractions:
            return extractions
        
        print(f"🔧 後處理 {len(extractions)} 個提取結果...")
        
        # 精確去重
        unique_extractions = []
        seen_combinations = set()
        
        for extraction in extractions:
            identifier = (
                extraction.keyword,
                extraction.value,
                extraction.value_type,
                extraction.paragraph[:100]
            )
            
            if identifier not in seen_combinations:
                seen_combinations.add(identifier)
                unique_extractions.append(extraction)
        
        print(f"📊 去重後: {len(unique_extractions)} 個結果")
        
        # 頁面去重（每頁最多保留2筆）
        page_filtered_extractions = self._apply_per_page_filtering(unique_extractions, max_per_page=2)
        
        # 按信心分數排序
        page_filtered_extractions.sort(key=lambda x: x.confidence, reverse=True)
        
        print(f"✅ 後處理完成: 保留 {len(page_filtered_extractions)} 個最終結果")
        return page_filtered_extractions
    
    def _apply_per_page_filtering(self, extractions: List[NumericExtraction], max_per_page: int = 2) -> List[NumericExtraction]:
        """按頁面去重"""
        if not extractions:
            return extractions
        
        print(f"📄 執行按頁面去重（每頁最多保留 {max_per_page} 筆）...")
        
        # 按頁碼分組
        page_groups = {}
        for extraction in extractions:
            page_key = str(extraction.page_number).strip()
            if page_key not in page_groups:
                page_groups[page_key] = []
            page_groups[page_key].append(extraction)
        
        # 每頁面內按信心分數排序並保留最佳結果
        filtered_extractions = []
        
        for page_key, page_extractions in page_groups.items():
            page_extractions.sort(key=lambda x: x.confidence, reverse=True)
            kept_extractions = page_extractions[:max_per_page]
            filtered_extractions.extend(kept_extractions)
        
        print(f"   ✅ 頁面去重完成: {len(filtered_extractions)} 筆最終結果")
        return filtered_extractions
    
    def _export_to_excel(self, extractions: List[NumericExtraction], summary: ProcessingSummary, doc_info: DocumentInfo) -> str:
        """匯出結果到Excel"""
        company_safe = re.sub(r'[^\w\s-]', '', doc_info.company_name).strip()
        
        # 根據提取結果數量決定檔名
        if len(extractions) == 0:
            output_filename = f"ESG提取結果_無提取_{company_safe}_{doc_info.report_year}.xlsx"
            status_message = "無提取結果"
        else:
            output_filename = f"ESG提取結果_{company_safe}_{doc_info.report_year}.xlsx"
            status_message = f"提取結果: {len(extractions)} 項"
        
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
            '段落內容': f"{status_message}（ESG報告書提取器 v1.0）",
            '段落編號': '',
            '頁碼': '',
            '信心分數': '',
            '上下文': ''
        }
        main_data.append(header_row)
        
        # 空行分隔
        main_data.append({col: '' for col in header_row.keys()})
        
        # 如果有提取結果，添加結果數據
        if len(extractions) > 0:
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
        else:
            # 如果沒有提取結果，添加說明行
            no_result_row = {
                '關鍵字': '無相關關鍵字匹配',
                '提取數值': 'N/A',
                '數據類型': 'no_data',
                '單位': '',
                '段落內容': '在此份ESG報告中未找到再生塑膠相關的數值數據',
                '段落編號': '',
                '頁碼': '',
                '信心分數': 0.0,
                '上下文': '可能的原因：1) 該公司未涉及再生塑膠業務 2) 報告中未詳細披露相關數據 3) 關鍵字匹配範圍需要調整'
            }
            main_data.append(no_result_row)
        
        # 統計數據
        stats_data = []
        if len(extractions) > 0:
            for keyword, count in summary.keywords_found.items():
                keyword_extractions = [e for e in extractions if e.keyword == keyword]
                
                stats_data.append({
                    '關鍵字': keyword,
                    '提取數量': count,
                    '平均信心分數': round(np.mean([e.confidence for e in keyword_extractions]) if keyword_extractions else 0, 3),
                    '最高信心分數': round(max([e.confidence for e in keyword_extractions]) if keyword_extractions else 0, 3)
                })
        else:
            stats_data.append({
                '關鍵字': '搜尋摘要',
                '提取數量': 0,
                '平均信心分數': 0.0,
                '最高信心分數': 0.0
            })
        
        # 寫入Excel
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            # 主要結果工作表
            sheet_name = '提取結果' if len(extractions) > 0 else '無提取結果'
            pd.DataFrame(main_data).to_excel(writer, sheet_name=sheet_name, index=False)
            
            # 統計工作表
            pd.DataFrame(stats_data).to_excel(writer, sheet_name='統計摘要', index=False)
            
            # 處理摘要
            summary_data = [{
                '公司名稱': summary.company_name,
                '報告年度': summary.report_year,
                '總文檔數': summary.total_documents,
                '總提取結果': summary.total_extractions,
                '處理狀態': '成功提取' if len(extractions) > 0 else '無相關數據',
                '處理時間(秒)': round(summary.processing_time, 2),
                '處理日期': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                '提取器版本': 'ESG報告書提取器 v1.0'
            }]
            pd.DataFrame(summary_data).to_excel(writer, sheet_name='處理摘要', index=False)
        
        if len(extractions) > 0:
            print(f"✅ Excel檔案已保存，包含 {len(extractions)} 項提取結果")
        else:
            print(f"✅ Excel檔案已保存，標記為無提取結果")
        
        return output_path
    
    # 輔助方法
    def _split_paragraphs(self, text: str) -> List[str]:
        """段落分割"""
        paragraphs = []
        
        # 標準分割
        standard_paras = re.split(r'\n{2,}|\r{2,}', text)
        paragraphs.extend([p.strip() for p in standard_paras if len(p.strip()) >= 15])
        
        # 句號分割
        sentence_paras = re.split(r'。{2,}|\.{2,}', text)
        paragraphs.extend([p.strip() for p in sentence_paras if len(p.strip()) >= 30])
        
        # 保持原文
        if len(text.strip()) >= 50:
            paragraphs.append(text.strip())
        
        # 去重
        unique_paragraphs = []
        seen = set()
        for para in paragraphs:
            para_hash = hash(para[:100])
            if para_hash not in seen:
                seen.add(para_hash)
                unique_paragraphs.append(para)
        
        return unique_paragraphs
    
    def _extract_unit(self, value_str: str) -> str:
        """從數值字符串中提取單位"""
        units = re.findall(r'[a-zA-Z\u4e00-\u9fff]+', value_str)
        return units[-1] if units else ""
    
    def _get_context_window(self, full_text: str, target_paragraph: str, window_size: int = 150) -> str:
        """獲取段落的上下文窗口"""
        try:
            pos = full_text.find(target_paragraph)
            if pos == -1:
                return target_paragraph[:300]
            
            start = max(0, pos - window_size)
            end = min(len(full_text), pos + len(target_paragraph) + window_size)
            
            return full_text[start:end]
        except:
            return target_paragraph[:300]

def main():
    """主函數 - 測試用"""
    print("📊 ESG報告書提取器測試模式")
    
    extractor = ESGExtractor(enable_llm=False)
    print("✅ ESG報告書提取器初始化完成")

if __name__ == "__main__":
    main()