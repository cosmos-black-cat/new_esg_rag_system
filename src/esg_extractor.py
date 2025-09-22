#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ESG報告書提取器核心模組 v2.0
支持新關鍵字配置和Word文件輸出
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

# Word文件處理
from docx import Document as WordDocument
from docx.shared import Inches, Cm, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.shared import OxmlElement, qn

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
    stock_code: str = ""  # 新增股票代號

@dataclass
class NumericExtraction:
    """數值提取結果"""
    keyword: str
    value: str
    value_type: str  # 'number', 'percentage', 'description'
    unit: str
    paragraph: str
    paragraph_number: int
    page_number: str
    confidence: float
    context_window: str
    company_name: str = ""
    report_year: str = ""
    stock_code: str = ""  # 新增股票代號
    keyword_distance: int = 0
    full_section: str = ""  # 新增完整段落內容

@dataclass
class ProcessingSummary:
    """處理摘要"""
    company_name: str
    report_year: str
    stock_code: str
    total_documents: int
    stage1_passed: int
    stage2_passed: int
    total_extractions: int
    keywords_found: Dict[str, int]
    processing_time: float

# =============================================================================
# 新關鍵字配置
# =============================================================================

class EnhancedKeywordConfig:
    """增強的關鍵字配置 - 支持新的ESG關鍵字"""
    
    # 組1：比率類關鍵字（通常與百分比數值關聯）
    RATIO_KEYWORDS = {
        "high_relevance_continuous": [
            "材料循環率", "材料可回收率", "再生能源使用率", 
            "單位經濟效益", "再生材料替代率", "碳排減量比率", "再生塑膠使用比率",
            "回收利用率", "資源循環率", "綠色材料比率", "環保材料占比",
            "再生料使用率", "循環經濟效率", "廢棄物資源化比例"
        ],
        
        "medium_relevance_continuous": [
            "使用率", "替代率", "回收率", "利用率", "減量比率",
            "循環率", "效益比", "占比", "比例", "效率"
        ],
        
        "discontinuous": [
            ("材料", "循環率"), ("材料", "可回收率"), ("再生能源", "使用率"),
            ("再生材料", "替代率"), ("碳排", "減量", "比率"), ("再生塑膠", "使用", "比率"),
            ("回收", "利用率"), ("資源", "循環率"), ("廢棄物", "資源化", "比例")
        ]
    }
    
    # 組2：數量類關鍵字（通常與數值單位關聯）
    QUANTITY_KEYWORDS = {
        "high_relevance_continuous": [
            "再生材料使用量", "材料總使用量", "綠電憑證", "太陽能電力",
            "購電協議", "再生能源", "再生材料碳排減量", "再生塑膠成本",
            "再生塑膠的使用量", "成本增加", "材料回收", "材質分離",
            "碳排放", "塑膠使用量", "材料使用量", "回收處理量", "廢料回收量",
            "寶特瓶回收量", "循環材料使用量", "環保材料使用量"
        ],
        
        "medium_relevance_continuous": [
            "使用量", "處理量", "回收量", "減量", "產量", "消耗量",
            "投入量", "產出量", "節約量", "替代量", "循環量"
        ],
        
        "discontinuous": [
            ("再生材料", "使用量"), ("材料", "總使用量"), ("綠電", "憑證"),
            ("太陽能", "電力"), ("購電", "協議"), ("再生", "能源"),
            ("再生材料", "碳排", "減量"), ("再生塑膠", "成本"), ("再生塑膠", "使用量"),
            ("材料", "回收"), ("材質", "分離"), ("碳", "排放"),
            ("塑膠", "使用量"), ("材料", "使用量"), ("MLCC", "回收"),
            ("寶特瓶", "回收"), ("廢料", "處理"), ("循環", "材料")
        ]
    }
    
    # 特殊關鍵字（分選辨視、單一材料等）
    SPECIAL_KEYWORDS = {
        "process_related": [
            "分選辨視", "單一材料", "材料純度", "品質控制", "分類處理",
            "材料識別", "自動分選", "人工智能分選", "光學分選", "密度分選"
        ],
        
        "technology_related": [
            "回收技術", "處理工藝", "循環技術", "再生工藝", "分離技術",
            "純化技術", "改質技術", "造粒技術", "熱解技術"
        ]
    }
    
    # 排除規則 - 更精確
    EXCLUSION_RULES = {
        "exclude_topics": [
            "職業災害", "工安", "安全事故", "職災", "員工傷亡",
            "馬拉松", "賽事", "選手", "比賽", "賽衣", "運動", "體育活動",
            "廢水處理", "水質監測", "污水處理", "水處理系統",
            "節能改善案", "改善專案", "案例選拔", "優良案例", "表揚大會",
            "鍋爐改善", "天然氣燃燒", "燃油改燃", "設備改善", "機台更新"
        ],
        
        "exclude_contexts": [
            "垂直馬拉松", "史上最環保賽衣", "各界好手", "參與盛會",
            "職業災害比率", "工安統計", "安全指標", "事故率",
            "節能改善案", "節水改善案", "優良案例選拔", "績優部門表揚",
            "雨水回收量減少", "降雨量減少", "月平均降雨", "氣象資料",
            "燃油改燃汽鍋爐", "天然氣燃燒機", "鍋爐改造", "設備汰換",
            "監測次數", "檢測頻率", "執行次數", "查核場次"
        ],
        
        "exclude_number_patterns": [
            r'職業災害.*?\d+(?:\.\d+)?%',
            r'災害比率.*?\d+(?:\.\d+)?%',
            r'降雨量.*?\d+(?:\.\d+)?%',
            r'雨水.*?\d+(?:\.\d+)?噸/日',
            r'\d+\s*件.*?改善案',
            r'改善案.*?\d+\s*件',
            r'\d+\s*次.*?監測',
            r'監測.*?\d+\s*次'
        ]
    }
    
    @classmethod
    def get_all_keywords(cls) -> List[Union[str, tuple]]:
        """獲取所有關鍵字"""
        all_keywords = []
        
        # 比率類關鍵字
        all_keywords.extend(cls.RATIO_KEYWORDS["high_relevance_continuous"])
        all_keywords.extend(cls.RATIO_KEYWORDS["medium_relevance_continuous"])
        all_keywords.extend(cls.RATIO_KEYWORDS["discontinuous"])
        
        # 數量類關鍵字
        all_keywords.extend(cls.QUANTITY_KEYWORDS["high_relevance_continuous"])
        all_keywords.extend(cls.QUANTITY_KEYWORDS["medium_relevance_continuous"])
        all_keywords.extend(cls.QUANTITY_KEYWORDS["discontinuous"])
        
        # 特殊關鍵字
        all_keywords.extend(cls.SPECIAL_KEYWORDS["process_related"])
        all_keywords.extend(cls.SPECIAL_KEYWORDS["technology_related"])
        
        return all_keywords

# =============================================================================
# 股票代號識別器
# =============================================================================

class StockCodeExtractor:
    """股票代號提取器"""
    
    def __init__(self):
        # 台灣股票代號模式
        self.stock_patterns = [
            r'股票代號[：:]\s*(\d{4})',
            r'代號[：:]\s*(\d{4})',
            r'證券代號[：:]\s*(\d{4})',
            r'上市代號[：:]\s*(\d{4})',
            r'股份代號[：:]\s*(\d{4})',
            r'公司代號[：:]\s*(\d{4})',
            r'統一編號[：:]\s*(\d{8})',  # 統一編號作為備用
        ]
        
        # 常見公司代號映射（可手動維護）
        self.known_mappings = {
            "台積電": "2330",
            "南亞": "1303", 
            "台塑": "1301",
            "聯電": "2303",
            "鴻海": "2317",
            "台化": "1326",
            "中油": "台灣中油公司"
        }
    
    def extract_stock_code(self, text: str, company_name: str) -> str:
        """提取股票代號"""
        
        # 方法1：從文本中直接提取
        for pattern in self.stock_patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(1)
        
        # 方法2：從已知映射中查找
        for known_company, code in self.known_mappings.items():
            if known_company in company_name:
                return code
        
        # 方法3：從公司名稱中推測（啟發式）
        if "台積電" in company_name or "TSMC" in company_name.upper():
            return "2330"
        elif "南亞" in company_name:
            return "1303"
        elif "台塑" in company_name and "南亞" not in company_name:
            return "1301"
        elif "台化" in company_name:
            return "1326"
        
        return ""  # 無法識別時返回空字符串

# =============================================================================
# 增強的匹配引擎
# =============================================================================

class EnhancedESGMatcher:
    """增強的ESG數據匹配引擎"""
    
    def __init__(self):
        self.config = EnhancedKeywordConfig()
        self.max_distance = 200  # 增加搜索距離
        
        # 更全面的數值匹配模式
        self.number_patterns = [
            # 基本數值模式
            r'\d+(?:,\d{3})*(?:\.\d+)?\s*(?:萬|千)?(?:噸|公斤|kg|KG)',
            r'\d+(?:,\d{3})*(?:\.\d+)?\s*(?:億|萬|千)?\s*(?:支|件|個|台|套|筆)',
            r'\d+(?:,\d{3})*(?:\.\d+)?\s*(?:噸/月|噸/年|噸/日|kg/月|kg/年)',
            
            # 能源相關
            r'\d+(?:,\d{3})*(?:\.\d+)?\s*(?:kWh|MWh|GWh|度)',
            r'\d+(?:,\d{3})*(?:\.\d+)?\s*(?:kW|MW|GW)',
            r'\d+(?:,\d{3})*(?:\.\d+)?\s*張.*?憑證',
            
            # 成本相關
            r'\d+(?:,\d{3})*(?:\.\d+)?\s*(?:萬|億)?(?:元|新台幣|TWD)',
            
            # 一般數值
            r'\d+(?:,\d{3})*(?:\.\d+)?'
        ]
        
        # 百分比模式
        self.percentage_patterns = [
            r'\d+(?:\.\d+)?\s*%',
            r'\d+(?:\.\d+)?\s*％',
            r'百分之\d+(?:\.\d+)?',
            r'\d+(?:\.\d+)?\s*個百分點'
        ]
        
        # 比率模式
        self.ratio_patterns = [
            r'\d+(?:\.\d+)?\s*:\s*\d+(?:\.\d+)?',
            r'\d+(?:\.\d+)?\s*比\s*\d+(?:\.\d+)?',
            r'\d+(?:\.\d+)?\s*倍'
        ]
    
    def comprehensive_relevance_check(self, text: str, keyword: Union[str, tuple]) -> Tuple[bool, float, str]:
        """綜合相關性檢查 - 增強版"""
        text_lower = text.lower()
        
        # 1. 強排除檢查
        if self._is_strongly_excluded(text_lower):
            return False, 0.0, "強排除內容"
        
        # 2. 關鍵字匹配檢查  
        keyword_match, keyword_confidence, keyword_details = self._match_keyword(text, keyword)
        if not keyword_match:
            return False, 0.0, "關鍵字不匹配"
        
        # 3. 上下文相關性檢查
        context_score = self._enhanced_context_check(text_lower, keyword)
        
        # 4. 數值相關性檢查
        value_score = self._enhanced_value_check(text_lower, keyword)
        
        # 5. 語義一致性檢查
        semantic_score = self._semantic_consistency_check(text_lower, keyword)
        
        # 計算綜合分數
        final_score = (
            keyword_confidence * 0.25 +
            context_score * 0.30 +
            value_score * 0.25 +
            semantic_score * 0.20
        )
        
        # 動態閾值
        threshold = self._get_dynamic_threshold(keyword)
        is_relevant = final_score >= threshold
        
        details = f"關鍵字:{keyword_confidence:.2f}, 上下文:{context_score:.2f}, 數值:{value_score:.2f}, 語義:{semantic_score:.2f}"
        
        return is_relevant, final_score, details
    
    def extract_keyword_value_pairs(self, text: str, keyword: Union[str, tuple]) -> List[Tuple[str, str, float, int]]:
        """提取關鍵字與數值的配對 - 增強版"""
        text_lower = text.lower()
        
        # 1. 檢查關鍵字是否存在
        keyword_match, keyword_confidence, _ = self._match_keyword(text, keyword)
        if not keyword_match:
            return []
        
        # 2. 找到關鍵字位置
        keyword_positions = self._get_keyword_positions(text_lower, keyword)
        if not keyword_positions:
            return []
        
        # 3. 智能數值搜索
        valid_pairs = []
        
        for kw_start, kw_end in keyword_positions:
            # 動態搜索窗口
            search_window_size = self._get_search_window_size(keyword)
            search_start = max(0, kw_start - search_window_size)
            search_end = min(len(text), kw_end + search_window_size)
            search_window = text[search_start:search_end]
            
            # 提取不同類型的數值
            all_values = []
            
            # 數值
            numbers = self._extract_enhanced_numbers(search_window)
            for num in numbers:
                all_values.append((num, 'number'))
            
            # 百分比
            percentages = self._extract_enhanced_percentages(search_window)  
            for pct in percentages:
                all_values.append((pct, 'percentage'))
            
            # 比率
            ratios = self._extract_ratios(search_window)
            for ratio in ratios:
                all_values.append((ratio, 'ratio'))
            
            # 驗證每個數值
            for value, value_type in all_values:
                value_pos = search_window.find(value)
                if value_pos != -1:
                    actual_value_pos = search_start + value_pos
                    distance = min(abs(actual_value_pos - kw_start), abs(actual_value_pos - kw_end))
                    
                    # 計算關聯分數
                    association_score = self._calculate_enhanced_association(
                        text, keyword, value, value_type, kw_start, kw_end, actual_value_pos
                    )
                    
                    if association_score > 0.5 and distance <= search_window_size:
                        valid_pairs.append((value, value_type, association_score, distance))
        
        # 去重和排序
        unique_pairs = self._deduplicate_value_pairs(valid_pairs)
        
        return unique_pairs[:5]  # 返回最多5個最佳結果
    
    # 輔助方法實現
    def _is_strongly_excluded(self, text: str) -> bool:
        """強排除檢查"""
        # 檢查排除主題
        for topic in self.config.EXCLUSION_RULES["exclude_topics"]:
            if topic in text:
                return True
        
        # 檢查排除上下文
        for context in self.config.EXCLUSION_RULES["exclude_contexts"]:
            if context in text:
                return True
        
        # 檢查排除數值模式
        for pattern in self.config.EXCLUSION_RULES["exclude_number_patterns"]:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        
        return False
    
    def _enhanced_context_check(self, text: str, keyword: Union[str, tuple]) -> float:
        """增強的上下文檢查"""
        context_indicators = {
            "material_related": ["材料", "物料", "原料", "塑膠", "塑料", "聚合物", "樹脂"],
            "recycling_related": ["回收", "再生", "循環", "再利用", "重複使用", "廢料"],
            "sustainability_related": ["永續", "環保", "綠色", "減碳", "節能", "ESG", "循環經濟"],
            "quantitative_related": ["使用", "消耗", "產生", "處理", "製造", "生產", "應用"],
            "performance_related": ["效率", "效益", "比率", "比例", "成效", "績效", "改善"]
        }
        
        total_score = 0.0
        category_weights = {
            "material_related": 0.25,
            "recycling_related": 0.25,
            "sustainability_related": 0.20,
            "quantitative_related": 0.15,
            "performance_related": 0.15
        }
        
        for category, indicators in context_indicators.items():
            category_score = 0.0
            for indicator in indicators:
                if indicator in text:
                    category_score += 1.0
            
            # 正規化分數
            normalized_score = min(category_score / len(indicators), 1.0)
            total_score += normalized_score * category_weights[category]
        
        return total_score
    
    def _enhanced_value_check(self, text: str, keyword: Union[str, tuple]) -> float:
        """增強的數值檢查"""
        # 尋找數值
        all_numbers = re.findall(r'\d+(?:,\d{3})*(?:\.\d+)?', text)
        all_percentages = re.findall(r'\d+(?:\.\d+)?\s*%', text)
        
        if not all_numbers and not all_percentages:
            return 0.0
        
        value_score = 0.0
        
        # 檢查數值合理性和相關性
        keyword_str = keyword if isinstance(keyword, str) else " ".join(keyword)
        
        # 根據關鍵字類型調整期望的數值類型
        expected_value_type = self._get_expected_value_type(keyword_str)
        
        if expected_value_type == "percentage" and all_percentages:
            value_score += 0.6
        elif expected_value_type == "quantity" and all_numbers:
            value_score += 0.6
        elif expected_value_type == "mixed" and (all_numbers or all_percentages):
            value_score += 0.5
        
        # 額外加分：數值在合理範圍內
        if all_numbers:
            for num_str in all_numbers[:3]:  # 只檢查前3個數值
                try:
                    num = float(num_str.replace(',', ''))
                    if self._is_reasonable_value(num, keyword_str):
                        value_score += 0.1
                except:
                    continue
        
        return min(value_score, 1.0)
    
    def _semantic_consistency_check(self, text: str, keyword: Union[str, tuple]) -> float:
        """語義一致性檢查"""
        keyword_str = keyword if isinstance(keyword, str) else " ".join(keyword)
        
        # 檢查語義一致性指標
        consistency_score = 0.0
        
        # 檢查動詞一致性
        if any(verb in text for verb in ["使用", "應用", "採用", "實施"]):
            if any(term in keyword_str for term in ["使用", "應用"]):
                consistency_score += 0.3
        
        # 檢查量詞一致性  
        if any(measure in text for measure in ["噸", "公斤", "件", "張"]):
            if "量" in keyword_str or "使用" in keyword_str:
                consistency_score += 0.3
        
        # 檢查比率一致性
        if any(ratio in text for ratio in ["%", "％", "比率", "比例"]):
            if any(term in keyword_str for term in ["率", "比", "效率"]):
                consistency_score += 0.4
        
        return min(consistency_score, 1.0)
    
    def _get_dynamic_threshold(self, keyword: Union[str, tuple]) -> float:
        """動態閾值計算"""
        keyword_str = keyword if isinstance(keyword, str) else " ".join(keyword)
        
        # 高優先級關鍵字較低閾值
        if any(high_priority in keyword_str for high_priority in 
               ["再生材料", "循環率", "回收率", "碳排減量", "使用量"]):
            return 0.6
        
        # 中優先級關鍵字中等閾值
        elif any(mid_priority in keyword_str for mid_priority in 
                 ["材料", "能源", "效益", "成本"]):
            return 0.65
        
        # 一般關鍵字較高閾值
        else:
            return 0.7
    
    def _extract_enhanced_numbers(self, text: str) -> List[str]:
        """增強的數值提取"""
        numbers = []
        
        for pattern in self.number_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            numbers.extend(matches)
        
        # 去重
        return list(set(numbers))
    
    def _extract_enhanced_percentages(self, text: str) -> List[str]:
        """增強的百分比提取"""
        percentages = []
        
        for pattern in self.percentage_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            percentages.extend(matches)
        
        return list(set(percentages))
    
    def _extract_ratios(self, text: str) -> List[str]:
        """提取比率"""
        ratios = []
        
        for pattern in self.ratio_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            ratios.extend(matches)
        
        return list(set(ratios))
    
    def _get_expected_value_type(self, keyword: str) -> str:
        """根據關鍵字推測期望的數值類型"""
        if any(ratio_term in keyword for ratio_term in ["率", "比", "%", "效率", "比例"]):
            return "percentage"
        elif any(quantity_term in keyword for quantity_term in ["量", "使用", "產生", "處理", "成本"]):
            return "quantity"
        else:
            return "mixed"
    
    def _is_reasonable_value(self, value: float, keyword: str) -> bool:
        """檢查數值合理性"""
        # 根據關鍵字類型檢查數值範圍
        if "率" in keyword or "比" in keyword:
            return 0 <= value <= 100  # 比率通常0-100%
        elif "使用量" in keyword or "處理量" in keyword:
            return 0 < value < 1000000  # 使用量應為正數且合理
        elif "成本" in keyword:
            return 0 < value  # 成本應為正數
        else:
            return value >= 0  # 一般情況下非負數
    
    def _get_search_window_size(self, keyword: Union[str, tuple]) -> int:
        """動態搜索窗口大小"""
        keyword_str = keyword if isinstance(keyword, str) else " ".join(keyword)
        
        # 複雜關鍵字需要更大的搜索窗口
        if isinstance(keyword, tuple) or len(keyword_str) > 10:
            return 150
        else:
            return 100
    
    def _calculate_enhanced_association(self, text: str, keyword: Union[str, tuple], 
                                      value: str, value_type: str,
                                      kw_start: int, kw_end: int, value_pos: int) -> float:
        """增強的關聯度計算"""
        
        # 距離因子
        distance = min(abs(value_pos - kw_start), abs(value_pos - kw_end))
        distance_score = max(0, 1.0 - distance / 100.0)
        
        # 類型匹配分數
        keyword_str = keyword if isinstance(keyword, str) else " ".join(keyword)
        expected_type = self._get_expected_value_type(keyword_str)
        
        type_score = 1.0
        if expected_type == "percentage" and value_type != "percentage":
            type_score = 0.7
        elif expected_type == "quantity" and value_type == "percentage":
            type_score = 0.8
        
        # 上下文分數
        context_start = min(kw_start, value_pos) - 50
        context_end = max(kw_end, value_pos + len(value)) + 50
        context_start = max(0, context_start)
        context_end = min(len(text), context_end)
        context = text[context_start:context_end].lower()
        
        context_score = self._calculate_local_context_score(context, keyword_str, value_type)
        
        # 綜合分數
        final_score = (
            distance_score * 0.4 +
            type_score * 0.3 +
            context_score * 0.3
        )
        
        return final_score
    
    def _calculate_local_context_score(self, context: str, keyword: str, value_type: str) -> float:
        """計算局部上下文分數"""
        score = 0.0
        
        # 檢查支持性詞彙
        support_words = ["為", "達", "約", "共", "總計", "合計", "達到", "實現"]
        for word in support_words:
            if word in context:
                score += 0.1
        
        # 檢查相關動詞
        relevant_verbs = ["使用", "產生", "處理", "回收", "再生", "循環", "替代", "減少", "增加"]
        for verb in relevant_verbs:
            if verb in context:
                score += 0.15
        
        # 根據數值類型調整
        if value_type == "percentage":
            if any(pct_word in context for pct_word in ["提升", "改善", "增長", "下降", "減少"]):
                score += 0.2
        
        return min(score, 1.0)
    
    def _deduplicate_value_pairs(self, pairs: List[Tuple[str, str, float, int]]) -> List[Tuple[str, str, float, int]]:
        """去重數值配對"""
        unique_pairs = []
        seen_values = set()
        
        # 按關聯分數排序
        sorted_pairs = sorted(pairs, key=lambda x: x[2], reverse=True)
        
        for value, value_type, score, distance in sorted_pairs:
            # 標準化數值用於去重比較
            normalized_value = re.sub(r'\s+', '', value.lower())
            
            if normalized_value not in seen_values:
                seen_values.add(normalized_value)
                unique_pairs.append((value, value_type, score, distance))
        
        return unique_pairs
    
    # 其他必要的輔助方法保持不變
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
            elif distance <= 150:
                return True, 0.8, f"中距離匹配({distance}字)"
            elif distance <= self.max_distance:
                return True, 0.7, f"遠距離匹配({distance}字)"
            else:
                return True, 0.5, f"極遠距離匹配({distance}字)"
        
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

# =============================================================================
# Word文件輸出器
# =============================================================================

class WordDocumentExporter:
    """Word文檔輸出器"""
    
    def __init__(self):
        self.font_name = "標楷體"
        self.font_size_title = 16
        self.font_size_heading = 14
        self.font_size_body = 12
    
    def create_word_document(self, extractions: List[NumericExtraction], 
                           doc_info: DocumentInfo, summary: ProcessingSummary) -> str:
        """創建Word文檔"""
        
        # 創建文檔
        doc = WordDocument()
        
        # 設置頁面格式
        sections = doc.sections
        for section in sections:
            section.page_height = Cm(29.7)  # A4高度
            section.page_width = Cm(21.0)   # A4寬度
            section.left_margin = Cm(2.5)
            section.right_margin = Cm(2.5)
            section.top_margin = Cm(2.5)
            section.bottom_margin = Cm(2.5)
        
        # 標題
        title = f"{doc_info.stock_code}_{doc_info.company_name}_{doc_info.report_year}_提取統整"
        title_para = doc.add_heading(title, level=0)
        title_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
        
        # 摘要信息
        summary_para = doc.add_paragraph()
        summary_para.add_run(f"提取日期：{datetime.now().strftime('%Y年%m月%d日')}\n").bold = True
        summary_para.add_run(f"提取結果總數：{len(extractions)} 項\n").bold = True  
        summary_para.add_run(f"處理時間：{summary.processing_time:.2f} 秒\n").bold = True
        
        # 添加分隔線
        doc.add_paragraph("=" * 60)
        
        if not extractions:
            # 如果沒有提取結果
            no_data_para = doc.add_paragraph()
            no_data_para.add_run("未找到相關數據").bold = True
            no_data_para.add_run("\n\n可能的原因：\n")
            no_data_para.add_run("1. 該公司ESG報告中未包含相關的再生材料或循環經濟數據\n")
            no_data_para.add_run("2. 相關數據存在但關鍵字匹配未能識別\n")  
            no_data_para.add_run("3. 數據格式特殊，需要調整提取規則\n")
        else:
            # 按頁碼排序
            sorted_extractions = sorted(extractions, key=lambda x: (x.page_number, x.paragraph_number))
            
            for i, extraction in enumerate(sorted_extractions, 1):
                # 每個提取結果作為一個段落組
                
                # 頁碼
                page_para = doc.add_paragraph()
                page_run = page_para.add_run(f"頁碼：{extraction.page_number}")
                page_run.bold = True
                page_run.font.size = 14
                
                # 關鍵字
                keyword_para = doc.add_paragraph()
                keyword_run = keyword_para.add_run(f"關鍵字：{extraction.keyword}")
                keyword_run.bold = True
                
                # 數值
                value_para = doc.add_paragraph()
                if extraction.value == "[相關描述]":
                    value_run = value_para.add_run("數值：相關描述內容（無具體數值）")
                else:
                    value_run = value_para.add_run(f"數值：{extraction.value} {extraction.unit}")
                value_run.font.color.rgb = RGBColor(0, 0, 255)  # 藍色
                
                # 信心分數
                confidence_para = doc.add_paragraph()
                confidence_run = confidence_para.add_run(f"信心分數：{extraction.confidence:.3f}")
                if extraction.confidence >= 0.8:
                    confidence_run.font.color.rgb = RGBColor(0, 128, 0)  # 綠色
                elif extraction.confidence >= 0.6:
                    confidence_run.font.color.rgb = RGBColor(255, 165, 0)  # 橙色
                else:
                    confidence_run.font.color.rgb = RGBColor(255, 0, 0)  # 紅色
                
                # 整個段落內容
                content_para = doc.add_paragraph()
                content_run = content_para.add_run("整個段落內容：")
                content_run.bold = True
                
                # 段落內容（縮排顯示）
                paragraph_content = extraction.full_section if extraction.full_section else extraction.paragraph
                content_detail_para = doc.add_paragraph(paragraph_content)
                content_detail_para.style.paragraph_format.left_indent = Cm(1)  # 縮排1公分
                
                # 分隔線（除了最後一項）
                if i < len(sorted_extractions):
                    doc.add_paragraph("-" * 80)
        
        # 保存文檔
        safe_company = re.sub(r'[^\w\s-]', '', doc_info.company_name).strip()
        filename = f"{doc_info.stock_code}_{safe_company}_{doc_info.report_year}_提取統整.docx"
        output_path = os.path.join(RESULTS_PATH, filename)
        
        doc.save(output_path)
        
        return output_path

# =============================================================================
# 主提取器類更新
# =============================================================================

class EnhancedESGExtractor:
    """增強的ESG報告書提取器"""
    
    def __init__(self, enable_llm: bool = True):
        self.enable_llm = enable_llm
        self.matcher = EnhancedESGMatcher()
        self.keyword_config = EnhancedKeywordConfig()
        self.stock_extractor = StockCodeExtractor()
        self.word_exporter = WordDocumentExporter()
        
        if self.enable_llm:
            self._init_llm()
        
        print("✅ 增強版ESG報告書提取器初始化完成")
    
    def _init_llm(self):
        """初始化LLM"""
        try:
            print("🤖 初始化Gemini API管理器...")
            self.api_manager = create_api_manager()
            print("✅ LLM初始化完成")
        except Exception as e:
            print(f"⚠️ LLM初始化失敗: {e}")
            self.enable_llm = False
    
    def process_single_document(self, doc_info: DocumentInfo, max_documents: int = 400) -> Tuple[List[NumericExtraction], ProcessingSummary, str, str]:
        """處理單個文檔 - 返回Excel和Word文件路徑"""
        start_time = datetime.now()
        print(f"\n📊 處理文檔: {doc_info.company_name} - {doc_info.report_year}")
        print("=" * 60)
        
        # 1. 載入向量資料庫
        db = self._load_vector_database(doc_info.db_path)
        
        # 2. 文檔檢索（使用新關鍵字）
        documents = self._enhanced_document_retrieval(db, max_documents)
        
        # 3. 股票代號識別
        if not doc_info.stock_code:
            stock_code = self._extract_stock_code_from_documents(documents, doc_info.company_name)
            doc_info.stock_code = stock_code
        
        # 4. 數據提取
        extractions = self._enhanced_extract_data(documents, doc_info)
        
        # 5. 後處理
        extractions = self._enhanced_post_process(extractions)
        
        # 6. 創建處理摘要
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        
        keywords_found = {}
        for extraction in extractions:
            keyword = extraction.keyword
            keywords_found[keyword] = keywords_found.get(keyword, 0) + 1
        
        summary = ProcessingSummary(
            company_name=doc_info.company_name,
            report_year=doc_info.report_year,
            stock_code=doc_info.stock_code,
            total_documents=len(documents),
            stage1_passed=len(documents),
            stage2_passed=len(extractions),
            total_extractions=len(extractions),
            keywords_found=keywords_found,
            processing_time=processing_time
        )
        
        # 7. 匯出結果到Excel
        excel_path = self._export_to_excel(extractions, summary, doc_info)
        
        # 8. 匯出結果到Word (新增)
        word_path = self._export_to_word(extractions, summary, doc_info)
        
        return extractions, summary, excel_path, word_path
    
    def _enhanced_document_retrieval(self, db, max_docs: int) -> List[Document]:
        """增強的文檔檢索 - 使用新關鍵字"""
        print("   🔍 執行增強關鍵字檢索...")
        
        all_docs = []
        keywords = self.keyword_config.get_all_keywords()
        
        # 分組檢索
        ratio_keywords = (self.keyword_config.RATIO_KEYWORDS["high_relevance_continuous"] + 
                         self.keyword_config.RATIO_KEYWORDS["medium_relevance_continuous"])
        quantity_keywords = (self.keyword_config.QUANTITY_KEYWORDS["high_relevance_continuous"] +
                           self.keyword_config.QUANTITY_KEYWORDS["medium_relevance_continuous"])
        
        # 比率類關鍵字檢索
        for keyword in ratio_keywords[:15]:
            search_term = keyword
            docs = db.similarity_search(search_term, k=12)
            all_docs.extend(docs)
        
        # 數量類關鍵字檢索
        for keyword in quantity_keywords[:15]:
            search_term = keyword
            docs = db.similarity_search(search_term, k=12)
            all_docs.extend(docs)
        
        # 特殊關鍵字檢索
        special_keywords = (self.keyword_config.SPECIAL_KEYWORDS["process_related"] +
                          self.keyword_config.SPECIAL_KEYWORDS["technology_related"])
        for keyword in special_keywords[:10]:
            docs = db.similarity_search(keyword, k=8)
            all_docs.extend(docs)
        
        # 主題檢索（更新主題）
        topic_queries = [
            "材料循環 回收利用",
            "再生能源 使用量",
            "碳排放 減量 效益",
            "綠電憑證 太陽能",
            "材料回收 分選技術"
        ]
        
        for query in topic_queries:
            docs = db.similarity_search(query, k=15)
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
    
    def _extract_stock_code_from_documents(self, documents: List[Document], company_name: str) -> str:
        """從文檔中提取股票代號"""
        print(f"   🔍 提取股票代號...")
        
        # 檢查前幾個文檔的內容
        for doc in documents[:5]:
            stock_code = self.stock_extractor.extract_stock_code(doc.page_content, company_name)
            if stock_code:
                print(f"   ✅ 找到股票代號: {stock_code}")
                return stock_code
        
        # 如果沒找到，使用公司名稱推測
        stock_code = self.stock_extractor.extract_stock_code("", company_name)
        if stock_code:
            print(f"   ✅ 推測股票代號: {stock_code}")
        else:
            print(f"   ⚠️ 無法識別股票代號")
            
        return stock_code
    
    def _enhanced_extract_data(self, documents: List[Document], doc_info: DocumentInfo) -> List[NumericExtraction]:
        """增強的數據提取"""
        print("🎯 執行增強數據提取...")
        
        keywords = self.keyword_config.get_all_keywords()
        extractions = []
        
        for doc in tqdm(documents, desc="增強數據提取"):
            # 段落分割 - 更智能的分割
            paragraphs = self._enhanced_split_paragraphs(doc.page_content)
            page_num = doc.metadata.get('page', '未知')
            
            for para_idx, paragraph in enumerate(paragraphs):
                if len(paragraph.strip()) < 20:  # 提高最小長度要求
                    continue
                
                # 對每個關鍵字進行匹配
                for keyword in keywords:
                    # 檢查相關性 - 使用增強匹配器
                    is_relevant, relevance_score, details = self.matcher.comprehensive_relevance_check(paragraph, keyword)
                    
                    if is_relevant and relevance_score > 0.6:  # 稍微降低閾值
                        # 提取數值配對
                        value_pairs = self.matcher.extract_keyword_value_pairs(paragraph, keyword)
                        
                        # 如果沒有找到數值但相關性很高，保留作為描述
                        if not value_pairs and relevance_score > 0.8:
                            keyword_str = keyword if isinstance(keyword, str) else " + ".join(keyword)
                            
                            # 獲取完整段落內容
                            full_section = self._get_full_section(doc.page_content, paragraph)
                            
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
                                stock_code=doc_info.stock_code,
                                keyword_distance=0,
                                full_section=full_section
                            )
                            extractions.append(extraction)
                        
                        # 處理找到的數值配對
                        for value, value_type, association_score, distance in value_pairs:
                            keyword_str = keyword if isinstance(keyword, str) else " + ".join(keyword)
                            
                            final_confidence = (relevance_score * 0.4 + association_score * 0.6)
                            
                            # 獲取完整段落內容
                            full_section = self._get_full_section(doc.page_content, paragraph)
                            
                            extraction = NumericExtraction(
                                keyword=keyword_str,
                                value=value,
                                value_type=value_type,
                                unit=self._extract_unit(value) if value_type in ['number', 'ratio'] else '%' if value_type == 'percentage' else '',
                                paragraph=paragraph.strip(),
                                paragraph_number=para_idx + 1,
                                page_number=f"第{page_num}頁",
                                confidence=final_confidence,
                                context_window=self._get_context_window(doc.page_content, paragraph),
                                company_name=doc_info.company_name,
                                report_year=doc_info.report_year,
                                stock_code=doc_info.stock_code,
                                keyword_distance=distance,
                                full_section=full_section
                            )
                            extractions.append(extraction)
        
        print(f"✅ 增強數據提取完成: 找到 {len(extractions)} 個結果")
        return extractions
    
    def _enhanced_split_paragraphs(self, text: str) -> List[str]:
        """增強的段落分割"""
        paragraphs = []
        
        # 方法1：標準分割
        standard_paras = re.split(r'\n{2,}|\r{2,}', text)
        paragraphs.extend([p.strip() for p in standard_paras if len(p.strip()) >= 20])
        
        # 方法2：句號分割
        sentence_paras = re.split(r'。{2,}|\.{2,}', text)
        paragraphs.extend([p.strip() for p in sentence_paras if len(p.strip()) >= 30])
        
        # 方法3：項目符號分割
        bullet_paras = re.split(r'\n\s*[•▶■▪]\s*', text)
        paragraphs.extend([p.strip() for p in bullet_paras if len(p.strip()) >= 25])
        
        # 方法4：編號分割
        number_paras = re.split(r'\n\s*\d+[\.\)]\s*', text)
        paragraphs.extend([p.strip() for p in number_paras if len(p.strip()) >= 25])
        
        # 保持原文
        if len(text.strip()) >= 50:
            paragraphs.append(text.strip())
        
        # 去重
        unique_paragraphs = []
        seen = set()
        for para in paragraphs:
            para_hash = hash(para[:100])  # 用前100字符作為去重依據
            if para_hash not in seen and len(para.strip()) >= 20:
                seen.add(para_hash)
                unique_paragraphs.append(para)
        
        return unique_paragraphs
    
    def _get_full_section(self, full_text: str, target_paragraph: str, expand_size: int = 300) -> str:
        """獲取完整段落內容（包含上下文）"""
        try:
            pos = full_text.find(target_paragraph)
            if pos == -1:
                return target_paragraph
            
            # 向前擴展到段落開始
            start = pos
            while start > 0 and full_text[start-1] not in ['\n', '\r']:
                start -= 1
            
            # 向後擴展到段落結束
            end = pos + len(target_paragraph)
            while end < len(full_text) and full_text[end] not in ['\n', '\r']:
                end += 1
            
            # 進一步擴展上下文
            start = max(0, start - expand_size)
            end = min(len(full_text), end + expand_size)
            
            return full_text[start:end].strip()
        except:
            return target_paragraph
    
    def _enhanced_post_process(self, extractions: List[NumericExtraction]) -> List[NumericExtraction]:
        """增強的後處理"""
        if not extractions:
            return extractions
        
        print(f"🔧 增強後處理 {len(extractions)} 個提取結果...")
        
        # 1. 精確去重（更嚴格）
        unique_extractions = []
        seen_combinations = set()
        
        for extraction in extractions:
            identifier = (
                extraction.keyword,
                extraction.value,
                extraction.value_type,
                extraction.paragraph[:150],  # 增加比較長度
                extraction.page_number
            )
            
            if identifier not in seen_combinations:
                seen_combinations.add(identifier)
                unique_extractions.append(extraction)
        
        print(f"📊 精確去重後: {len(unique_extractions)} 個結果")
        
        # 2. 信心分數篩選（動態閾值）
        filtered_extractions = []
        for extraction in unique_extractions:
            dynamic_threshold = self._get_post_process_threshold(extraction.keyword)
            if extraction.confidence >= dynamic_threshold:
                filtered_extractions.append(extraction)
        
        print(f"📊 信心分數篩選後: {len(filtered_extractions)} 個結果")
        
        # 3. 頁面級去重（每頁最多保留3筆高質量結果）
        page_filtered_extractions = self._enhanced_per_page_filtering(filtered_extractions, max_per_page=3)
        
        # 4. 按信心分數排序
        page_filtered_extractions.sort(key=lambda x: x.confidence, reverse=True)
        
        print(f"✅ 增強後處理完成: 保留 {len(page_filtered_extractions)} 個最終結果")
        return page_filtered_extractions
    
    def _get_post_process_threshold(self, keyword: str) -> float:
        """獲取後處理的動態閾值"""
        # 高價值關鍵字較低閾值
        high_value_keywords = ["使用量", "循環率", "回收率", "碳排減量", "再生材料"]
        if any(hvk in keyword for hvk in high_value_keywords):
            return 0.65
        
        # 中價值關鍵字中等閾值
        mid_value_keywords = ["材料", "能源", "效益", "成本", "處理"]
        if any(mvk in keyword for mvk in mid_value_keywords):
            return 0.7
        
        # 其他關鍵字較高閾值
        return 0.75
    
    def _enhanced_per_page_filtering(self, extractions: List[NumericExtraction], max_per_page: int = 3) -> List[NumericExtraction]:
        """增強的按頁面去重"""
        if not extractions:
            return extractions
        
        print(f"📄 執行增強按頁面去重（每頁最多保留 {max_per_page} 筆）...")
        
        # 按頁碼分組
        page_groups = {}
        for extraction in extractions:
            page_key = str(extraction.page_number).strip()
            if page_key not in page_groups:
                page_groups[page_key] = []
            page_groups[page_key].append(extraction)
        
        # 每頁面內按質量排序並保留最佳結果
        filtered_extractions = []
        
        for page_key, page_extractions in page_groups.items():
            # 按信心分數和關鍵字重要性排序
            page_extractions.sort(key=lambda x: (
                x.confidence,  # 信心分數
                self._get_keyword_importance(x.keyword),  # 關鍵字重要性
                len(x.value) if x.value != "[相關描述]" else 0  # 數值長度
            ), reverse=True)
            
            # 保留頂部結果，但要避免相同關鍵字重複
            kept_extractions = []
            used_keywords = set()
            
            for extraction in page_extractions:
                if len(kept_extractions) >= max_per_page:
                    break
                
                # 檢查關鍵字是否重複（允許少量重複）
                if extraction.keyword not in used_keywords or len(used_keywords) < max_per_page // 2:
                    kept_extractions.append(extraction)
                    used_keywords.add(extraction.keyword)
            
            filtered_extractions.extend(kept_extractions)
        
        print(f"   ✅ 增強頁面去重完成: {len(filtered_extractions)} 筆最終結果")
        return filtered_extractions
    
    def _get_keyword_importance(self, keyword: str) -> float:
        """獲取關鍵字重要性分數"""
        # 高重要性關鍵字
        high_importance = ["再生材料使用量", "材料循環率", "碳排減量", "回收率"]
        if any(hi in keyword for hi in high_importance):
            return 1.0
        
        # 中重要性關鍵字
        mid_importance = ["使用量", "處理量", "效益", "成本"]
        if any(mi in keyword for mi in mid_importance):
            return 0.8
        
        # 一般重要性
        return 0.6
    
    def _export_to_word(self, extractions: List[NumericExtraction], 
                       summary: ProcessingSummary, doc_info: DocumentInfo) -> str:
        """匯出到Word文檔"""
        print(f"📄 匯出結果到Word文檔...")
        
        try:
            word_path = self.word_exporter.create_word_document(extractions, doc_info, summary)
            print(f"✅ Word文檔已保存: {Path(word_path).name}")
            return word_path
        except Exception as e:
            print(f"❌ Word文檔匯出失敗: {e}")
            return ""
    
    # 其他必要方法保持不變或略作調整...
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
    
    def _export_to_excel(self, extractions: List[NumericExtraction], summary: ProcessingSummary, doc_info: DocumentInfo) -> str:
        """匯出結果到Excel（保持原有功能）"""
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
        
        # Excel匯出邏輯（與原版本相似，但包含股票代號）
        main_data = []
        
        # 第一行：公司信息
        header_row = {
            '股票代號': doc_info.stock_code,
            '公司名稱': doc_info.company_name,
            '報告年度': doc_info.report_year,
            '關鍵字': f"處理時間: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            '提取數值': f"{status_message}（增強版ESG報告書提取器 v2.0）",
            '數據類型': '',
            '單位': '',
            '段落內容': '',
            '段落編號': '',
            '頁碼': '',
            '信心分數': ''
        }
        main_data.append(header_row)
        
        # 空行分隔
        main_data.append({col: '' for col in header_row.keys()})
        
        # 如果有提取結果，添加結果數據
        if len(extractions) > 0:
            for extraction in extractions:
                main_data.append({
                    '股票代號': extraction.stock_code,
                    '公司名稱': extraction.company_name,
                    '報告年度': extraction.report_year,
                    '關鍵字': extraction.keyword,
                    '提取數值': extraction.value,
                    '數據類型': extraction.value_type,
                    '單位': extraction.unit,
                    '段落內容': extraction.paragraph,
                    '段落編號': extraction.paragraph_number,
                    '頁碼': extraction.page_number,
                    '信心分數': round(extraction.confidence, 3)
                })
        else:
            # 無結果說明
            no_result_row = {
                '股票代號': doc_info.stock_code,
                '公司名稱': doc_info.company_name,
                '報告年度': doc_info.report_year,
                '關鍵字': '無相關關鍵字匹配',
                '提取數值': 'N/A',
                '數據類型': 'no_data',
                '單位': '',
                '段落內容': '在此份ESG報告中未找到相關的數值數據',
                '段落編號': '',
                '頁碼': '',
                '信心分數': 0.0
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
            sheet_name = '增強提取結果' if len(extractions) > 0 else '無提取結果'
            pd.DataFrame(main_data).to_excel(writer, sheet_name=sheet_name, index=False)
            
            # 統計工作表
            pd.DataFrame(stats_data).to_excel(writer, sheet_name='統計摘要', index=False)
            
            # 處理摘要
            summary_data = [{
                '股票代號': summary.stock_code,
                '公司名稱': summary.company_name,
                '報告年度': summary.report_year,
                '總文檔數': summary.total_documents,
                '總提取結果': summary.total_extractions,
                '處理狀態': '成功提取' if len(extractions) > 0 else '無相關數據',
                '處理時間(秒)': round(summary.processing_time, 2),
                '處理日期': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                '提取器版本': '增強版ESG報告書提取器 v2.0'
            }]
            pd.DataFrame(summary_data).to_excel(writer, sheet_name='處理摘要', index=False)
        
        if len(extractions) > 0:
            print(f"✅ Excel檔案已保存，包含 {len(extractions)} 項提取結果")
        else:
            print(f"✅ Excel檔案已保存，標記為無提取結果")
        
        return output_path
    
    def _extract_unit(self, value_str: str) -> str:
        """從數值字符串中提取單位"""
        units = re.findall(r'[a-zA-Z\u4e00-\u9fff]+', value_str)
        return units[-1] if units else ""
    
    def _get_context_window(self, full_text: str, target_paragraph: str, window_size: int = 200) -> str:
        """獲取段落的上下文窗口"""
        try:
            pos = full_text.find(target_paragraph)
            if pos == -1:
                return target_paragraph[:400]
            
            start = max(0, pos - window_size)
            end = min(len(full_text), pos + len(target_paragraph) + window_size)
            
            return full_text[start:end]
        except:
            return target_paragraph[:400]

def main():
    """主函數 - 測試用"""
    print("📊 增強版ESG報告書提取器測試模式")
    
    extractor = EnhancedESGExtractor(enable_llm=False)
    print("✅ 增強版ESG報告書提取器初始化完成")

if __name__ == "__main__":
    main()