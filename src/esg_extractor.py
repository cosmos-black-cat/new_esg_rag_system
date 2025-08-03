#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ESG資料提取器 v2.1
整合增強關鍵字過濾、精確相關性檢查、LLM增強、智能去重
"""

import json
import re
import os
import sys
import pandas as pd
from pathlib import Path
from typing import Dict, List, Union, Tuple, Optional
from dataclasses import dataclass
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

# 導入增強的關鍵字配置和過濾管道
try:
    from keywords_config import (
        enhanced_filtering_pipeline, 
        EnhancedKeywordConfig, 
        KeywordConfig,
        EnhancedMatcher
    )
    ENHANCED_KEYWORDS_AVAILABLE = True
    print("✅ 增強關鍵字配置已載入")
except ImportError as e:
    print(f"⚠️ 增強關鍵字配置載入失敗: {e}")
    ENHANCED_KEYWORDS_AVAILABLE = False
    # 回退到基本配置
    class KeywordConfig:
        @classmethod 
        def get_all_keywords(cls):
            return ["再生塑膠", "再生塑料", "再生料", "再生pp"]

# 導入API管理器
try:
    from api_manager import GeminiAPIManager
    API_MANAGER_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ API管理器載入失敗: {e}")
    API_MANAGER_AVAILABLE = False

# =============================================================================
# 數據結構定義
# =============================================================================

@dataclass
class ExtractionMatch:
    """單個匹配結果"""
    keyword: str
    keyword_type: str  # 'continuous' or 'discontinuous'
    confidence: float
    matched_text: str
    
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

@dataclass
class ProcessingSummary:
    """處理摘要"""
    total_documents: int
    stage1_passed: int
    stage2_passed: int
    total_extractions: int
    keywords_found: Dict[str, int]
    processing_time: float
    enhanced_filtering_used: bool = False
    filtering_stats: Dict = None

# =============================================================================
# 增強的LLM管理器
# =============================================================================

class EnhancedLLMManager:
    """增強的LLM管理器，支援多API和改進的響應解析"""
    
    def __init__(self, api_keys: List[str], model_name: str):
        self.api_keys = api_keys
        self.model_name = model_name
        self.success_count = 0
        self.total_count = 0
        
        if len(api_keys) > 1 and API_MANAGER_AVAILABLE:
            print(f"🔄 啟用多API輪換模式，共 {len(api_keys)} 個Keys")
            self.api_manager = GeminiAPIManager(api_keys, model_name)
            self.mode = "multi_api"
        else:
            print("🔑 使用單API模式")
            self.llm = ChatGoogleGenerativeAI(
                model=model_name,
                google_api_key=api_keys[0],
                temperature=0.1,
                max_tokens=1024,
                convert_system_message_to_human=True
            )
            self.mode = "single_api"
    
    def invoke(self, prompt: str) -> str:
        """統一的LLM調用介面"""
        self.total_count += 1
        
        try:
            if self.mode == "multi_api":
                response = self.api_manager.invoke(prompt)
                self.success_count += 1
                return response
            else:
                response = self.llm.invoke(prompt)
                self.success_count += 1
                return response.content if hasattr(response, 'content') else str(response)
        except Exception as e:
            print(f"⚠️ LLM調用失敗: {e}")
            raise e
    
    def get_success_rate(self) -> float:
        """獲取成功率"""
        if self.total_count == 0:
            return 0.0
        return (self.success_count / self.total_count) * 100
    
    def print_stats(self):
        """打印統計信息"""
        print(f"📊 LLM調用統計: {self.success_count}/{self.total_count} ({self.get_success_rate():.1f}%)")
        
        if hasattr(self, 'api_manager'):
            self.api_manager.print_usage_statistics()

# =============================================================================
# 智能去重器
# =============================================================================

class ESGResultDeduplicator:
    """ESG提取結果去重器"""
    
    def __init__(self, similarity_threshold: float = 0.8):
        self.similarity_threshold = similarity_threshold
        self.value_match_threshold = 0.95
    
    def deduplicate_extractions(self, extractions: List[NumericExtraction]) -> List[NumericExtraction]:
        """去重提取結果列表"""
        if not extractions:
            return extractions
        
        print(f"🔄 開始智能去重: {len(extractions)} 個結果")
        
        # 將提取結果轉換為DataFrame進行處理
        data = []
        for i, extraction in enumerate(extractions):
            data.append({
                'index': i,
                'keyword': extraction.keyword,
                'value': extraction.value,
                'value_type': extraction.value_type,
                'paragraph': extraction.paragraph,
                'page_number': extraction.page_number,
                'confidence': extraction.confidence,
                'context_window': extraction.context_window
            })
        
        df = pd.DataFrame(data)
        
        # 識別重複組
        groups = self._group_similar_results(df)
        
        if not groups:
            print("✅ 未發現重複數據")
            return extractions
        
        # 執行去重
        deduplicated_extractions = self._merge_duplicate_groups(extractions, df, groups)
        
        print(f"✅ 智能去重完成: {len(extractions)} → {len(deduplicated_extractions)} 個結果")
        print(f"   合併了 {len(groups)} 個重複組")
        
        return deduplicated_extractions
    
    def _group_similar_results(self, df: pd.DataFrame) -> List[List[int]]:
        """識別相似的提取結果"""
        groups = []
        processed = set()
        
        for i, row1 in df.iterrows():
            if i in processed:
                continue
            
            current_group = [i]
            value1 = self._normalize_value(row1['value'])
            paragraph1 = str(row1['paragraph'])
            
            for j, row2 in df.iterrows():
                if j <= i or j in processed:
                    continue
                
                value2 = self._normalize_value(row2['value'])
                paragraph2 = str(row2['paragraph'])
                
                # 檢查是否為相似結果
                is_similar = False
                
                # 條件1: 相同數值 + 相似文本
                if value1 == value2 and value1 != "N/A":
                    text_similarity = self._calculate_text_similarity(paragraph1, paragraph2)
                    if text_similarity > self.similarity_threshold:
                        is_similar = True
                
                # 條件2: 完全相同的段落文本
                if self._calculate_text_similarity(paragraph1, paragraph2) > 0.95:
                    is_similar = True
                
                if is_similar:
                    current_group.append(j)
                    processed.add(j)
            
            if len(current_group) > 1:
                groups.append(current_group)
                for idx in current_group:
                    processed.add(idx)
        
        return groups
    
    def _merge_duplicate_groups(self, extractions: List[NumericExtraction], 
                               df: pd.DataFrame, groups: List[List[int]]) -> List[NumericExtraction]:
        """合併重複的提取結果組"""
        # 收集被合併的索引
        merged_indices = set()
        for group in groups:
            merged_indices.update(group)
        
        # 保留未被合併的結果
        deduplicated = []
        for i, extraction in enumerate(extractions):
            if i not in merged_indices:
                deduplicated.append(extraction)
        
        # 處理每個重複組
        for group in groups:
            group_extractions = [extractions[i] for i in group]
            merged_extraction = self._merge_extraction_group(group_extractions)
            deduplicated.append(merged_extraction)
        
        # 按信心分數排序
        deduplicated.sort(key=lambda x: x.confidence, reverse=True)
        
        return deduplicated
    
    def _merge_extraction_group(self, group_extractions: List[NumericExtraction]) -> NumericExtraction:
        """合併一組重複的提取結果"""
        # 選擇信心分數最高的作為基礎
        best_extraction = max(group_extractions, key=lambda x: x.confidence)
        
        # 合併關鍵字
        keywords = [e.keyword for e in group_extractions]
        unique_keywords = list(dict.fromkeys(keywords))  # 保持順序去重
        primary_keyword = self._select_primary_keyword(unique_keywords)
        
        # 合併頁碼
        pages = [e.page_number for e in group_extractions if e.page_number]
        unique_pages = list(dict.fromkeys(pages))
        
        # 計算平均信心分數
        avg_confidence = np.mean([e.confidence for e in group_extractions])
        
        # 合併上下文
        contexts = [e.context_window for e in group_extractions if e.context_window]
        merged_context = "\n---合併結果---\n".join(set(contexts))
        
        # 創建合併後的結果
        merged_extraction = NumericExtraction(
            keyword=primary_keyword,
            value=best_extraction.value,
            value_type=best_extraction.value_type,
            unit=best_extraction.unit,
            paragraph=best_extraction.paragraph,
            paragraph_number=best_extraction.paragraph_number,
            page_number=" | ".join(unique_pages),
            confidence=avg_confidence,
            context_window=f"{merged_context}\n[智能合併了{len(group_extractions)}個結果: {', '.join(unique_keywords)}]"
        )
        
        return merged_extraction
    
    def _select_primary_keyword(self, keywords: List[str]) -> str:
        """選擇主要關鍵字（優先中文、簡潔）"""
        if not keywords:
            return 'N/A'
        
        # 優先級：中文關鍵字 > 英文關鍵字
        chinese_keywords = [k for k in keywords if re.search(r'[\u4e00-\u9fff]', k)]
        english_keywords = [k for k in keywords if not re.search(r'[\u4e00-\u9fff]', k)]
        
        if chinese_keywords:
            # 選擇最短的中文關鍵字
            return min(chinese_keywords, key=len)
        else:
            # 選擇最短的英文關鍵字
            return min(english_keywords, key=len)
    
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """計算文本相似度"""
        if pd.isna(text1) or pd.isna(text2):
            return 0.0
        
        text1 = str(text1).strip()
        text2 = str(text2).strip()
        
        if not text1 or not text2:
            return 0.0
        
        return SequenceMatcher(None, text1, text2).ratio()
    
    def _normalize_value(self, value: str) -> str:
        """標準化數值格式"""
        if pd.isna(value):
            return "N/A"
        
        value_str = str(value).strip()
        
        if not value_str or value_str.lower() in ['nan', 'none', 'null']:
            return "N/A"
        
        # 移除多餘空格
        value_str = re.sub(r'\s+', ' ', value_str)
        
        # 標準化百分比
        if '%' in value_str or '％' in value_str:
            numbers = re.findall(r'\d+\.?\d*', value_str)
            if numbers:
                return f"{numbers[0]}%"
        
        # 標準化數值帶單位
        number_match = re.search(r'(\d+(?:,\d{3})*(?:\.\d+)?)\s*([^\d\s]+)', value_str)
        if number_match:
            number, unit = number_match.groups()
            return f"{number}{unit}"
        
        return value_str

    def deduplicate_excel_file(self, file_path: str) -> str:
        """去重Excel文件的公開介面"""
        print(f"📊 處理Excel文件去重: {Path(file_path).name}")
        
        try:
            # 這裡可以添加Excel文件去重的具體實現
            # 目前返回原文件路徑
            return file_path
        except Exception as e:
            print(f"❌ Excel去重失敗: {e}")
            return None

# =============================================================================
# 主要提取器類
# =============================================================================

class ESGExtractor:
    """增強版ESG資料提取器主類"""
    
    def __init__(self, vector_db_path: str = None, enable_llm: bool = True, auto_dedupe: bool = True):
        """
        初始化增強版提取器
        
        Args:
            vector_db_path: 向量資料庫路徑
            enable_llm: 是否啟用LLM增強
            auto_dedupe: 是否自動去重
        """
        self.vector_db_path = vector_db_path or VECTOR_DB_PATH
        self.enable_llm = enable_llm
        self.auto_dedupe = auto_dedupe
        
        # 初始化組件
        if ENHANCED_KEYWORDS_AVAILABLE:
            self.keyword_config = EnhancedKeywordConfig()
            self.matcher = EnhancedMatcher()
            print("✅ 使用增強關鍵字配置")
        else:
            self.keyword_config = KeywordConfig()
            self.matcher = self._create_basic_matcher()
            print("⚠️ 使用基本關鍵字配置")
        
        self.deduplicator = ESGResultDeduplicator()
        
        # 載入向量資料庫
        self._load_vector_database()
        
        # 初始化LLM（如果啟用）
        if self.enable_llm:
            self._init_llm()
        
        print("✅ 增強版ESG提取器初始化完成")
        if self.auto_dedupe:
            print("✅ 智能去重功能已啟用")

    def _create_basic_matcher(self):
        """創建基本匹配器（回退方案）"""
        class BasicMatcher:
            def extract_numbers_and_percentages(self, text: str):
                numbers = re.findall(r'\d+(?:,\d{3})*(?:\.\d+)?(?:\s*(?:萬|千)?(?:噸|kg|KG|公斤))', text)
                percentages = re.findall(r'\d+(?:\.\d+)?(?:\s*%|％)', text)
                return numbers, percentages
        
        return BasicMatcher()

    def _load_vector_database(self):
        """載入向量資料庫"""
        if not os.path.exists(self.vector_db_path):
            raise FileNotFoundError(f"向量資料庫不存在: {self.vector_db_path}")
        
        print(f"📚 載入向量資料庫: {self.vector_db_path}")
        embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={'device': 'cpu'}
        )
        self.db = FAISS.load_local(
            self.vector_db_path, 
            embeddings, 
            allow_dangerous_deserialization=True
        )
        print(f"✅ 向量資料庫載入完成")
    
    def _init_llm(self):
        """初始化增強LLM管理器"""
        try:
            print(f"🤖 初始化增強LLM管理器...")
            
            self.llm_manager = EnhancedLLMManager(
                api_keys=GEMINI_API_KEYS,
                model_name=GEMINI_MODEL
            )
            
            print("✅ 增強LLM管理器初始化完成")
            
        except Exception as e:
            print(f"⚠️ LLM初始化失敗: {e}")
            self.enable_llm = False
    
    def enhanced_stage1_filtering(self, documents: List[Document]) -> Tuple[List[Document], List[ExtractionMatch]]:
        """增強的第一階段篩選：使用精確過濾管道"""
        print("🔍 執行增強的第一階段篩選...")
        
        if not ENHANCED_KEYWORDS_AVAILABLE:
            return self._basic_stage1_filtering(documents)
        
        keywords = self.keyword_config.get_all_keywords()
        passed_docs = []
        all_matches = []
        filtering_stats = {
            'total_docs': len(documents),
            'passed_docs': 0,
            'rejected_docs': 0,
            'total_matches': 0
        }
        
        for doc in tqdm(documents, desc="增強第一階段篩選"):
            # 使用增強過濾管道
            passed, matches = enhanced_filtering_pipeline(doc.page_content, keywords)
            
            if passed and matches:
                # 只保留高相關性的匹配
                high_relevance_matches = [m for m in matches if m.get('relevance_score', 0) > 0.75]
                
                if high_relevance_matches:
                    passed_docs.append(doc)
                    filtering_stats['passed_docs'] += 1
                    
                    # 轉換為ExtractionMatch格式
                    for match in high_relevance_matches:
                        extraction_match = ExtractionMatch(
                            keyword=match['keyword'],
                            keyword_type=match['match_type'],
                            confidence=match.get('relevance_score', 0.8),
                            matched_text=match.get('match_details', '')
                        )
                        all_matches.append(extraction_match)
                        filtering_stats['total_matches'] += 1
                else:
                    filtering_stats['rejected_docs'] += 1
            else:
                filtering_stats['rejected_docs'] += 1
        
        print(f"✅ 增強第一階段完成: {len(passed_docs)}/{len(documents)} 文檔通過")
        print(f"   精確過濾效果: 拒絕了 {filtering_stats['rejected_docs']} 個不相關文檔")
        print(f"   找到高質量匹配: {filtering_stats['total_matches']} 個")
        
        return passed_docs, all_matches
    
    def _basic_stage1_filtering(self, documents: List[Document]) -> Tuple[List[Document], List[ExtractionMatch]]:
        """基本的第一階段篩選（回退方案）"""
        print("🔍 執行基本第一階段篩選...")
        
        keywords = self.keyword_config.get_all_keywords()
        passed_docs = []
        all_matches = []
        
        for doc in tqdm(documents, desc="基本第一階段篩選"):
            doc_matches = []
            
            for keyword in keywords:
                if isinstance(keyword, str) and keyword.lower() in doc.page_content.lower():
                    match = ExtractionMatch(
                        keyword=keyword,
                        keyword_type='continuous',
                        confidence=0.8,
                        matched_text=f"基本匹配: {keyword}"
                    )
                    doc_matches.append(match)
            
            if doc_matches:
                passed_docs.append(doc)
                all_matches.extend(doc_matches)
        
        print(f"✅ 基本第一階段完成: {len(passed_docs)}/{len(documents)} 文檔通過")
        return passed_docs, all_matches
    
    def stage2_filtering(self, documents: List[Document]) -> List[NumericExtraction]:
        """第二階段篩選：提取包含數值或百分比的內容"""
        print("🔢 執行第二階段篩選...")
        
        keywords = self.keyword_config.get_all_keywords()
        extractions = []
        
        for doc in tqdm(documents, desc="第二階段篩選"):
            # 分割成段落
            paragraphs = self._split_into_paragraphs(doc.page_content)
            page_num = doc.metadata.get('page', '未知')
            
            for para_idx, paragraph in enumerate(paragraphs):
                if len(paragraph.strip()) < 10:
                    continue
                
                # 檢查段落中的關鍵字
                if ENHANCED_KEYWORDS_AVAILABLE:
                    # 使用增強過濾檢查段落
                    passed, matches = enhanced_filtering_pipeline(paragraph, keywords)
                    if not passed or not matches:
                        continue
                    
                    para_matches = [(m['original_keyword'], m.get('relevance_score', 0.8), m.get('match_details', '')) 
                                   for m in matches]
                else:
                    # 基本關鍵字檢查
                    para_matches = []
                    for keyword in keywords:
                        if isinstance(keyword, str) and keyword.lower() in paragraph.lower():
                            para_matches.append((keyword, 0.8, f"基本匹配: {keyword}"))
                
                if para_matches:
                    # 提取數值和百分比
                    numbers, percentages = self.matcher.extract_numbers_and_percentages(paragraph)
                    
                    if numbers or percentages:
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
                                    context_window=self._get_context_window(doc.page_content, paragraph)
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
                                    context_window=self._get_context_window(doc.page_content, paragraph)
                                )
                                extractions.append(extraction)
        
        print(f"✅ 第二階段完成: 找到 {len(extractions)} 個數值提取結果")
        return extractions
    
    def llm_enhancement(self, extractions: List[NumericExtraction]) -> List[NumericExtraction]:
        """LLM增強：驗證和豐富提取結果"""
        if not self.enable_llm or not extractions:
            return extractions
        
        print("🤖 執行LLM增強驗證...")
        print(f"📊 處理 {len(extractions)} 個提取結果")
        
        enhanced_extractions = []
        
        for i, extraction in enumerate(tqdm(extractions, desc="LLM增強")):
            try:
                # 構建改進的驗證提示
                prompt = self._build_enhanced_verification_prompt(extraction)
                
                # 調用LLM
                response_content = self.llm_manager.invoke(prompt)
                llm_result = self._parse_enhanced_llm_response(response_content)
                
                # 更新提取結果
                if llm_result and llm_result.get("is_relevant", True):
                    # 更新信心分數
                    llm_confidence = llm_result.get("confidence", extraction.confidence)
                    extraction.confidence = min(
                        (extraction.confidence + llm_confidence) / 2, 
                        1.0
                    )
                    
                    # 添加LLM的解釋
                    extraction.context_window += f"\n[LLM驗證]: {llm_result.get('explanation', '')}"
                    enhanced_extractions.append(extraction)
                elif llm_result and llm_result.get("confidence", 0) > 0.7:
                    # 即使不相關但信心分數高，降低信心後保留
                    extraction.confidence *= 0.7
                    extraction.context_window += f"\n[LLM注意]: {llm_result.get('explanation', '')}"
                    enhanced_extractions.append(extraction)
                # 其他情況丟棄
                
            except Exception as e:
                print(f"⚠️ LLM增強失敗 (第{i+1}個): {e}")
                enhanced_extractions.append(extraction)  # 保留原始結果
        
        # 顯示處理統計
        success_rate = self.llm_manager.get_success_rate()
        retention_rate = (len(enhanced_extractions) / len(extractions)) * 100
        
        print(f"✅ LLM增強完成:")
        print(f"   API成功率: {success_rate:.1f}%")
        print(f"   結果保留率: {retention_rate:.1f}% ({len(enhanced_extractions)}/{len(extractions)})")
        
        # 顯示API使用統計
        self.llm_manager.print_stats()
        
        return enhanced_extractions
    
    def export_to_excel(self, extractions: List[NumericExtraction], summary: ProcessingSummary) -> str:
        """匯出結果到Excel"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(RESULTS_PATH, f"esg_extraction_results_{timestamp}.xlsx")
        
        # 確保輸出目錄存在
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        print(f"📊 匯出增強版結果到Excel: {output_path}")
        
        # 準備主要數據
        main_data = []
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
            
            stats_data.append({
                '關鍵字': keyword,
                '總提取數': len(keyword_extractions),
                '數值類型': len(numbers),
                '百分比類型': len(percentages),
                '平均信心分數': round(np.mean([e.confidence for e in keyword_extractions]) if keyword_extractions else 0, 3),
                '最高信心分數': round(max([e.confidence for e in keyword_extractions]) if keyword_extractions else 0, 3)
            })
        
        # 準備處理摘要
        process_summary = [{
            '項目': '增強版處理摘要',
            '總文檔數': summary.total_documents,
            '第一階段通過': summary.stage1_passed,
            '第二階段通過': summary.stage2_passed,
            '總提取結果': summary.total_extractions,
            '處理時間(秒)': round(summary.processing_time, 2),
            '增強關鍵字過濾': '已啟用' if ENHANCED_KEYWORDS_AVAILABLE else '未啟用',
            '自動去重': '已啟用' if self.auto_dedupe else '未啟用',
            'LLM增強': '已啟用' if self.enable_llm else '未啟用'
        }]
        
        # 寫入Excel
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            # 主要結果
            pd.DataFrame(main_data).to_excel(writer, sheet_name='提取結果', index=False)
            
            # 統計摘要
            pd.DataFrame(stats_data).to_excel(writer, sheet_name='關鍵字統計', index=False)
            
            # 處理摘要
            pd.DataFrame(process_summary).to_excel(writer, sheet_name='處理摘要', index=False)
        
        print(f"✅ 增強版Excel檔案已保存")
        return output_path
    
    def run_complete_extraction(self, max_documents: int = 200) -> Tuple[List[NumericExtraction], ProcessingSummary, str]:
        """執行完整的增強版資料提取流程"""
        start_time = datetime.now()
        print("🚀 開始增強版ESG資料提取流程")
        print("=" * 60)
        
        # 1. 獲取相關文檔
        print("📄 檢索相關文檔...")
        documents = self._retrieve_relevant_documents(max_documents)
        
        # 2. 增強的第一階段篩選
        stage1_docs, stage1_matches = self.enhanced_stage1_filtering(documents)
        
        # 3. 第二階段篩選
        stage2_extractions = self.stage2_filtering(stage1_docs)
        
        # 4. LLM增強（如果啟用）
        enhanced_extractions = self.llm_enhancement(stage2_extractions)
        
        # 5. 智能去重（如果啟用）
        if self.auto_dedupe:
            print("\n🔄 執行智能去重...")
            final_extractions = self.deduplicator.deduplicate_extractions(enhanced_extractions)
        else:
            final_extractions = enhanced_extractions
        
        # 6. 創建處理摘要
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        
        keywords_found = {}
        for extraction in final_extractions:
            keyword = extraction.keyword
            keywords_found[keyword] = keywords_found.get(keyword, 0) + 1
        
        summary = ProcessingSummary(
            total_documents=len(documents),
            stage1_passed=len(stage1_docs),
            stage2_passed=len(stage2_extractions),
            total_extractions=len(final_extractions),
            keywords_found=keywords_found,
            processing_time=processing_time,
            enhanced_filtering_used=ENHANCED_KEYWORDS_AVAILABLE
        )
        
        # 7. 匯出結果
        excel_path = self.export_to_excel(final_extractions, summary)
        
        # 8. 顯示最終摘要
        self._print_enhanced_final_summary(summary, final_extractions)
        
        return final_extractions, summary, excel_path
    
    # =============================================================================
    # 輔助方法
    # =============================================================================
    
    def _retrieve_relevant_documents(self, max_docs: int) -> List[Document]:
        """檢索相關文檔"""
        if ENHANCED_KEYWORDS_AVAILABLE:
            config = EnhancedKeywordConfig()
            keywords = (
                config.CORE_RECYCLED_PLASTIC_KEYWORDS["高相關連續關鍵字"] +
                config.CORE_RECYCLED_PLASTIC_KEYWORDS["高相關不連續關鍵字"]
            )
        else:
            keywords = self.keyword_config.get_all_keywords()
        
        all_docs = []
        
        # 對每個關鍵字進行檢索
        for keyword in keywords:
            search_term = keyword if isinstance(keyword, str) else " ".join(keyword)
            docs = self.db.similarity_search(search_term, k=30)
            all_docs.extend(docs)
        
        # 去重
        unique_docs = {}
        for doc in all_docs:
            doc_hash = hash(doc.page_content)
            if doc_hash not in unique_docs:
                unique_docs[doc_hash] = doc
        
        result_docs = list(unique_docs.values())[:max_docs]
        print(f"📚 檢索到 {len(result_docs)} 個相關文檔")
        return result_docs
    
    def _split_into_paragraphs(self, text: str) -> List[str]:
        """將文本分割成段落"""
        paragraphs = re.split(r'\n{2,}|\r{2,}|。{2,}|\.{2,}', text)
        
        cleaned_paragraphs = []
        for para in paragraphs:
            para = para.strip()
            if len(para) >= 10:
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
    
    def _build_enhanced_verification_prompt(self, extraction: NumericExtraction) -> str:
        """構建增強的LLM驗證提示"""
        return f"""請分析以下數據提取結果是否與再生塑膠/回收塑料的實際生產使用相關：

關鍵字: {extraction.keyword}
提取值: {extraction.value}
數據類型: {extraction.value_type}

段落內容: {extraction.paragraph[:300]}

判斷標準:
1. 是否與再生塑膠、回收塑料、PCR材料的實際生產或使用相關？
2. 是否排除了賽事活動、職業災害、水資源管理等無關主題？
3. 數值是否確實描述再生材料的產能、產量、使用量或比例？

請嚴格按照JSON格式回答（不要包含其他文字）：
{{"is_relevant": true, "confidence": 0.85, "explanation": "簡短說明相關性"}}"""
    
    def _parse_enhanced_llm_response(self, response_text: str) -> Optional[Dict]:
        """解析增強的LLM響應"""
        if not response_text:
            return None
        
        try:
            # 方法1: 直接解析JSON
            json_match = re.search(r'\{[^{}]*\}', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                result = json.loads(json_str)
                
                if 'is_relevant' in result:
                    return {
                        'is_relevant': bool(result.get('is_relevant', False)),
                        'confidence': float(result.get('confidence', 0.5)),
                        'explanation': str(result.get('explanation', '無說明'))
                    }
        except:
            pass
        
        # 方法2: 關鍵字解析
        try:
            response_lower = response_text.lower()
            
            # 判斷相關性
            is_relevant = False
            if any(word in response_lower for word in ['true', '相關', '是', 'relevant']):
                is_relevant = True
            
            # 提取信心分數
            confidence = 0.5
            confidence_match = re.search(r'(?:confidence|信心).*?(\d+\.?\d*)', response_lower)
            if confidence_match:
                confidence = min(float(confidence_match.group(1)), 1.0)
                if confidence > 1:
                    confidence = confidence / 100
            
            return {
                'is_relevant': is_relevant,
                'confidence': confidence,
                'explanation': response_text[:100] + "..." if len(response_text) > 100 else response_text
            }
            
        except:
            pass
        
        # 方法3: 保守默認
        return {
            'is_relevant': False,
            'confidence': 0.3,
            'explanation': f"響應解析失敗: {response_text[:50]}..."
        }
    
    def _print_enhanced_final_summary(self, summary: ProcessingSummary, extractions: List[NumericExtraction]):
        """打印增強版最終摘要"""
        print("\n" + "=" * 70)
        print("📋 增強版提取完成摘要")
        print("=" * 70)
        print(f"📚 處理文檔數: {summary.total_documents}")
        print(f"🔍 第一階段通過: {summary.stage1_passed}")
        print(f"🔢 第二階段通過: {summary.stage2_passed}")
        print(f"📊 總提取結果: {summary.total_extractions}")
        print(f"⏱️ 處理時間: {summary.processing_time:.2f} 秒")
        print(f"🎯 增強關鍵字過濾: {'已啟用' if summary.enhanced_filtering_used else '未啟用'}")
        print(f"🧹 智能去重: {'已啟用' if self.auto_dedupe else '未啟用'}")
        print(f"🤖 LLM增強: {'已啟用' if self.enable_llm else '未啟用'}")
        
        print(f"\n📈 關鍵字分布:")
        for keyword, count in summary.keywords_found.items():
            print(f"   {keyword}: {count} 個結果")
        
        if extractions:
            numbers = [e for e in extractions if e.value_type == 'number']
            percentages = [e for e in extractions if e.value_type == 'percentage']
            
            print(f"\n🔢 數據類型分布:")
            print(f"   數值: {len(numbers)} 個")
            print(f"   百分比: {len(percentages)} 個")
            
            avg_confidence = np.mean([e.confidence for e in extractions])
            print(f"📊 平均信心分數: {avg_confidence:.3f}")
            
            # 顯示LLM統計
            if self.enable_llm and hasattr(self, 'llm_manager'):
                print(f"🤖 LLM處理成功率: {self.llm_manager.get_success_rate():.1f}%")

def main():
    """主函數 - 獨立運行測試"""
    try:
        print("🚀 增強版ESG資料提取器 - 獨立測試模式")
        print("=" * 60)
        
        # 初始化提取器（啟用所有增強功能）
        extractor = ESGExtractor(enable_llm=True, auto_dedupe=True)
        
        # 執行完整提取
        extractions, summary, excel_path = extractor.run_complete_extraction()
        
        if extractions:
            print(f"\n🎉 增強版提取完成！")
            print(f"📁 結果已保存至: {excel_path}")
            
            # 顯示前幾個結果作為樣例
            print(f"\n📋 樣例結果 (前3個):")
            for i, extraction in enumerate(extractions[:3], 1):
                print(f"\n{i}. 關鍵字: {extraction.keyword}")
                print(f"   數值: {extraction.value}")
                print(f"   類型: {extraction.value_type}")
                print(f"   頁碼: {extraction.page_number}")
                print(f"   信心: {extraction.confidence:.2f}")
                print(f"   段落: {extraction.paragraph[:100]}...")
        
        else:
            print("❌ 未找到任何提取結果")
    
    except Exception as e:
        print(f"❌ 執行錯誤: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()