#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ESG資料提取器 v2.0
整合兩段式篩選、不連續關鍵字匹配、LLM增強、智能去重
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

# =============================================================================
# 關鍵字配置類
# =============================================================================

class KeywordConfig:
    """關鍵字配置管理類"""
    
    # 簡化的四個核心關鍵字
    CORE_KEYWORDS = {
        "再生塑膠材料": {
            "continuous": [
                "再生塑膠",
                "再生塑料", 
                "再生料",
                "再生pp"
            ],
            "discontinuous": [
                ("再生", "塑膠"),
                ("再生", "塑料"),
                ("再生", "PP"),
                ("PP", "回收"),
                ("PP", "再生"),
                ("PP", "棧板", "回收"),
                ("塑膠", "回收"),
                ("塑料", "回收"),
                ("PCR", "塑膠"),
                ("PCR", "塑料"),
                ("PCR", "材料"),
                ("回收", "塑膠"),
                ("回收", "塑料"),
                ("rPET", "含量"),
                ("再生", "材料"),
                ("MLCC", "回收"),
                ("回收", "產能")
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
    
    @classmethod
    def get_keyword_category(cls, keyword: Union[str, tuple]) -> str:
        """獲取關鍵字所屬類別"""
        for category_name, category_data in cls.CORE_KEYWORDS.items():
            if keyword in category_data["continuous"] or keyword in category_data["discontinuous"]:
                return category_name
        return "未知類別"

# =============================================================================
# 增強匹配引擎
# =============================================================================

class EnhancedMatcher:
    """增強的關鍵字匹配引擎"""
    
    def __init__(self, max_distance: int = 150):
        self.max_distance = max_distance
        
        # 數值匹配模式（更全面）
        self.number_patterns = [
            r'\d+(?:,\d{3})*(?:\.\d+)?(?:\s*(?:kg|KG|公斤|噸|克|g|G|公克|萬噸|千噸))',
            r'\d+(?:,\d{3})*(?:\.\d+)?(?:\s*噸/月)',
            r'\d+(?:,\d{3})*(?:\.\d+)?(?:\s*萬|千)?(?:噸|公斤|kg|g)',
            r'\d+(?:,\d{3})*(?:\.\d+)?(?:\s*(?:個|件|批|台|套|次|倍))',
            r'\d+(?:,\d{3})*(?:\.\d+)?(?:\s*立方米|m³)',
        ]
        
        # 百分比匹配模式
        self.percentage_patterns = [
            r'\d+(?:\.\d+)?(?:\s*%|％|百分比)',
            r'\d+(?:\.\d+)?(?:\s*成)',
            r'百分之\d+(?:\.\d+)?',
        ]
    
    def match_keyword(self, text: str, keyword: Union[str, tuple]) -> Tuple[bool, float, str]:
        """
        匹配關鍵字
        
        Returns:
            Tuple[是否匹配, 信心分數, 匹配詳情]
        """
        text_lower = text.lower()
        
        if isinstance(keyword, str):
            # 連續關鍵字匹配
            if keyword.lower() in text_lower:
                # 尋找精確匹配位置，提供上下文
                pos = text_lower.find(keyword.lower())
                start = max(0, pos - 20)
                end = min(len(text), pos + len(keyword) + 20)
                context = text[start:end]
                return True, 1.0, f"精確匹配: {context}"
            return False, 0.0, ""
        
        elif isinstance(keyword, tuple):
            # 不連續關鍵字匹配
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
            start = max(0, min_pos - 30)
            end = min(len(text), max_pos + 30)
            context = text[start:end]
            
            if distance <= 30:
                return True, 0.95, f"近距離匹配({distance}字): {context}"
            elif distance <= 80:
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
        
        print(f"🔄 開始去重處理: {len(extractions)} 個結果")
        
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
        
        print(f"✅ 去重完成: {len(extractions)} → {len(deduplicated_extractions)} 個結果")
        
        return deduplicated_extractions
    
    def deduplicate_excel_file(self, file_path: str) -> str:
        """去重Excel文件"""
        print(f"📊 處理Excel文件: {file_path}")
        
        try:
            # 載入Excel數據
            df = self._load_excel_data(file_path)
            if df is None:
                return None
            
            # 標準化列名
            df = self._standardize_excel_columns(df)
            
            # 識別重複組
            groups = self._group_similar_excel_results(df)
            
            if not groups:
                print("✅ Excel文件中未發現重複數據")
                return file_path
            
            # 創建去重後的DataFrame
            deduplicated_df = self._create_deduplicated_dataframe(df, groups)
            
            # 生成統計摘要
            summary_df = self._create_summary_statistics(df, deduplicated_df)
            
            # 導出結果
            output_path = self._export_deduplicated_excel(deduplicated_df, summary_df, file_path)
            
            # 顯示處理摘要
            self._print_excel_dedup_summary(df, deduplicated_df, groups)
            
            return output_path
            
        except Exception as e:
            print(f"❌ Excel去重處理失敗: {e}")
            import traceback
            traceback.print_exc()
            return None
    
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
    
    def _group_similar_excel_results(self, df: pd.DataFrame) -> List[List[int]]:
        """識別Excel中的相似結果"""
        groups = []
        processed = set()
        
        for i, row1 in df.iterrows():
            if i in processed:
                continue
            
            current_group = [i]
            value1 = self._normalize_value(row1['value'])
            paragraph1 = str(row1.get('paragraph', ''))
            
            for j, row2 in df.iterrows():
                if j <= i or j in processed:
                    continue
                
                value2 = self._normalize_value(row2['value'])
                paragraph2 = str(row2.get('paragraph', ''))
                
                # 檢查是否為相似結果
                is_similar = False
                
                # 條件1: 相同數值 + 相似關鍵字
                if value1 == value2 and value1 not in ["N/A", "未提及", ""]:
                    keyword_similarity = self._calculate_text_similarity(
                        str(row1.get('keyword', '')), str(row2.get('keyword', ''))
                    )
                    if keyword_similarity > 0.6:  # 關鍵字相似度較低的閾值
                        is_similar = True
                
                # 條件2: 相似的段落文本 + 相同數值
                if value1 == value2 and paragraph1 and paragraph2:
                    text_similarity = self._calculate_text_similarity(paragraph1, paragraph2)
                    if text_similarity > self.similarity_threshold:
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
            context_window=f"{merged_context}\n[合併了{len(group_extractions)}個結果: {', '.join(unique_keywords)}]"
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
    
    def _load_excel_data(self, file_path: str) -> Optional[pd.DataFrame]:
        """載入Excel數據"""
        try:
            # 嘗試讀取不同的工作表
            possible_sheets = ['提取結果', 'extraction_results', 'results', 'Sheet1']
            
            for sheet_name in possible_sheets:
                try:
                    df = pd.read_excel(file_path, sheet_name=sheet_name)
                    print(f"✅ 讀取工作表: {sheet_name}")
                    return df
                except:
                    continue
            
            # 如果都失敗，讀取第一個工作表
            df = pd.read_excel(file_path)
            print("✅ 讀取第一個工作表")
            return df
            
        except Exception as e:
            print(f"❌ 載入Excel失敗: {e}")
            return None
    
    def _standardize_excel_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """標準化Excel列名"""
        column_mapping = {
            '關鍵字': 'keyword',
            '提取數值': 'value', 
            '數據類型': 'data_type',
            '段落內容': 'paragraph',
            '段落編號': 'paragraph_number',
            '頁碼': 'page_number',
            '信心分數': 'confidence',
            '上下文': 'context',
            '指標類別': 'indicator',
            '提取值': 'value',
            '來源頁面': 'page_number',
            '來源文本': 'paragraph',
            '說明': 'explanation'
        }
        
        df_renamed = df.rename(columns=column_mapping)
        
        # 確保必要列存在
        required_columns = ['keyword', 'value']
        for col in required_columns:
            if col not in df_renamed.columns:
                df_renamed[col] = 'N/A'
        
        return df_renamed
    
    def _create_deduplicated_dataframe(self, df: pd.DataFrame, groups: List[List[int]]) -> pd.DataFrame:
        """創建去重後的DataFrame"""
        # 收集被合併的索引
        merged_indices = set()
        for group in groups:
            merged_indices.update(group)
        
        # 未被合併的記錄
        unmerged_df = df[~df.index.isin(merged_indices)].copy()
        
        # 創建合併後的記錄
        merged_records = []
        for group in groups:
            merged_record = self._merge_excel_group(df, group)
            merged_records.append(merged_record)
        
        # 合併數據
        if merged_records:
            merged_df = pd.DataFrame(merged_records)
            
            # 標準化未合併數據
            unmerged_standardized = []
            for _, row in unmerged_df.iterrows():
                record = {
                    'keyword': str(row.get('keyword', 'N/A')),
                    'alternative_keywords': '',
                    'value': str(row.get('value', 'N/A')),
                    'data_type': str(row.get('data_type', 'N/A')),
                    'confidence': float(row.get('confidence', 0.5)) if pd.notna(row.get('confidence')) else 0.5,
                    'paragraph': str(row.get('paragraph', 'N/A')),
                    'page_number': str(row.get('page_number', 'N/A')),
                    'merged_count': 1,
                    'original_indices': str([row.name])
                }
                unmerged_standardized.append(record)
            
            if unmerged_standardized:
                unmerged_std_df = pd.DataFrame(unmerged_standardized)
                final_df = pd.concat([merged_df, unmerged_std_df], ignore_index=True)
            else:
                final_df = merged_df
        else:
            # 沒有合併記錄的情況
            unmerged_standardized = []
            for _, row in unmerged_df.iterrows():
                record = {
                    'keyword': str(row.get('keyword', 'N/A')),
                    'alternative_keywords': '',
                    'value': str(row.get('value', 'N/A')),
                    'data_type': str(row.get('data_type', 'N/A')),
                    'confidence': float(row.get('confidence', 0.5)) if pd.notna(row.get('confidence')) else 0.5,
                    'paragraph': str(row.get('paragraph', 'N/A')),
                    'page_number': str(row.get('page_number', 'N/A')),
                    'merged_count': 1,
                    'original_indices': str([row.name])
                }
                unmerged_standardized.append(record)
            
            final_df = pd.DataFrame(unmerged_standardized)
        
        # 按信心分數排序
        final_df = final_df.sort_values('confidence', ascending=False).reset_index(drop=True)
        
        return final_df
    
    def _merge_excel_group(self, df: pd.DataFrame, group_indices: List[int]) -> Dict:
        """合併Excel中的一組重複記錄"""
        group_data = df.iloc[group_indices]
        
        # 選擇最佳記錄
        if 'confidence' in group_data.columns:
            best_idx = group_data['confidence'].idxmax()
        else:
            best_idx = group_indices[0]
        
        best_record = group_data.loc[best_idx]
        
        # 合併關鍵字
        keywords = [str(row.get('keyword', 'N/A')) for _, row in group_data.iterrows() 
                   if pd.notna(row.get('keyword'))]
        unique_keywords = list(dict.fromkeys(keywords))
        primary_keyword = self._select_primary_keyword(unique_keywords)
        secondary_keywords = [k for k in unique_keywords if k != primary_keyword]
        
        # 合併其他信息
        pages = [str(row.get('page_number', 'N/A')) for _, row in group_data.iterrows() 
                if pd.notna(row.get('page_number'))]
        unique_pages = list(dict.fromkeys(pages))
        
        # 計算平均信心分數
        confidences = []
        for _, row in group_data.iterrows():
            conf = row.get('confidence')
            if pd.notna(conf) and conf != 'N/A':
                try:
                    confidences.append(float(conf))
                except:
                    pass
        
        avg_confidence = np.mean(confidences) if confidences else 0.5
        
        return {
            'keyword': primary_keyword,
            'alternative_keywords': ' | '.join(secondary_keywords) if secondary_keywords else '',
            'value': str(best_record.get('value', 'N/A')),
            'data_type': str(best_record.get('data_type', 'N/A')),
            'confidence': round(avg_confidence, 3),
            'paragraph': str(best_record.get('paragraph', 'N/A')),
            'page_number': ' | '.join(unique_pages),
            'merged_count': len(group_indices),
            'original_indices': str(group_indices)
        }
    
    def _create_summary_statistics(self, original_df: pd.DataFrame, deduplicated_df: pd.DataFrame) -> pd.DataFrame:
        """創建統計摘要"""
        stats = []
        
        # 整體統計
        stats.append({
            '項目': '總記錄數',
            '原始': len(original_df),
            '去重後': len(deduplicated_df),
            '減少數量': len(original_df) - len(deduplicated_df),
            '減少比例': f"{((len(original_df) - len(deduplicated_df)) / len(original_df) * 100):.1f}%"
        })
        
        # 數據類型統計
        if 'data_type' in original_df.columns:
            for data_type in original_df['data_type'].unique():
                if pd.isna(data_type):
                    continue
                
                original_count = len(original_df[original_df['data_type'] == data_type])
                deduplicated_count = len(deduplicated_df[deduplicated_df['data_type'] == data_type])
                
                stats.append({
                    '項目': f'{data_type}類型',
                    '原始': original_count,
                    '去重後': deduplicated_count,
                    '減少數量': original_count - deduplicated_count,
                    '減少比例': f"{((original_count - deduplicated_count) / original_count * 100):.1f}%" if original_count > 0 else "0%"
                })
        
        return pd.DataFrame(stats)
    
    def _export_deduplicated_excel(self, deduplicated_df: pd.DataFrame, 
                                  summary_df: pd.DataFrame, original_file_path: str) -> str:
        """導出去重後的Excel"""
        original_path = Path(original_file_path)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"{original_path.stem}_deduplicated_{timestamp}.xlsx"
        output_path = original_path.parent / output_filename
        
        # 準備展示用的DataFrame
        display_columns = {
            'keyword': '主要關鍵字',
            'alternative_keywords': '其他相關關鍵字',
            'value': '提取數值',
            'data_type': '數據類型',
            'confidence': '信心分數',
            'paragraph': '段落內容',
            'page_number': '頁碼',
            'merged_count': '合併數量'
        }
        
        # 選擇和重命名列
        final_columns = ['keyword', 'alternative_keywords', 'value', 'data_type', 
                        'confidence', 'paragraph', 'page_number', 'merged_count']
        
        # 確保所有列都存在
        for col in final_columns:
            if col not in deduplicated_df.columns:
                deduplicated_df[col] = 'N/A'
        
        display_df = deduplicated_df[final_columns].rename(columns=display_columns)
        
        # 截短過長文本
        if '段落內容' in display_df.columns:
            display_df['段落內容'] = display_df['段落內容'].apply(
                lambda x: str(x)[:200] + "..." if len(str(x)) > 200 else str(x)
            )
        
        # 寫入Excel
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            display_df.to_excel(writer, sheet_name='去重結果', index=False)
            summary_df.to_excel(writer, sheet_name='去重統計', index=False)
        
        return str(output_path)
    
    def _print_excel_dedup_summary(self, original_df: pd.DataFrame, 
                                  deduplicated_df: pd.DataFrame, groups: List[List[int]]):
        """打印Excel去重摘要"""
        print("\n" + "="*60)
        print("📊 Excel去重處理摘要")
        print("="*60)
        
        print(f"原始記錄數: {len(original_df)}")
        print(f"去重後記錄數: {len(deduplicated_df)}")
        print(f"刪除重複記錄: {len(original_df) - len(deduplicated_df)}")
        
        if len(original_df) > 0:
            reduction_rate = ((len(original_df) - len(deduplicated_df)) / len(original_df) * 100)
            print(f"去重比例: {reduction_rate:.1f}%")
        
        print(f"\n📋 發現 {len(groups)} 個重複組")
        
        # 顯示重複程度最高的幾組
        group_sizes = [len(group) for group in groups]
        if group_sizes:
            print(f"最大重複組: {max(group_sizes)} 個記錄")
            print(f"平均重複組大小: {np.mean(group_sizes):.1f} 個記錄")

# =============================================================================
# 主要提取器類
# =============================================================================

class ESGExtractor:
    """ESG資料提取器主類"""
    
    def __init__(self, vector_db_path: str = None, enable_llm: bool = True, auto_dedupe: bool = True):
        """
        初始化提取器
        
        Args:
            vector_db_path: 向量資料庫路徑
            enable_llm: 是否啟用LLM增強
            auto_dedupe: 是否自動去重
        """
        self.vector_db_path = vector_db_path or VECTOR_DB_PATH
        self.enable_llm = enable_llm
        self.auto_dedupe = auto_dedupe
        
        # 初始化組件
        self.matcher = EnhancedMatcher()
        self.keyword_config = KeywordConfig()
        self.deduplicator = ESGResultDeduplicator()
        
        # 載入向量資料庫
        self._load_vector_database()
        
        # 初始化LLM（如果啟用）
        if self.enable_llm:
            self._init_llm()
        
        print("✅ ESG提取器初始化完成")
        if self.auto_dedupe:
            print("✅ 自動去重功能已啟用")
    
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
        """初始化LLM"""
        try:
            print(f"🤖 初始化Gemini模型: {GEMINI_MODEL}")
            self.llm = ChatGoogleGenerativeAI(
                model=GEMINI_MODEL,
                google_api_key=GOOGLE_API_KEY,
                temperature=0.1,
                max_tokens=1024
            )
            print("✅ Gemini模型初始化完成")
        except Exception as e:
            print(f"⚠️ LLM初始化失敗: {e}")
            self.enable_llm = False
    
    def stage1_filtering(self, documents: List[Document]) -> Tuple[List[Document], List[ExtractionMatch]]:
        """第一階段篩選：檢查文檔是否包含目標關鍵字"""
        print("🔍 執行第一階段篩選...")
        
        keywords = self.keyword_config.get_all_keywords()
        passed_docs = []
        all_matches = []
        
        for doc in tqdm(documents, desc="第一階段篩選"):
            doc_matches = []
            
            for keyword in keywords:
                is_match, confidence, details = self.matcher.match_keyword(doc.page_content, keyword)
                
                if is_match:
                    keyword_str = keyword if isinstance(keyword, str) else " + ".join(keyword)
                    match = ExtractionMatch(
                        keyword=keyword_str,
                        keyword_type='continuous' if isinstance(keyword, str) else 'discontinuous',
                        confidence=confidence,
                        matched_text=details
                    )
                    doc_matches.append(match)
            
            if doc_matches:
                passed_docs.append(doc)
                all_matches.extend(doc_matches)
        
        print(f"✅ 第一階段完成: {len(passed_docs)}/{len(documents)} 文檔通過")
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
                para_matches = []
                for keyword in keywords:
                    is_match, confidence, details = self.matcher.match_keyword(paragraph, keyword)
                    if is_match:
                        para_matches.append((keyword, confidence, details))
                
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
        
        print("🤖 執行LLM增強...")
        
        enhanced_extractions = []
        
        for extraction in tqdm(extractions, desc="LLM增強"):
            try:
                # 構建驗證提示
                prompt = self._build_verification_prompt(extraction)
                
                # 呼叫LLM
                response = self.llm.invoke(prompt)
                llm_result = self._parse_llm_response(response.content)
                
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
                
            except Exception as e:
                print(f"⚠️ LLM增強失敗: {e}")
                enhanced_extractions.append(extraction)  # 保留原始結果
        
        return enhanced_extractions
    
    def export_to_excel(self, extractions: List[NumericExtraction], summary: ProcessingSummary) -> str:
        """匯出結果到Excel"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(RESULTS_PATH, f"esg_extraction_results_{timestamp}.xlsx")
        
        # 確保輸出目錄存在
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        print(f"📊 匯出結果到Excel: {output_path}")
        
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
            '項目': '處理摘要',
            '總文檔數': summary.total_documents,
            '第一階段通過': summary.stage1_passed,
            '第二階段通過': summary.stage2_passed,
            '總提取結果': summary.total_extractions,
            '處理時間(秒)': round(summary.processing_time, 2),
            '自動去重': '已啟用' if self.auto_dedupe else '未啟用'
        }]
        
        # 寫入Excel
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            # 主要結果
            pd.DataFrame(main_data).to_excel(writer, sheet_name='提取結果', index=False)
            
            # 統計摘要
            pd.DataFrame(stats_data).to_excel(writer, sheet_name='關鍵字統計', index=False)
            
            # 處理摘要
            pd.DataFrame(process_summary).to_excel(writer, sheet_name='處理摘要', index=False)
        
        print(f"✅ Excel檔案已保存")
        return output_path
    
    def run_complete_extraction(self, max_documents: int = 200) -> Tuple[List[NumericExtraction], ProcessingSummary, str]:
        """執行完整的資料提取流程（含自動去重）"""
        start_time = datetime.now()
        print("🚀 開始完整的ESG資料提取流程")
        print("=" * 60)
        
        # 1. 獲取相關文檔
        print("📄 檢索相關文檔...")
        documents = self._retrieve_relevant_documents(max_documents)
        
        # 2. 第一階段篩選
        stage1_docs, stage1_matches = self.stage1_filtering(documents)
        
        # 3. 第二階段篩選
        stage2_extractions = self.stage2_filtering(stage1_docs)
        
        # 4. LLM增強（如果啟用）
        enhanced_extractions = self.llm_enhancement(stage2_extractions)
        
        # 5. 自動去重（如果啟用）
        if self.auto_dedupe:
            print("\n🔄 執行自動去重...")
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
            stage2_passed=len([e for e in final_extractions]),
            total_extractions=len(final_extractions),
            keywords_found=keywords_found,
            processing_time=processing_time
        )
        
        # 7. 匯出結果
        excel_path = self.export_to_excel(final_extractions, summary)
        
        # 8. 顯示最終摘要
        self._print_final_summary(summary, final_extractions)
        
        return final_extractions, summary, excel_path
    
    def manual_deduplicate_results(self, excel_path: str) -> str:
        """手動去重現有的Excel結果文件"""
        return self.deduplicator.deduplicate_excel_file(excel_path)
    
    # =============================================================================
    # 輔助方法
    # =============================================================================
    
    def _retrieve_relevant_documents(self, max_docs: int) -> List[Document]:
        """檢索相關文檔"""
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
    
    def _build_verification_prompt(self, extraction: NumericExtraction) -> str:
        """構建LLM驗證提示"""
        return f"""
請驗證以下數據提取結果的準確性：

關鍵字: {extraction.keyword}
提取值: {extraction.value}
數據類型: {extraction.value_type}

段落內容:
{extraction.paragraph}

請判斷：
1. 提取的數值是否與關鍵字相關？
2. 數值提取是否準確？
3. 數據類型分類是否正確？

請以JSON格式回答：
{{
    "is_relevant": true/false,
    "is_accurate": true/false,
    "confidence": 0-1之間的分數,
    "explanation": "簡短解釋"
}}
"""
    
    def _parse_llm_response(self, response_text: str) -> Optional[Dict]:
        """解析LLM回應"""
        try:
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        except:
            pass
        
        return None
    
    def _print_final_summary(self, summary: ProcessingSummary, extractions: List[NumericExtraction]):
        """打印最終摘要"""
        print("\n" + "=" * 60)
        print("📋 提取完成摘要")
        print("=" * 60)
        print(f"📚 處理文檔數: {summary.total_documents}")
        print(f"🔍 第一階段通過: {summary.stage1_passed}")
        print(f"🔢 第二階段通過: {summary.stage2_passed}")
        print(f"📊 總提取結果: {summary.total_extractions}")
        print(f"⏱️ 處理時間: {summary.processing_time:.2f} 秒")
        print(f"🧹 自動去重: {'已啟用' if self.auto_dedupe else '未啟用'}")
        
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

def main():
    """主函數 - 獨立運行測試"""
    try:
        print("🚀 ESG資料提取器 - 獨立測試模式")
        print("=" * 50)
        
        # 初始化提取器（啟用自動去重）
        extractor = ESGExtractor(enable_llm=True, auto_dedupe=True)
        
        # 執行完整提取
        extractions, summary, excel_path = extractor.run_complete_extraction()
        
        if extractions:
            print(f"\n🎉 提取完成！")
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
            
            # 測試手動去重功能
            print(f"\n🧹 測試手動去重功能...")
            dedupe_path = extractor.manual_deduplicate_results(excel_path)
            if dedupe_path:
                print(f"✅ 手動去重完成: {Path(dedupe_path).name}")
        
        else:
            print("❌ 未找到任何提取結果")
    
    except Exception as e:
        print(f"❌ 執行錯誤: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()