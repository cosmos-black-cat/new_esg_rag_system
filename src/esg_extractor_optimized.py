#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ESG資料提取器 - 優化版 v3.0
精確提取再生塑膠相關數據，減少LLM調用次數
"""

import os
import re
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from datetime import datetime
from tqdm import tqdm
import numpy as np
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI

from config import *

@dataclass
class ExtractionResult:
    """提取結果"""
    keyword: str
    value: str
    value_type: str
    unit: str
    paragraph: str
    page_number: str
    confidence: float
    context: str

class PreciseKeywordMatcher:
    """精確關鍵字匹配器"""
    
    def __init__(self):
        # 核心再生塑膠關鍵字（更精確）
        self.core_keywords = {
            "再生塑膠": ["再生塑膠", "再生塑料"],
            "再生聚酯": ["再生聚酯", "回收聚酯", "rPET", "再生PET"],
            "PP回收": ["PP回收", "PP棧板回收", "回收PP", "再生PP"],
            "MLCC回收": ["MLCC回收", "離型膜回收"],
            "寶特瓶回收": ["寶特瓶回收", "回收寶特瓶"],
            "織物回收": ["織物回收", "纖維回收", "成衣回收"],
            "再生料": ["再生料", "回收料", "PCR材料"]
        }
        
        # 排除關鍵字（用於過濾不相關內容）
        self.exclude_keywords = [
            "職業災害", "環保罰單", "地下水", "監測", "執行次數", 
            "改善案", "節能", "節水", "廢水", "雨水", "用電量",
            "用汽量", "燃料", "溫室氣體", "CO2", "碳排", "罰單",
            "違反", "稽查", "檢查", "合規", "法規", "標準"
        ]
        
        # 數值模式（更精確）
        self.number_patterns = [
            r'\d+(?:,\d{3})*(?:\.\d+)?\s*(?:萬)?(?:噸|公斤|kg|g)\b',
            r'\d+(?:,\d{3})*(?:\.\d+)?\s*噸/月\b',
            r'\d+(?:,\d{3})*(?:\.\d+)?\s*(?:萬)?(?:件|個|支|批)\b',
            r'\d+(?:,\d{3})*(?:\.\d+)?\s*億支\b'
        ]
        
        self.percentage_patterns = [
            r'\d+(?:\.\d+)?\s*%\b',
            r'\d+(?:\.\d+)?\s*％\b',
            r'百分之\d+(?:\.\d+)?\b'
        ]

    def is_relevant_context(self, text: str) -> bool:
        """檢查文本是否為相關上下文"""
        text_lower = text.lower()
        
        # 檢查是否包含排除關鍵字
        for exclude_word in self.exclude_keywords:
            if exclude_word in text_lower:
                return False
        
        # 檢查是否包含塑膠/材料相關詞彙
        material_keywords = [
            "塑膠", "塑料", "聚酯", "PET", "PP", "材料", 
            "寶特瓶", "回收", "再生", "環保", "循環"
        ]
        
        has_material = any(word in text_lower for word in material_keywords)
        return has_material

    def match_keywords(self, text: str) -> List[Tuple[str, float]]:
        """匹配關鍵字"""
        matches = []
        text_lower = text.lower()
        
        # 先檢查是否為相關上下文
        if not self.is_relevant_context(text):
            return matches
        
        for category, keywords in self.core_keywords.items():
            for keyword in keywords:
                if keyword.lower() in text_lower:
                    # 計算匹配位置的上下文相關性
                    pos = text_lower.find(keyword.lower())
                    context_window = text_lower[max(0, pos-50):pos+len(keyword)+50]
                    
                    # 檢查上下文是否真的與塑膠回收相關
                    if self._is_plastic_recycling_context(context_window):
                        confidence = self._calculate_confidence(keyword, context_window)
                        matches.append((category, confidence))
        
        return matches

    def _is_plastic_recycling_context(self, context: str) -> bool:
        """檢查是否為塑膠回收相關上下文"""
        positive_indicators = [
            "回收", "再生", "環保", "循環", "減碳", "產能", "產量",
            "製造", "生產", "應用", "材料", "製品", "產品"
        ]
        
        negative_indicators = [
            "災害", "罰單", "監測", "檢查", "稽查", "用電", "用水",
            "燃料", "廢水", "雨水", "法規", "合規", "標準"
        ]
        
        positive_score = sum(1 for word in positive_indicators if word in context)
        negative_score = sum(1 for word in negative_indicators if word in context)
        
        return positive_score > negative_score and positive_score >= 1

    def _calculate_confidence(self, keyword: str, context: str) -> float:
        """計算匹配信心分數"""
        base_confidence = 0.7
        
        # 直接關鍵字有更高信心
        if any(direct in keyword for direct in ["再生塑膠", "再生聚酯", "PP回收"]):
            base_confidence = 0.9
        
        # 檢查數值的存在
        numbers = re.findall(r'\d+(?:,\d{3})*(?:\.\d+)?', context)
        if numbers:
            base_confidence += 0.1
        
        # 檢查單位的存在
        units = re.findall(r'(?:噸|kg|件|個|%|億支)', context)
        if units:
            base_confidence += 0.1
        
        return min(base_confidence, 1.0)

    def extract_values(self, text: str) -> Tuple[List[str], List[str]]:
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

class ESGExtractorOptimized:
    """優化版ESG資料提取器"""
    
    def __init__(self):
        self.matcher = PreciseKeywordMatcher()
        self.vector_db_path = VECTOR_DB_PATH
        self._load_vector_database()
        self._init_llm()
        
        print("✅ 優化版ESG提取器初始化完成")

    def _load_vector_database(self):
        """載入向量資料庫"""
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

    def _init_llm(self):
        """初始化LLM（僅在需要時使用）"""
        try:
            self.llm = ChatGoogleGenerativeAI(
                model=GEMINI_MODEL,
                google_api_key=GOOGLE_API_KEY,
                temperature=0,
                max_tokens=512
            )
            self.llm_available = True
            print("✅ LLM初始化完成（僅用於批量驗證）")
        except Exception as e:
            print(f"⚠️ LLM初始化失敗: {e}")
            self.llm_available = False

    def retrieve_documents(self, max_docs: int = 100) -> List[Document]:
        """檢索相關文檔"""
        search_terms = [
            "再生塑膠", "再生聚酯", "PP回收", "寶特瓶回收", 
            "MLCC回收", "織物回收", "回收料", "循環經濟"
        ]
        
        all_docs = []
        for term in search_terms:
            docs = self.db.similarity_search(term, k=15)
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

    def extract_from_documents(self, documents: List[Document]) -> List[ExtractionResult]:
        """從文檔中提取數據"""
        print("🔍 開始精確提取...")
        extractions = []
        
        for doc in tqdm(documents, desc="處理文檔"):
            # 分割成段落
            paragraphs = self._split_into_paragraphs(doc.page_content)
            page_num = doc.metadata.get('page', '未知')
            
            for para_idx, paragraph in enumerate(paragraphs):
                if len(paragraph.strip()) < 20:
                    continue
                
                # 匹配關鍵字
                keyword_matches = self.matcher.match_keywords(paragraph)
                
                if keyword_matches:
                    # 提取數值
                    numbers, percentages = self.matcher.extract_values(paragraph)
                    
                    if numbers or percentages:
                        # 為每個匹配創建提取結果
                        for keyword, confidence in keyword_matches:
                            # 處理數值
                            for number in numbers:
                                extraction = ExtractionResult(
                                    keyword=keyword,
                                    value=number.strip(),
                                    value_type='number',
                                    unit=self._extract_unit(number),
                                    paragraph=paragraph.strip(),
                                    page_number=f"第{page_num}頁",
                                    confidence=confidence,
                                    context=self._get_context(doc.page_content, paragraph)
                                )
                                extractions.append(extraction)
                            
                            # 處理百分比
                            for percentage in percentages:
                                extraction = ExtractionResult(
                                    keyword=keyword,
                                    value=percentage.strip(),
                                    value_type='percentage',
                                    unit='%',
                                    paragraph=paragraph.strip(),
                                    page_number=f"第{page_num}頁",
                                    confidence=confidence,
                                    context=self._get_context(doc.page_content, paragraph)
                                )
                                extractions.append(extraction)
        
        print(f"✅ 初步提取完成: {len(extractions)} 個結果")
        return extractions

    def batch_llm_validation(self, extractions: List[ExtractionResult]) -> List[ExtractionResult]:
        """批量LLM驗證（減少API調用）"""
        if not self.llm_available or not extractions:
            return extractions
        
        print("🤖 執行批量LLM驗證...")
        
        # 將提取結果分批，每批5個
        batch_size = 5
        validated_extractions = []
        
        for i in range(0, len(extractions), batch_size):
            batch = extractions[i:i+batch_size]
            
            try:
                # 構建批量驗證提示
                batch_text = self._build_batch_prompt(batch)
                response = self.llm.invoke(batch_text)
                
                # 解析批量回應
                validations = self._parse_batch_response(response.content, len(batch))
                
                # 更新提取結果
                for j, extraction in enumerate(batch):
                    if j < len(validations) and validations[j].get('is_relevant', True):
                        extraction.confidence = min(
                            (extraction.confidence + validations[j].get('confidence', 0.5)) / 2,
                            1.0
                        )
                        validated_extractions.append(extraction)
                
                print(f"✅ 批量驗證完成 {i+1}-{min(i+batch_size, len(extractions))}")
                
            except Exception as e:
                print(f"⚠️ 批量驗證失敗: {e}")
                # 如果LLM失敗，保留原始結果
                validated_extractions.extend(batch)
        
        print(f"🎯 LLM驗證後: {len(validated_extractions)} 個有效結果")
        return validated_extractions

    def _build_batch_prompt(self, batch: List[ExtractionResult]) -> str:
        """構建批量驗證提示"""
        prompt = "請驗證以下數據提取是否與再生塑膠相關。回答格式：每行一個數字(1-5)，1=相關，0=不相關\n\n"
        
        for i, extraction in enumerate(batch, 1):
            prompt += f"{i}. 關鍵字：{extraction.keyword}\n"
            prompt += f"   數值：{extraction.value}\n"
            prompt += f"   段落：{extraction.paragraph[:100]}...\n\n"
        
        prompt += "請只回答數字序列，例如：1,0,1,1,0"
        return prompt

    def _parse_batch_response(self, response: str, batch_size: int) -> List[Dict]:
        """解析批量回應"""
        try:
            # 尋找數字序列
            numbers = re.findall(r'[01]', response)
            
            validations = []
            for i in range(batch_size):
                if i < len(numbers):
                    is_relevant = numbers[i] == '1'
                    validations.append({
                        'is_relevant': is_relevant,
                        'confidence': 0.8 if is_relevant else 0.3
                    })
                else:
                    validations.append({'is_relevant': True, 'confidence': 0.5})
            
            return validations
        except:
            # 如果解析失敗，全部標記為相關
            return [{'is_relevant': True, 'confidence': 0.5}] * batch_size

    def deduplicate_results(self, extractions: List[ExtractionResult]) -> List[ExtractionResult]:
        """去重結果"""
        if not extractions:
            return extractions
        
        print("🧹 執行智能去重...")
        
        # 按關鍵字和數值分組
        groups = {}
        for extraction in extractions:
            key = f"{extraction.keyword}_{extraction.value}_{extraction.value_type}"
            if key not in groups:
                groups[key] = []
            groups[key].append(extraction)
        
        deduplicated = []
        for group in groups.values():
            if len(group) == 1:
                deduplicated.extend(group)
            else:
                # 選擇信心分數最高的
                best = max(group, key=lambda x: x.confidence)
                # 合併頁碼信息
                pages = list(set([e.page_number for e in group]))
                best.page_number = " | ".join(pages)
                best.context += f" [合併了{len(group)}個重複結果]"
                deduplicated.append(best)
        
        print(f"✅ 去重完成: {len(extractions)} → {len(deduplicated)}")
        return deduplicated

    def export_to_excel(self, extractions: List[ExtractionResult]) -> str:
        """匯出到Excel"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(RESULTS_PATH, f"esg_extraction_optimized_{timestamp}.xlsx")
        
        # 確保輸出目錄存在
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        print(f"📊 匯出結果到Excel: {output_path}")
        
        # 準備數據
        data = []
        for extraction in extractions:
            data.append({
                '關鍵字類別': extraction.keyword,
                '提取數值': extraction.value,
                '數據類型': extraction.value_type,
                '單位': extraction.unit,
                '信心分數': round(extraction.confidence, 3),
                '頁碼': extraction.page_number,
                '段落內容': extraction.paragraph[:200] + "..." if len(extraction.paragraph) > 200 else extraction.paragraph
            })
        
        # 統計數據
        stats_data = []
        keyword_counts = {}
        for extraction in extractions:
            keyword_counts[extraction.keyword] = keyword_counts.get(extraction.keyword, 0) + 1
        
        for keyword, count in keyword_counts.items():
            keyword_extractions = [e for e in extractions if e.keyword == keyword]
            avg_confidence = np.mean([e.confidence for e in keyword_extractions])
            
            stats_data.append({
                '關鍵字類別': keyword,
                '提取數量': count,
                '平均信心分數': round(avg_confidence, 3),
                '數值類型數': len([e for e in keyword_extractions if e.value_type == 'number']),
                '百分比類型數': len([e for e in keyword_extractions if e.value_type == 'percentage'])
            })
        
        # 寫入Excel
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            pd.DataFrame(data).to_excel(writer, sheet_name='提取結果', index=False)
            pd.DataFrame(stats_data).to_excel(writer, sheet_name='統計摘要', index=False)
        
        print(f"✅ Excel檔案已保存")
        return output_path

    def run_complete_extraction(self) -> Tuple[List[ExtractionResult], Dict, str]:
        """執行完整提取流程"""
        start_time = datetime.now()
        print("🚀 開始優化版資料提取流程")
        print("=" * 50)
        
        # 1. 檢索文檔
        documents = self.retrieve_documents()
        
        # 2. 提取數據
        extractions = self.extract_from_documents(documents)
        
        # 3. 批量LLM驗證（減少API調用）
        validated_extractions = self.batch_llm_validation(extractions)
        
        # 4. 去重
        final_extractions = self.deduplicate_results(validated_extractions)
        
        # 5. 匯出結果
        excel_path = self.export_to_excel(final_extractions)
        
        # 6. 生成摘要
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        
        summary = {
            'total_documents': len(documents),
            'total_extractions': len(final_extractions),
            'processing_time': processing_time,
            'keywords_found': len(set([e.keyword for e in final_extractions]))
        }
        
        # 7. 顯示摘要
        self._print_summary(summary, final_extractions)
        
        return final_extractions, summary, excel_path

    def _split_into_paragraphs(self, text: str) -> List[str]:
        """分割段落"""
        paragraphs = re.split(r'\n{2,}|\r{2,}|。{2,}', text)
        return [p.strip() for p in paragraphs if len(p.strip()) >= 20]

    def _extract_unit(self, value_str: str) -> str:
        """提取單位"""
        units = re.findall(r'[a-zA-Z\u4e00-\u9fff/]+', value_str)
        return units[-1] if units else ""

    def _get_context(self, full_text: str, paragraph: str, window_size: int = 80) -> str:
        """獲取上下文"""
        try:
            pos = full_text.find(paragraph)
            if pos == -1:
                return paragraph[:150]
            
            start = max(0, pos - window_size)
            end = min(len(full_text), pos + len(paragraph) + window_size)
            return full_text[start:end]
        except:
            return paragraph[:150]

    def _print_summary(self, summary: Dict, extractions: List[ExtractionResult]):
        """顯示摘要"""
        print("\n" + "=" * 50)
        print("📋 優化版提取完成摘要")
        print("=" * 50)
        print(f"📚 處理文檔數: {summary['total_documents']}")
        print(f"📊 總提取結果: {summary['total_extractions']}")
        print(f"🎯 關鍵字類別: {summary['keywords_found']}")
        print(f"⏱️ 處理時間: {summary['processing_time']:.2f} 秒")
        
        if extractions:
            keyword_counts = {}
            for extraction in extractions:
                keyword_counts[extraction.keyword] = keyword_counts.get(extraction.keyword, 0) + 1
            
            print(f"\n📈 關鍵字分布:")
            for keyword, count in sorted(keyword_counts.items(), key=lambda x: x[1], reverse=True):
                print(f"   {keyword}: {count} 個結果")
            
            avg_confidence = np.mean([e.confidence for e in extractions])
            print(f"\n📊 平均信心分數: {avg_confidence:.3f}")

def main():
    """主函數 - 獨立測試"""
    try:
        extractor = ESGExtractorOptimized()
        extractions, summary, excel_path = extractor.run_complete_extraction()
        
        if extractions:
            print(f"\n🎉 提取完成！共 {len(extractions)} 個結果")
            print(f"📁 結果已保存至: {excel_path}")
        else:
            print("❌ 未找到任何相關數據")
    
    except Exception as e:
        print(f"❌ 執行錯誤: {e}")

if __name__ == "__main__":
    main()