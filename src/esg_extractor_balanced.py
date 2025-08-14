#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ESG資料提取器 v2.4 - 平衡版
在提取準確度和覆蓋率之間取得平衡
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
# 平衡版關鍵字配置
# =============================================================================

class BalancedKeywordConfig:
    """平衡版關鍵字配置，確保基本覆蓋率同時提高精確度"""
    
    # 核心再生塑膠關鍵字
    RECYCLED_PLASTIC_KEYWORDS = {
        # 高相關連續關鍵字
        "high_relevance_continuous": [
            "再生塑膠", "再生塑料", "再生料", "再生PET", "再生PP",
            "回收塑膠", "回收塑料", "回收PP", "回收PET", 
            "rPET", "PCR塑膠", "PCR塑料", "PCR材料",
            "寶特瓶回收", "廢塑膠回收", "塑膠循環",
            "回收造粒", "再生聚酯", "回收聚酯",
            "循環經濟", "物料回收", "材料回收"
        ],
        
        # 中相關連續關鍵字
        "medium_relevance_continuous": [
            "環保塑膠", "綠色材料", "永續材料",
            "廢料回收", "資源回收", "循環利用"
        ],
        
        # 高相關不連續關鍵字組合
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
        ],
        
        # 中相關不連續關鍵字組合
        "medium_relevance_discontinuous": [
            ("環保", "材料"), ("綠色", "產品"), ("永續", "材料"),
            ("廢棄", "物料"), ("資源", "化"), ("循環", "利用")
        ]
    }
    
    # 平衡版排除規則 - 只排除明確無關的內容
    BALANCED_EXCLUSION_RULES = {
        # 明確排除的主題（減少排除項目）
        "exclude_topics": [
            # 職業安全
            "職業災害", "工安", "安全事故", "職災",
            
            # 活動賽事
            "馬拉松", "賽事", "選手", "比賽", "賽衣", "運動",
            
            # 水資源（只排除明確的水處理相關）
            "雨水回收", "廢水處理", "水質監測",
            
            # 改善案數量統計
            "改善案", "改善專案", "案例選拔"
        ],
        
        # 排除的特定上下文片段
        "exclude_contexts": [
            "垂直馬拉松", "史上最環保賽衣", "各界好手",
            "職業災害比率", "工安統計", 
            "節能改善案", "節水改善案", "優良案例",
            "雨水回收量減少", "降雨量減少"
        ],
        
        # 排除的數值模式（更精確）
        "exclude_patterns": [
            r'職業災害.*?\d+(?:\.\d+)?%',
            r'工安.*?\d+(?:\.\d+)?',
            r'馬拉松.*?\d+',
            r'賽事.*?\d+',
            r'改善案.*?\d+\s*件',
            r'案例.*?\d+\s*件'
        ]
    }
    
    # 相關性指標（調整權重，降低要求）
    RELEVANCE_INDICATORS = {
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
        ],
        
        "quantity_indicators": [
            "億支", "萬支", "噸", "公斤", "kg", "萬噸", "千噸",
            "件", "個", "批", "%", "百分比"
        ]
    }
    
    @classmethod
    def get_all_keywords(cls) -> List[Union[str, tuple]]:
        """獲取所有關鍵字"""
        all_keywords = []
        all_keywords.extend(cls.RECYCLED_PLASTIC_KEYWORDS["high_relevance_continuous"])
        all_keywords.extend(cls.RECYCLED_PLASTIC_KEYWORDS["medium_relevance_continuous"])
        all_keywords.extend(cls.RECYCLED_PLASTIC_KEYWORDS["high_relevance_discontinuous"])
        all_keywords.extend(cls.RECYCLED_PLASTIC_KEYWORDS["medium_relevance_discontinuous"])
        return all_keywords

# =============================================================================
# 平衡版匹配引擎
# =============================================================================

class BalancedMatcher:
    """平衡版匹配引擎，確保合理的提取覆蓋率"""
    
    def __init__(self):
        self.config = BalancedKeywordConfig()
        self.max_distance = 300
        
        # 數值匹配模式（保持全面）
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
    
    def comprehensive_relevance_check(self, text: str, keyword: Union[str, tuple]) -> Tuple[bool, float, str]:
        """
        平衡版相關性檢查 - 降低門檻但保持質量
        """
        text_lower = text.lower()
        
        # 第1步：快速排除檢查（只排除明確無關的）
        if self._is_clearly_excluded(text_lower):
            return False, 0.0, "明確無關內容"
        
        # 第2步：關鍵字匹配檢查
        keyword_match, keyword_confidence, keyword_details = self._match_keyword_flexible(text, keyword)
        if not keyword_match:
            return False, 0.0, "關鍵字不匹配"
        
        # 第3步：相關性指標檢查（降低要求）
        relevance_score = self._calculate_balanced_relevance_score(text_lower)
        
        # 第4步：特殊情況加分
        bonus_score = self._calculate_bonus_score(text_lower)
        
        # 計算最終分數（更寬鬆的評分）
        final_score = keyword_confidence * 0.4 + relevance_score * 0.4 + bonus_score * 0.2
        
        # 降低門檻到0.5
        is_relevant = final_score > 0.5
        
        details = f"關鍵字:{keyword_confidence:.2f}, 相關性:{relevance_score:.2f}, 加分:{bonus_score:.2f}"
        
        return is_relevant, final_score, details
    
    def _is_clearly_excluded(self, text: str) -> bool:
        """只排除明確無關的內容"""
        # 檢查明確排除主題
        for topic in self.config.BALANCED_EXCLUSION_RULES["exclude_topics"]:
            if topic in text:
                return True
        
        # 檢查特定排除上下文
        for context in self.config.BALANCED_EXCLUSION_RULES["exclude_contexts"]:
            if context in text:
                return True
        
        # 檢查排除模式
        for pattern in self.config.BALANCED_EXCLUSION_RULES["exclude_patterns"]:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        
        return False
    
    def _calculate_balanced_relevance_score(self, text: str) -> float:
        """計算平衡版相關性分數（降低要求）"""
        total_score = 0.0
        category_weights = {
            "plastic_materials": 0.25,
            "recycling_process": 0.30,
            "production_application": 0.15,
            "environmental_benefit": 0.15,
            "quantity_indicators": 0.15
        }
        
        for category, indicators in self.config.RELEVANCE_INDICATORS.items():
            category_score = 0.0
            for indicator in indicators:
                if indicator in text:
                    category_score += 1
            
            # 正規化分數
            normalized_score = min(category_score / len(indicators), 1.0)
            weight = category_weights.get(category, 0.1)
            total_score += normalized_score * weight
        
        return total_score
    
    def _calculate_bonus_score(self, text: str) -> float:
        """計算加分項目"""
        bonus_score = 0.0
        
        # 特殊情況加分
        bonus_indicators = [
            ("億支", 0.3),  # 寶特瓶數量
            ("寶特瓶", 0.2),
            ("回收數量", 0.2),
            ("減碳", 0.15),
            ("循環經濟", 0.15),
            ("歷年", 0.1),
            ("產能", 0.1)
        ]
        
        for indicator, bonus in bonus_indicators:
            if indicator in text:
                bonus_score += bonus
        
        return min(bonus_score, 1.0)
    
    def _match_keyword_flexible(self, text: str, keyword: Union[str, tuple]) -> Tuple[bool, float, str]:
        """靈活的關鍵字匹配"""
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
            
            # 更寬鬆的距離判斷
            if distance <= 80:
                return True, 0.9, f"近距離匹配({distance}字)"
            elif distance <= 200:
                return True, 0.8, f"中距離匹配({distance}字)"
            elif distance <= self.max_distance:
                return True, 0.6, f"遠距離匹配({distance}字)"
            else:
                return True, 0.4, f"極遠距離匹配({distance}字)"  # 即使很遠也給低分
        
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

# =============================================================================
# 平衡版多文件ESG提取器
# =============================================================================

class BalancedMultiFileESGExtractor:
    """平衡版多文件ESG提取器"""
    
    def __init__(self, enable_llm: bool = True):
        self.enable_llm = enable_llm
        self.matcher = BalancedMatcher()
        self.keyword_config = BalancedKeywordConfig()
        
        if self.enable_llm:
            self._init_llm()
        
        print("✅ 平衡版多文件ESG提取器初始化完成")

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
        """處理單個文檔 - 平衡版"""
        start_time = datetime.now()
        print(f"\n⚖️ 平衡版處理文檔: {doc_info.company_name} - {doc_info.report_year}")
        print("=" * 60)
        
        # 1. 載入向量資料庫
        db = self._load_vector_database(doc_info.db_path)
        
        # 2. 增強文檔檢索
        documents = self._enhanced_document_retrieval(db, max_documents)
        
        # 3. 平衡版篩選
        extractions = self._balanced_filtering(documents, doc_info)
        
        # 4. 後處理和去重
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
        print(f"⚖️ 開始平衡版批量處理 {len(docs_info)} 個文檔")
        print("=" * 60)
        
        results = {}
        
        for pdf_path, doc_info in docs_info.items():
            try:
                print(f"\n📄 處理: {doc_info.company_name} - {doc_info.report_year}")
                
                extractions, summary, excel_path = self.process_single_document(doc_info, max_documents)
                
                results[pdf_path] = (extractions, summary, excel_path)
                
                print(f"✅ 完成: 生成 {len(extractions)} 個平衡結果 -> {Path(excel_path).name}")
                
            except Exception as e:
                print(f"❌ 處理失敗 {doc_info.company_name}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        print(f"\n🎉 平衡版批量處理完成！成功處理 {len(results)}/{len(docs_info)} 個文檔")
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
    
    def _enhanced_document_retrieval(self, db, max_docs: int) -> List[Document]:
        """增強的文檔檢索"""
        keywords = self.keyword_config.get_all_keywords()
        all_docs = []
        
        # 策略1: 關鍵字檢索
        print("   🔍 執行關鍵字檢索...")
        for keyword in keywords[:20]:  # 增加關鍵字數量
            search_term = keyword if isinstance(keyword, str) else " ".join(keyword)
            docs = db.similarity_search(search_term, k=15)
            all_docs.extend(docs)
        
        # 策略2: 廣泛主題檢索
        print("   🔍 執行主題檢索...")
        topic_queries = [
            "塑膠 回收 材料",
            "寶特瓶 循環 經濟",
            "再生 環保 永續",
            "廢料 處理 利用",
            "減碳 效益 環境"
        ]
        
        for query in topic_queries:
            docs = db.similarity_search(query, k=20)
            all_docs.extend(docs)
        
        # 策略3: 數值檢索
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
    
    def _balanced_filtering(self, documents: List[Document], doc_info: DocumentInfo) -> List[NumericExtraction]:
        """平衡版篩選 - 確保基本覆蓋率"""
        print("⚖️ 執行平衡版篩選...")
        
        keywords = self.keyword_config.get_all_keywords()
        extractions = []
        
        for doc in tqdm(documents, desc="平衡篩選"):
            # 使用多種段落分割策略
            paragraphs = self._flexible_paragraph_split(doc.page_content)
            page_num = doc.metadata.get('page', '未知')
            
            for para_idx, paragraph in enumerate(paragraphs):
                if len(paragraph.strip()) < 15:  # 降低最小長度要求
                    continue
                
                # 對每個關鍵字進行匹配
                for keyword in keywords:
                    is_relevant, relevance_score, details = self.matcher.comprehensive_relevance_check(paragraph, keyword)
                    
                    if is_relevant and relevance_score > 0.5:  # 平衡門檻
                        # 提取數值
                        numbers, percentages = self.matcher.extract_numbers_and_percentages(paragraph)
                        
                        # 如果沒有明確數值，但有重要關鍵字，也保留
                        if not numbers and not percentages:
                            if relevance_score > 0.7:  # 高相關性的描述性內容
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
                                    report_year=doc_info.report_year
                                )
                                extractions.append(extraction)
                        
                        # 為數值創建提取結果
                        for number in numbers:
                            keyword_str = keyword if isinstance(keyword, str) else " + ".join(keyword)
                            
                            extraction = NumericExtraction(
                                keyword=keyword_str,
                                value=number,
                                value_type='number',
                                unit=self._extract_unit(number),
                                paragraph=paragraph.strip(),
                                paragraph_number=para_idx + 1,
                                page_number=f"第{page_num}頁",
                                confidence=relevance_score,
                                context_window=self._get_context_window(doc.page_content, paragraph),
                                company_name=doc_info.company_name,
                                report_year=doc_info.report_year
                            )
                            extractions.append(extraction)
                        
                        # 為百分比創建提取結果
                        for percentage in percentages:
                            keyword_str = keyword if isinstance(keyword, str) else " + ".join(keyword)
                            
                            extraction = NumericExtraction(
                                keyword=keyword_str,
                                value=percentage,
                                value_type='percentage',
                                unit='%',
                                paragraph=paragraph.strip(),
                                paragraph_number=para_idx + 1,
                                page_number=f"第{page_num}頁",
                                confidence=relevance_score,
                                context_window=self._get_context_window(doc.page_content, paragraph),
                                company_name=doc_info.company_name,
                                report_year=doc_info.report_year
                            )
                            extractions.append(extraction)
        
        print(f"✅ 平衡篩選完成: 找到 {len(extractions)} 個候選結果")
        return extractions
    
    def _flexible_paragraph_split(self, text: str) -> List[str]:
        """靈活的段落分割"""
        # 嘗試多種分割方式
        paragraphs = []
        
        # 方式1: 標準分割
        standard_paras = re.split(r'\n{2,}|\r{2,}', text)
        paragraphs.extend([p.strip() for p in standard_paras if len(p.strip()) >= 15])
        
        # 方式2: 句號分割（對於緊密文本）
        sentence_paras = re.split(r'。{2,}|\.{2,}', text)
        paragraphs.extend([p.strip() for p in sentence_paras if len(p.strip()) >= 30])
        
        # 方式3: 保持原文的大塊文本（對於表格）
        if len(text.strip()) >= 50:
            paragraphs.append(text.strip())
        
        # 去重
        unique_paragraphs = []
        seen = set()
        for para in paragraphs:
            para_hash = hash(para[:100])  # 使用前100字符作為標識
            if para_hash not in seen:
                seen.add(para_hash)
                unique_paragraphs.append(para)
        
        return unique_paragraphs
    
    def _post_process_extractions(self, extractions: List[NumericExtraction]) -> List[NumericExtraction]:
        """後處理和去重"""
        if not extractions:
            return extractions
        
        print(f"🔧 後處理 {len(extractions)} 個提取結果...")
        
        # 去重
        unique_extractions = []
        seen_combinations = set()
        
        for extraction in extractions:
            # 創建唯一標識
            identifier = (
                extraction.keyword,
                extraction.value,
                extraction.paragraph[:50]  # 使用段落前50字符
            )
            
            if identifier not in seen_combinations:
                seen_combinations.add(identifier)
                unique_extractions.append(extraction)
        
        # 按信心分數排序
        unique_extractions.sort(key=lambda x: x.confidence, reverse=True)
        
        print(f"✅ 後處理完成: 保留 {len(unique_extractions)} 個唯一結果")
        return unique_extractions
    
    def _export_to_excel(self, extractions: List[NumericExtraction], summary: ProcessingSummary, doc_info: DocumentInfo) -> str:
        """匯出結果到Excel"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        company_safe = re.sub(r'[^\w\s-]', '', doc_info.company_name).strip()
        
        output_filename = f"ESG提取結果_平衡版_{company_safe}_{doc_info.report_year}_{timestamp}.xlsx"
        output_path = os.path.join(RESULTS_PATH, output_filename)
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        print(f"📊 匯出平衡版結果到Excel: {output_filename}")
        
        # 準備主要數據
        main_data = []
        
        # 第一行：公司信息
        header_row = {
            '關鍵字': f"公司: {doc_info.company_name}",
            '提取數值': f"報告年度: {doc_info.report_year}",
            '數據類型': f"處理時間: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            '單位': '',
            '段落內容': f"平衡版提取結果: {len(extractions)} 項",
            '段落編號': '',
            '頁碼': '',
            '信心分數': '',
            '上下文': f"提取器版本: v2.4 平衡版"
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
            pd.DataFrame(main_data).to_excel(writer, sheet_name='平衡版提取結果', index=False)
            pd.DataFrame(stats_data).to_excel(writer, sheet_name='關鍵字統計', index=False)
            
            # 處理摘要
            summary_data = [{
                '公司名稱': summary.company_name,
                '報告年度': summary.report_year,
                '總文檔數': summary.total_documents,
                '總提取結果': summary.total_extractions,
                '處理時間(秒)': round(summary.processing_time, 2),
                '處理日期': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                '提取器版本': 'v2.4 平衡版'
            }]
            pd.DataFrame(summary_data).to_excel(writer, sheet_name='處理摘要', index=False)
        
        print(f"✅ 平衡版Excel檔案已保存")
        return output_path
    
    # =============================================================================
    # 輔助方法
    # =============================================================================
    
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
    print("⚖️ 平衡版ESG提取器測試模式")
    
    extractor = BalancedMultiFileESGExtractor(enable_llm=False)
    print("✅ 平衡版提取器初始化完成")

if __name__ == "__main__":
    main()