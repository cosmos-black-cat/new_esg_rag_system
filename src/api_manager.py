# =============================================================================
# 修復的LLM增強和API輪換邏輯
# =============================================================================

import json
import re
import time
import random
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

class GeminiAPIManager:
    """改進的API管理器，確保輪換生效"""
    
    def __init__(self, api_keys: List[str], model_name: str = "gemini-1.5-flash"):
        self.api_keys = api_keys
        self.model_name = model_name
        self.current_key_index = 0
        self.key_cooldowns = {}
        self.key_usage_count = {key: 0 for key in api_keys}
        self.last_request_time = 0
        self.min_request_interval = 0.8  # 減少間隔以更快觸發輪換
        self.request_count = 0
        self.rotation_threshold = 20  # 每20次請求強制輪換
        
        print(f"🔑 初始化改進API管理器，共有 {len(api_keys)} 個API key")
        self._initialize_current_llm()
    
    def _initialize_current_llm(self):
        """初始化當前的LLM實例"""
        from langchain_google_genai import ChatGoogleGenerativeAI
        
        current_key = self.api_keys[self.current_key_index]
        self.current_llm = ChatGoogleGenerativeAI(
            model=self.model_name,
            google_api_key=current_key,
            temperature=0.1,
            max_tokens=1024,
            convert_system_message_to_human=True
        )
        print(f"🎯 當前使用API key: {current_key[:10]}... (第{self.current_key_index + 1}個)")
    
    def _should_rotate_api(self) -> bool:
        """判斷是否應該輪換API"""
        # 條件1: 達到輪換閾值
        if self.request_count % self.rotation_threshold == 0 and self.request_count > 0:
            return True
        
        # 條件2: 當前API使用次數過多
        current_key = self.api_keys[self.current_key_index]
        if self.key_usage_count[current_key] > 50:  # 降低閾值
            return True
        
        # 條件3: 隨機輪換（增加輪換機會）
        if len(self.api_keys) > 1 and random.random() < 0.1:  # 10%機率
            return True
        
        return False
    
    def _force_rotate_to_next_key(self):
        """強制輪換到下一個可用的API key"""
        if len(self.api_keys) <= 1:
            return False
        
        original_index = self.current_key_index
        
        # 嘗試找到下一個可用的key
        for i in range(1, len(self.api_keys)):
            next_index = (self.current_key_index + i) % len(self.api_keys)
            next_key = self.api_keys[next_index]
            
            # 檢查是否在冷卻期
            if next_key not in self.key_cooldowns or datetime.now() >= self.key_cooldowns[next_key]:
                self.current_key_index = next_index
                self._initialize_current_llm()
                print(f"🔄 強制輪換API key: 第{original_index + 1}個 → 第{next_index + 1}個")
                return True
        
        print("⚠️ 所有其他API key都在冷卻期，繼續使用當前key")
        return False
    
    def invoke(self, prompt: str, max_retries: int = 3) -> str:
        """改進的API調用方法"""
        self.request_count += 1
        
        # 檢查是否需要輪換
        if self._should_rotate_api():
            self._force_rotate_to_next_key()
        
        for attempt in range(max_retries):
            try:
                # 添加請求延遲
                self._add_request_delay()
                
                # 記錄使用次數
                current_key = self.api_keys[self.current_key_index]
                self.key_usage_count[current_key] += 1
                
                # 調用API
                response = self.current_llm.invoke(prompt)
                
                print(f"✅ API調用成功 (key {self.current_key_index + 1}, 第{self.key_usage_count[current_key]}次使用)")
                return response.content if hasattr(response, 'content') else str(response)
                
            except Exception as e:
                print(f"⚠️ API調用失敗 (嘗試 {attempt + 1}/{max_retries}): {e}")
                
                if "rate limit" in str(e).lower() or "quota" in str(e).lower():
                    # 速率限制，嘗試切換API
                    if self._force_rotate_to_next_key():
                        continue
                
                if attempt == max_retries - 1:
                    raise e
                
                # 等待重試
                wait_time = (attempt + 1) * 2
                print(f"⏳ 等待 {wait_time} 秒後重試...")
                time.sleep(wait_time)
        
        raise Exception("所有重試嘗試都失敗了")
    
    def _add_request_delay(self):
        """添加請求間隔"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.min_request_interval:
            sleep_time = self.min_request_interval - time_since_last
            sleep_time += random.uniform(0, 0.3)  # 添加隨機延遲
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()

class ImprovedLLMEnhancer:
    """改進的LLM增強器，提高成功率"""
    
    def __init__(self, api_manager):
        self.api_manager = api_manager
        self.success_count = 0
        self.total_count = 0
        self.failed_responses = []  # 記錄失敗的響應用於調試
    
    def build_improved_prompt(self, extraction) -> str:
        """構建改進的驗證提示"""
        return f"""請分析以下數據提取結果是否與再生塑膠/塑料相關：

關鍵字: {extraction.keyword}
提取值: {extraction.value}
數據類型: {extraction.value_type}

文本內容:
{extraction.paragraph[:300]}...

判斷標準:
1. 是否與再生塑膠、回收塑料、PCR材料等直接相關？
2. 提取的數值是否確實描述再生材料的使用量、比例或產能？
3. 是否排除了無關主題（如員工、降雨、賽事等）？

請嚴格按照以下JSON格式回答（不要包含其他文字）：
{{
    "is_relevant": true,
    "confidence": 0.85,
    "explanation": "描述相關性的簡短說明"
}}"""
    
    def parse_llm_response(self, response_text: str) -> Optional[Dict]:
        """改進的LLM響應解析"""
        if not response_text:
            return None
        
        # 方法1: 嘗試直接解析JSON
        try:
            # 清理響應文本
            cleaned_response = response_text.strip()
            
            # 尋找JSON內容
            json_match = re.search(r'\{[^{}]*\}', cleaned_response, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                result = json.loads(json_str)
                
                # 驗證必要字段
                if 'is_relevant' in result and 'confidence' in result:
                    return {
                        'is_relevant': bool(result.get('is_relevant', False)),
                        'confidence': float(result.get('confidence', 0.5)),
                        'explanation': str(result.get('explanation', '無說明'))
                    }
        except Exception as e:
            pass
        
        # 方法2: 關鍵字解析
        try:
            response_lower = response_text.lower()
            
            # 判斷相關性
            is_relevant = False
            if any(word in response_lower for word in ['true', '相關', '是', 'relevant', 'yes']):
                is_relevant = True
            elif any(word in response_lower for word in ['false', '不相關', '否', 'irrelevant', 'no']):
                is_relevant = False
            else:
                # 如果無法判斷，默認為不相關（保守策略）
                is_relevant = False
            
            # 提取信心分數
            confidence = 0.5
            confidence_match = re.search(r'(?:confidence|信心).*?(\d+\.?\d*)', response_lower)
            if confidence_match:
                confidence = min(float(confidence_match.group(1)), 1.0)
                if confidence > 5:  # 如果是百分比格式
                    confidence = confidence / 100
            
            return {
                'is_relevant': is_relevant,
                'confidence': confidence,
                'explanation': response_text[:100] + "..." if len(response_text) > 100 else response_text
            }
            
        except Exception as e:
            pass
        
        # 方法3: 默認處理
        return {
            'is_relevant': False,  # 保守策略：無法解析時認為不相關
            'confidence': 0.3,
            'explanation': f"響應解析失敗: {response_text[:50]}..."
        }
    
    def enhance_extraction(self, extraction) -> Tuple[bool, Dict]:
        """增強單個提取結果"""
        self.total_count += 1
        
        try:
            # 構建提示
            prompt = self.build_improved_prompt(extraction)
            
            # 調用LLM
            response_text = self.api_manager.invoke(prompt)
            
            # 解析響應
            llm_result = self.parse_llm_response(response_text)
            
            if llm_result:
                self.success_count += 1
                return True, llm_result
            else:
                self.failed_responses.append({
                    'prompt': prompt[:100],
                    'response': response_text[:200],
                    'keyword': extraction.keyword
                })
                return False, {'is_relevant': False, 'confidence': 0.3, 'explanation': '解析失敗'}
        
        except Exception as e:
            self.failed_responses.append({
                'error': str(e),
                'keyword': extraction.keyword
            })
            return False, {'is_relevant': False, 'confidence': 0.2, 'explanation': f'API調用失敗: {e}'}
    
    def get_enhancement_stats(self) -> Dict:
        """獲取增強統計信息"""
        success_rate = (self.success_count / self.total_count * 100) if self.total_count > 0 else 0
        
        return {
            'success_count': self.success_count,
            'total_count': self.total_count,
            'success_rate': success_rate,
            'failed_responses_count': len(self.failed_responses)
        }
    
    def print_debug_info(self):
        """打印調試信息"""
        print(f"\n🔧 LLM增強調試信息:")
        print(f"成功次數: {self.success_count}")
        print(f"總次數: {self.total_count}")
        print(f"成功率: {(self.success_count/self.total_count*100):.1f}%")
        
        if self.failed_responses:
            print(f"\n❌ 失敗案例 (前3個):")
            for i, failure in enumerate(self.failed_responses[:3], 1):
                print(f"{i}. 關鍵字: {failure.get('keyword', 'N/A')}")
                if 'error' in failure:
                    print(f"   錯誤: {failure['error']}")
                else:
                    print(f"   響應: {failure.get('response', 'N/A')[:100]}...")

class ImprovedESGExtractor:
    """改進的ESG提取器"""
    
    def __init__(self, vector_db_path: str = None, enable_llm: bool = True, auto_dedupe: bool = True):
        self.vector_db_path = vector_db_path or VECTOR_DB_PATH
        self.enable_llm = enable_llm
        self.auto_dedupe = auto_dedupe
        
        # 初始化組件
        self.matcher = EnhancedMatcher()  # 使用增強的匹配器
        self.keyword_config = EnhancedKeywordConfig()  # 使用增強的關鍵字配置
        
        # 載入向量資料庫
        self._load_vector_database()
        
        # 初始化改進的LLM
        if self.enable_llm:
            self._init_improved_llm()
    
    def _init_improved_llm(self):
        """初始化改進的LLM"""
        try:
            print(f"🤖 初始化改進的LLM增強器...")
            
            if len(GEMINI_API_KEYS) > 1:
                print(f"🔄 啟用改進的多API輪換模式，共 {len(GEMINI_API_KEYS)} 個Keys")
                self.api_manager = ImprovedAPIManager(
                    api_keys=GEMINI_API_KEYS,
                    model_name=GEMINI_MODEL
                )
                self.llm_enhancer = ImprovedLLMEnhancer(self.api_manager)
                self.llm_mode = "improved_multi_api"
            else:
                print("🔑 使用改進的單API模式")
                single_api_manager = ImprovedAPIManager(
                    api_keys=GEMINI_API_KEYS,
                    model_name=GEMINI_MODEL
                )
                self.llm_enhancer = ImprovedLLMEnhancer(single_api_manager)
                self.llm_mode = "improved_single_api"
            
            print("✅ 改進的LLM初始化完成")
            
        except Exception as e:
            print(f"⚠️ LLM初始化失敗: {e}")
            self.enable_llm = False
    
    def improved_llm_enhancement(self, extractions: List) -> List:
        """改進的LLM增強過程"""
        if not self.enable_llm or not extractions:
            return extractions
        
        print("🤖 執行改進的LLM增強...")
        print(f"🔄 處理 {len(extractions)} 個提取結果")
        
        enhanced_extractions = []
        
        for i, extraction in enumerate(extractions):
            if i % 10 == 0:  # 每10個顯示進度
                progress = (i / len(extractions)) * 100
                print(f"📊 進度: {progress:.1f}% ({i}/{len(extractions)})")
            
            # LLM增強
            success, llm_result = self.llm_enhancer.enhance_extraction(extraction)
            
            if success and llm_result.get("is_relevant", False):
                # 更新信心分數
                llm_confidence = llm_result.get("confidence", extraction.confidence)
                extraction.confidence = min((extraction.confidence + llm_confidence) / 2, 1.0)
                
                # 添加LLM的解釋
                extraction.context_window += f"\n[LLM驗證]: {llm_result.get('explanation', '')}"
                
                enhanced_extractions.append(extraction)
            elif llm_result.get("confidence", 0) > 0.7:
                # 即使LLM認為不相關，但信心分數很高的情況下保留
                extraction.confidence *= 0.8  # 降低信心分數
                extraction.context_window += f"\n[LLM注意]: {llm_result.get('explanation', '')}"
                enhanced_extractions.append(extraction)
            # 其他情況下丟棄該提取結果
        
        # 打印統計信息
        stats = self.llm_enhancer.get_enhancement_stats()
        print(f"\n✅ 改進的LLM增強完成:")
        print(f"   LLM成功率: {stats['success_rate']:.1f}% ({stats['success_count']}/{stats['total_count']})")
        print(f"   保留結果: {len(enhanced_extractions)}/{len(extractions)} ({len(enhanced_extractions)/len(extractions)*100:.1f}%)")
        
        # 顯示API使用統計
        if hasattr(self, 'api_manager'):
            self.api_manager.print_usage_statistics()
        
        # 顯示調試信息（如果成功率太低）
        if stats['success_rate'] < 70:
            print("⚠️ LLM成功率偏低，顯示調試信息:")
            self.llm_enhancer.print_debug_info()
        
        return enhanced_extractions

# =============================================================================
# 測試函數
# =============================================================================

def test_improved_llm_enhancement():
    """測試改進的LLM增強功能"""
    print("🧪 測試改進的LLM增強功能")
    print("=" * 50)
    
    # 模擬提取結果
    from dataclasses import dataclass
    
    @dataclass
    class MockExtraction:
        keyword: str
        value: str
        value_type: str
        paragraph: str
        confidence: float
        context_window: str = ""
    
    test_extractions = [
        MockExtraction(
            keyword="再生塑膠",
            value="12,000噸",
            value_type="number",
            paragraph="公司再生塑膠材料產能達到12,000噸，主要用於環保包裝產品製造。",
            confidence=0.9
        ),
        MockExtraction(
            keyword="回收",
            value="3,500件",
            value_type="number",
            paragraph="為睽違三年重啟的盛大賽事「垂直馬拉松」打造史上最環保賽衣，預計提供3,500件。",
            confidence=0.7
        ),
        MockExtraction(
            keyword="回收量",
            value="249噸",
            value_type="number",
            paragraph="2023年因月平均降雨量減少18%，致雨水回收量減少249噸。",
            confidence=0.6
        )
    ]
    
    try:
        # 初始化API管理器
        api_manager = ImprovedAPIManager(GEMINI_API_KEYS, GEMINI_MODEL)
        enhancer = ImprovedLLMEnhancer(api_manager)
        
        print(f"測試 {len(test_extractions)} 個提取結果...")
        
        enhanced_results = []
        for extraction in test_extractions:
            print(f"\n測試: {extraction.keyword} - {extraction.value}")
            success, result = enhancer.enhance_extraction(extraction)
            
            print(f"LLM判斷: {'相關' if result.get('is_relevant') else '不相關'}")
            print(f"信心分數: {result.get('confidence', 0):.2f}")
            print(f"說明: {result.get('explanation', 'N/A')[:100]}...")
            
            if success and result.get('is_relevant'):
                enhanced_results.append(extraction)
        
        print(f"\n📊 測試結果:")
        print(f"原始結果: {len(test_extractions)}")
        print(f"通過LLM驗證: {len(enhanced_results)}")
        
        # 顯示統計
        enhancer.print_debug_info()
        api_manager.print_usage_statistics()
        
    except Exception as e:
        print(f"❌ 測試失敗: {e}")

if __name__ == "__main__":
    test_improved_llm_enhancement()