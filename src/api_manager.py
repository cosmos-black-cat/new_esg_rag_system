import time
import random
from typing import List, Optional
from datetime import datetime, timedelta
from langchain_google_genai import ChatGoogleGenerativeAI
import google.generativeai as genai
from google.api_core.exceptions import ResourceExhausted, ServiceUnavailable, TooManyRequests

class GeminiAPIManager:
    """Gemini API多key管理器，支持自動輪換和速率限制處理 - 完整修復版"""
    
    def __init__(self, api_keys: List[str], model_name: str = "models/gemini-1.5-flash"):
        """
        初始化API管理器
        
        Args:
            api_keys: API key列表
            model_name: Gemini模型名稱
        """
        self.api_keys = api_keys
        self.model_name = model_name
        self.current_key_index = 0
        self.key_cooldowns = {}  # 記錄每個key的冷卻時間
        self.key_usage_count = {key: 0 for key in api_keys}  # 記錄每個key的使用次數
        self.last_request_time = 0
        self.min_request_interval = 1  # 降低請求間隔以提高速度
        self.request_count = 0  # 總請求計數
        self.rotation_threshold = 15  # 每15次請求強制輪換
        
        print(f"🔑 初始化Gemini API管理器，共有 {len(api_keys)} 個API key")
        self._initialize_current_llm()
    
    def _initialize_current_llm(self):
        """初始化當前的LLM實例"""
        current_key = self.api_keys[self.current_key_index]
        self.current_llm = ChatGoogleGenerativeAI(
            model=self.model_name,
            google_api_key=current_key,
            temperature=0,
            convert_system_message_to_human=True
        )
        print(f"🎯 當前使用API key: {current_key[:10]}... (第{self.current_key_index + 1}個)")
    
    def _is_key_available(self, key_index: int) -> bool:
        """檢查指定的API key是否可用"""
        key = self.api_keys[key_index]
        
        # 檢查是否在冷卻期
        if key in self.key_cooldowns:
            if datetime.now() < self.key_cooldowns[key]:
                remaining = (self.key_cooldowns[key] - datetime.now()).total_seconds()
                print(f"⏳ API key {key[:10]}... 還在冷卻期，剩餘 {remaining:.1f} 秒")
                return False
        
        return True
    
    def _get_next_available_key(self) -> Optional[int]:
        """獲取下一個可用的API key索引"""
        # 從當前key的下一個開始尋找
        for i in range(len(self.api_keys)):
            key_index = (self.current_key_index + i + 1) % len(self.api_keys)
            if self._is_key_available(key_index):
                return key_index
        
        return None
    
    def _should_rotate_api(self) -> bool:
        """判斷是否應該輪換API - 改進版"""
        # 條件1: 達到輪換閾值
        if self.request_count > 0 and self.request_count % self.rotation_threshold == 0:
            return True
        
        # 條件2: 當前API使用次數過多
        current_key = self.api_keys[self.current_key_index]
        if self.key_usage_count[current_key] > 30:  # 降低閾值
            return True
        
        # 條件3: 隨機輪換（增加輪換機會）
        if len(self.api_keys) > 1 and random.random() < 0.15:  # 15%機率
            return True
        
        return False
    
    def _switch_to_next_key(self) -> bool:
        """切換到下一個可用的API key"""
        next_key_index = self._get_next_available_key()
        
        if next_key_index is None:
            print("⚠️ 所有API key都不可用")
            return False
        
        old_index = self.current_key_index
        self.current_key_index = next_key_index
        self._initialize_current_llm()
        
        print(f"🔄 切換API key: 第{old_index + 1}個 → 第{next_key_index + 1}個")
        return True
    
    def _force_rotate_to_next_key(self):
        """強制輪換到下一個API key - 新增方法"""
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
    
    def _set_key_cooldown(self, key_index: int, cooldown_minutes: int = 1):
        """設置API key的冷卻時間"""
        key = self.api_keys[key_index]
        cooldown_until = datetime.now() + timedelta(minutes=cooldown_minutes)
        self.key_cooldowns[key] = cooldown_until
        
        print(f"❄️ API key {key[:10]}... 進入冷卻期 {cooldown_minutes} 分鐘")
    
    def _wait_for_all_keys_available(self):
        """等待所有API key可用"""
        print("🛑 所有API key都已達到限制")
        
        # 計算最短等待時間
        min_wait_time = 10 * 60  # 預設10分鐘
        
        if self.key_cooldowns:
            # 找到最快可用的key的時間
            min_cooldown = min(self.key_cooldowns.values())
            wait_seconds = max((min_cooldown - datetime.now()).total_seconds(), min_wait_time)
        else:
            wait_seconds = min_wait_time
        
        print(f"⏰ 等待 {wait_seconds/60:.1f} 分鐘後重試...")
        
        # 顯示倒計時
        for remaining in range(int(wait_seconds), 0, -30):
            print(f"⏳ 剩餘等待時間: {remaining//60}分{remaining%60}秒...")
            time.sleep(min(30, remaining))
        
        # 清除所有冷卻時間
        self.key_cooldowns.clear()
        print("✅ 等待完成，重置所有API key狀態")
    
    def _handle_rate_limit_error(self, error):
        """處理速率限制錯誤"""
        error_str = str(error).lower()
        
        if "quota exceeded" in error_str or "rate limit" in error_str:
            print(f"🚫 API key達到速率限制: {error}")
            
            # 設置當前key的冷卻時間
            self._set_key_cooldown(self.current_key_index, cooldown_minutes=2)
            
            # 嘗試切換到下一個key
            if self._switch_to_next_key():
                return True
            else:
                # 所有key都不可用，等待
                self._wait_for_all_keys_available()
                self.current_key_index = 0
                self._initialize_current_llm()
                return True
        
        return False
    
    def _add_request_delay(self):
        """添加請求間隔以避免過於頻繁的請求"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.min_request_interval:
            sleep_time = self.min_request_interval - time_since_last
            # 添加隨機延遲以避免所有請求同時發送
            sleep_time += random.uniform(0, 0.3)
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()
    
    def invoke(self, prompt: str, max_retries: int = 3) -> str:
        """
        調用Gemini API，支持自動重試和key輪換 - 改進版
        
        Args:
            prompt: 輸入提示
            max_retries: 最大重試次數
            
        Returns:
            API響應內容
        """
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
                
                # 成功則返回結果
                print(f"✅ API調用成功 (key {self.current_key_index + 1}, 第{self.key_usage_count[current_key]}次使用)")
                return response.content if hasattr(response, 'content') else str(response)
                
            except (ResourceExhausted, TooManyRequests, ServiceUnavailable) as e:
                print(f"⚠️ API限制錯誤 (嘗試 {attempt + 1}/{max_retries}): {e}")
                
                # 處理速率限制
                if self._handle_rate_limit_error(e):
                    continue
                else:
                    raise e
                    
            except Exception as e:
                print(f"❌ API調用錯誤 (嘗試 {attempt + 1}/{max_retries}): {e}")
                
                # 如果是API限制相關錯誤，嘗試輪換
                if "rate limit" in str(e).lower() or "quota" in str(e).lower():
                    if self._force_rotate_to_next_key():
                        continue
                
                if attempt == max_retries - 1:
                    raise e
                
                # 等待後重試
                wait_time = (attempt + 1) * 2
                print(f"⏳ 等待 {wait_time} 秒後重試...")
                time.sleep(wait_time)
        
        raise Exception("所有重試嘗試都失敗了")
    
    def get_usage_statistics(self) -> dict:
        """獲取API使用統計 - 保持原有接口"""
        total_usage = sum(self.key_usage_count.values())
        
        stats = {
            "total_requests": total_usage,
            "keys_usage": {}
        }
        
        for i, key in enumerate(self.api_keys):
            usage = self.key_usage_count[key]
            stats["keys_usage"][f"key_{i+1}"] = {
                "key_preview": key[:10] + "...",
                "usage_count": usage,
                "usage_percentage": (usage / total_usage * 100) if total_usage > 0 else 0,
                "is_cooling": key in self.key_cooldowns
            }
        
        return stats
    
    def print_usage_statistics(self):
        """打印API使用統計 - 保持原有方法"""
        stats = self.get_usage_statistics()
        
        print("\n" + "="*50)
        print("📊 API使用統計")
        print("="*50)
        print(f"總請求次數: {stats['total_requests']}")
        
        for key_name, key_stats in stats["keys_usage"].items():
            status = "❄️ 冷卻中" if key_stats["is_cooling"] else "✅ 可用"
            print(f"{key_name}: {key_stats['usage_count']} 次 ({key_stats['usage_percentage']:.1f}%) {status}")
        
        print("="*50)
    
    def reset_statistics(self):
        """重置使用統計 - 額外功能"""
        self.key_usage_count = {key: 0 for key in self.api_keys}
        self.request_count = 0
        self.key_cooldowns.clear()
        print("🔄 已重置API使用統計")
    
    def get_current_key_info(self) -> dict:
        """獲取當前key信息 - 額外功能"""
        current_key = self.api_keys[self.current_key_index]
        return {
            "current_index": self.current_key_index + 1,
            "current_key_preview": current_key[:10] + "...",
            "current_key_usage": self.key_usage_count[current_key],
            "total_requests": self.request_count,
            "is_cooling": current_key in self.key_cooldowns
        }

# 配置你的API keys - 保持原有配置格式
GEMINI_API_KEYS = [
    " #Your API Key 1 ",
    " #Your API Key 2 "
]

def create_api_manager(model_name: str = "models/gemini-1.5-flash") -> GeminiAPIManager:
    """創建API管理器實例 - 保持原有接口"""
    return GeminiAPIManager(GEMINI_API_KEYS, model_name)

# =============================================================================
# 測試和調試功能
# =============================================================================

def test_api_rotation():
    """測試API輪換功能"""
    print("🧪 測試API輪換功能")
    print("=" * 50)
    
    try:
        api_manager = GeminiAPIManager(GEMINI_API_KEYS, "models/gemini-1.5-flash")
        
        print(f"初始狀態:")
        api_manager.print_usage_statistics()
        
        # 快速連續調用以觸發輪換
        test_prompts = [
            "說 'Hello'",
            "計算 1+1",
            "什麼是AI?", 
            "今天星期幾?",
            "簡短回答：什麼是ESG?"
        ] * 5  # 重複5次，總共25個請求
        
        print(f"\n🔄 執行 {len(test_prompts)} 次API調用...")
        
        success_count = 0
        for i, prompt in enumerate(test_prompts, 1):
            try:
                response = api_manager.invoke(prompt)
                success_count += 1
                
                if i % 5 == 0:  # 每5次顯示一次進度
                    print(f"進度: {i}/{len(test_prompts)} 完成")
                    current_info = api_manager.get_current_key_info()
                    print(f"當前使用: Key {current_info['current_index']}")
                
            except Exception as e:
                print(f"第{i}次調用失敗: {e}")
        
        print(f"\n📊 測試結果:")
        print(f"成功調用: {success_count}/{len(test_prompts)}")
        api_manager.print_usage_statistics()
        
        # 檢查是否有輪換
        used_keys = [stats['usage_count'] for stats in api_manager.get_usage_statistics()['keys_usage'].values()]
        keys_used = sum(1 for count in used_keys if count > 0)
        
        if keys_used > 1:
            print(f"✅ API輪換成功！使用了 {keys_used} 個不同的keys")
        else:
            print(f"⚠️ 未觸發輪換，只使用了 1 個key")
            
    except Exception as e:
        print(f"❌ 測試失敗: {e}")

def quick_test():
    """快速測試"""
    print("⚡ 快速測試API管理器")
    print("=" * 30)
    
    try:
        api_manager = GeminiAPIManager(GEMINI_API_KEYS, "models/gemini-1.5-flash")
        
        # 測試基本調用
        response = api_manager.invoke("說 'Hello, API Manager!'")
        print(f"✅ 基本調用成功")
        print(f"響應: {response[:50]}...")
        
        # 顯示統計
        api_manager.print_usage_statistics()
        
        # 顯示當前key信息
        current_info = api_manager.get_current_key_info()
        print(f"\n📋 當前Key信息:")
        for key, value in current_info.items():
            print(f"   {key}: {value}")
        
    except Exception as e:
        print(f"❌ 快速測試失敗: {e}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "--test":
            test_api_rotation()
        elif sys.argv[1] == "--quick":
            quick_test()
        else:
            print("用法: python api_manager.py [--test|--quick]")
    else:
        quick_test()