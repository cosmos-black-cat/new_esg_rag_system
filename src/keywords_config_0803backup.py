# =============================================================================
# 整合版關鍵字配置文件 - 支援連續和不連續關鍵字
# =============================================================================

import re
from typing import List, Dict, Tuple, Union

# 增強的關鍵字配置
ESG_KEYWORDS_CONFIG = {
    "再生塑膠材料使用": {
        "description": "再生塑膠材料的使用情況和數據",
        "type": "percentage_or_number",
        "keywords": [
            # ===== 連續關鍵字 =====
            "再生塑膠",
            "再生塑料", 
            "再生料",
            "再生pp",
            
            # ===== 不連續關鍵字組合 (元組格式) =====
            ("PP", "棧板", "回收"),      # PP棧板回收
            ("PP", "回收"),             # PP...回收
            ("PP", "再生"),             # PP再生材料  
            ("塑膠", "回收"),           # 塑膠...回收
            ("塑料", "再生"),           # 塑料再生技術
            ("再生", "材料"),           # 再生...材料
            ("回收", "塑膠"),           # 回收塑膠製品
            ("回收", "塑料"),           # 回收塑料產品
            ("PCR", "材料"),            # PCR材料使用
            ("rPET", "含量"),           # rPET含量比例
            ("回收", "產能"),           # 回收產能數據
            ("MLCC", "回收"),           # MLCC回收相關
        ]
    }
}

def smart_keyword_match(text: str, keyword: Union[str, tuple], max_distance: int = 100) -> Tuple[bool, float]:
    """
    智能關鍵字匹配函數
    
    Args:
        text: 要搜索的文本
        keyword: 關鍵字（字符串或元組）
        max_distance: 不連續關鍵字組件間的最大距離
    
    Returns:
        Tuple[bool, float]: (是否匹配, 信心分數)
    """
    text_lower = text.lower()
    
    if isinstance(keyword, str):
        # 連續關鍵字匹配
        if keyword.lower() in text_lower:
            return True, 1.0
        return False, 0.0
    
    elif isinstance(keyword, tuple):
        # 不連續關鍵字匹配
        components = [comp.lower() for comp in keyword]
        positions = []
        
        # 找到每個組件的位置
        for comp in components:
            pos = text_lower.find(comp)
            if pos == -1:
                return False, 0.0  # 如果任何組件未找到，直接返回False
            positions.append(pos)
        
        # 計算組件間的距離
        min_pos = min(positions)
        max_pos = max(positions)
        distance = max_pos - min_pos
        
        # 根據距離計算信心分數
        if distance <= 20:
            return True, 0.95  # 非常近
        elif distance <= 50:
            return True, 0.85  # 近
        elif distance <= max_distance:
            return True, 0.7   # 可接受的距離
        else:
            return True, 0.5   # 距離較遠但仍然相關
    
    return False, 0.0

def enhanced_first_stage_filter(text_content: str, keywords: list) -> Tuple[bool, List[Dict]]:
    """
    增強的第一階段篩選
    """
    matches = []
    
    for keyword in keywords:
        is_match, confidence = smart_keyword_match(text_content, keyword)
        
        if is_match:
            keyword_str = keyword if isinstance(keyword, str) else " + ".join(keyword)
            matches.append({
                'keyword': keyword_str,
                'original_keyword': keyword,
                'confidence': confidence,
                'match_type': 'continuous' if isinstance(keyword, str) else 'discontinuous'
            })
    
    return len(matches) > 0, matches

def enhanced_second_stage_filter(text_content: str, keywords: list) -> list:
    """
    增強的第二階段篩選
    """
    # 更全面的數值和百分比正則表達式
    number_patterns = [
        r'\d+(?:,\d{3})*(?:\.\d+)?(?:\s*(?:kg|KG|公斤|噸|克|g|G|公克|萬噸|千噸|萬|千|百|個|件|批|台|套))',
        r'\d+(?:,\d{3})*(?:\.\d+)?(?:\s*噸/月)',  # 特殊格式：噸/月
        r'\d+(?:,\d{3})*(?:\.\d+)?\s*(?:萬|千)?(?:噸|公斤|kg)',  # 帶萬、千的數量
    ]
    
    percentage_patterns = [
        r'\d+(?:\.\d+)?(?:\s*%|百分比|成)',
        r'\d+(?:\.\d+)?(?:\s*％)',  # 全角百分號
    ]
    
    results = []
    
    # 分割成段落（支援多種分割符）
    paragraphs = re.split(r'\n+|\r+|。{2,}', text_content)
    
    for i, paragraph in enumerate(paragraphs):
        if len(paragraph.strip()) < 10:
            continue
        
        # 使用增強匹配檢查段落
        has_keyword, matches = enhanced_first_stage_filter(paragraph, keywords)
        
        if has_keyword:
            # 檢查所有數值模式
            all_numbers = []
            for pattern in number_patterns:
                numbers = re.findall(pattern, paragraph, re.IGNORECASE)
                all_numbers.extend(numbers)
            
            # 檢查所有百分比模式
            all_percentages = []
            for pattern in percentage_patterns:
                percentages = re.findall(pattern, paragraph, re.IGNORECASE)
                all_percentages.extend(percentages)
            
            if all_numbers or all_percentages:
                for match in matches:
                    results.append({
                        'keyword': match['keyword'],
                        'original_keyword': match['original_keyword'],
                        'paragraph': paragraph.strip(),
                        'paragraph_number': i + 1,
                        'numbers': all_numbers,
                        'percentages': all_percentages,
                        'page_info': f"段落{i+1}",
                        'match_confidence': match['confidence'],
                        'match_type': match['match_type']
                    })
    
    return results

def get_all_keywords():
    """獲取所有關鍵字（包含連續和不連續）"""
    all_keywords = []
    for config in ESG_KEYWORDS_CONFIG.values():
        all_keywords.extend(config["keywords"])
    return all_keywords

# =============================================================================
# 測試和驗證函數
# =============================================================================

def test_with_real_data():
    """使用真實數據測試（基於上傳的圖片內容）"""
    
    test_text = """
    PP棧板回收
    導入使用外採回收PP粒及企業內生產之廢料為主，生產優板材之環保型塑膠板，2023年度產量為12,809噸
    （其中含有88.4%），具環保產品標章認證，較一般產品減有76%之減碳效益。
    
    纖物回收
    已具有綠PET原布、成布及聚羥乙烯回收重產技術，並已達成日產1,000噸之前置理及檢視產能。並已
    開發高端服務BHET化學間收技術，並建設在地廢工廠，做為未來放大量產之基礎。
    
    MLCC(積層陶瓷電容)用銀膜運回收
    全戶使用液原溶收型蒙集再生重，消戶已引進成分後已用效果，中水回收能力UP置實改善場智產品，回
    收產能每600噸/月，可回收國內MLCC及光學等學客使用後之廢料關。
    """
    
    keywords = get_all_keywords()
    
    print("🧪 真實數據測試結果")
    print("=" * 60)
    
    # 第一階段篩選
    passed, matches = enhanced_first_stage_filter(test_text, keywords)
    print(f"第一階段篩選: {'✅ 通過' if passed else '❌ 未通過'}")
    print(f"找到匹配: {len(matches)} 個\n")
    
    for match in matches:
        print(f"🎯 關鍵字: {match['keyword']}")
        print(f"   信心分數: {match['confidence']:.2f}")
        print(f"   匹配類型: {match['match_type']}")
        print()
    
    # 第二階段篩選
    results = enhanced_second_stage_filter(test_text, keywords)
    print(f"第二階段篩選: 找到 {len(results)} 個含數值的結果")
    print("=" * 60)
    
    for i, result in enumerate(results, 1):
        print(f"\n📊 結果 {i}:")
        print(f"關鍵字: {result['keyword']}")
        print(f"匹配類型: {result['match_type']}")
        print(f"數值: {result['numbers']}")
        print(f"百分比: {result['percentages']}")
        print(f"信心分數: {result['match_confidence']:.2f}")
        print(f"段落內容: {result['paragraph'][:150]}...")
        print("-" * 40)

def validate_keywords_config():
    """驗證關鍵字配置"""
    print("🔍 驗證關鍵字配置...")
    
    total_keywords = 0
    continuous_count = 0
    discontinuous_count = 0
    
    for indicator, config in ESG_KEYWORDS_CONFIG.items():
        keywords = config["keywords"]
        total_keywords += len(keywords)
        
        for keyword in keywords:
            if isinstance(keyword, str):
                continuous_count += 1
            elif isinstance(keyword, tuple):
                discontinuous_count += 1
    
    print(f"✅ 總關鍵字數: {total_keywords}")
    print(f"   連續關鍵字: {continuous_count}")
    print(f"   不連續關鍵字: {discontinuous_count}")
    print(f"   不連續關鍵字比例: {discontinuous_count/total_keywords*100:.1f}%")

if __name__ == "__main__":
    print("🚀 整合版關鍵字匹配系統測試")
    print("=" * 60)
    
    validate_keywords_config()
    print()
    test_with_real_data()