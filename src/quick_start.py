#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ESG報告書提取器 - 快速開始腳本 v1.0
一鍵設置和運行ESG數據提取
"""

import os
import sys
from pathlib import Path

def check_python_version():
    """檢查Python版本"""
    if sys.version_info < (3, 8):
        print("❌ 錯誤: 需要Python 3.8或更高版本")
        print(f"   當前版本: {sys.version}")
        return False
    print(f"✅ Python版本: {sys.version.split()[0]}")
    return True

def check_api_key():
    """檢查API Key設置"""
    env_file = Path('.env')
    if not env_file.exists():
        print("❌ 找不到.env文件")
        create_env_file()
        return False
    
    # 讀取.env文件檢查API Key
    with open(env_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    if 'GOOGLE_API_KEY=your_api_key_here' in content or 'GOOGLE_API_KEY=' in content and 'AIza' not in content:
        print("⚠️ API Key尚未設置")
        print("請編輯.env文件，設置您的Google API Key")
        print("獲取API Key: https://makersuite.google.com/app/apikey")
        return False
    
    print("✅ API Key已設置")
    return True

def create_env_file():
    """創建.env文件模板"""
    env_template = """# =============================================================================
# ESG報告書提取器環境配置文件
# =============================================================================

# Google API Key (必填) - 請替換為您的實際API Key
# 獲取方式：https://makersuite.google.com/app/apikey
GOOGLE_API_KEY=your_api_key_here

# 其他配置（通常不需要修改）
GEMINI_MODEL=gemini-1.5-flash
EMBEDDING_MODEL=BAAI/bge-large-zh-v1.5
VECTOR_DB_PATH=./vector_db/esg_db
DATA_PATH=./data
RESULTS_PATH=./results
CHUNK_SIZE=800
CHUNK_OVERLAP=150
SEARCH_K=10
CONFIDENCE_THRESHOLD=0.6
MAX_DOCS_PER_RUN=300
ENABLE_LLM_ENHANCEMENT=true
LLM_MAX_RETRIES=3
"""
    
    with open('.env', 'w', encoding='utf-8') as f:
        f.write(env_template)
    
    print("✅ 已創建.env文件模板")
    print("請編輯.env文件，設置您的Google API Key")

def check_directories():
    """檢查和創建必要目錄"""
    directories = ['data', 'results', 'vector_db']
    
    for directory in directories:
        dir_path = Path(directory)
        if not dir_path.exists():
            dir_path.mkdir(parents=True, exist_ok=True)
            print(f"✅ 創建目錄: {directory}/")
        else:
            print(f"✅ 目錄存在: {directory}/")

def check_pdf_files():
    """檢查PDF文件"""
    data_dir = Path('data')
    pdf_files = list(data_dir.glob('*.pdf'))
    
    if not pdf_files:
        print("⚠️ data目錄中沒有PDF文件")
        print("請將ESG報告PDF文件放入data目錄中")
        return False
    
    print(f"✅ 找到 {len(pdf_files)} 個PDF文件:")
    for pdf_file in pdf_files:
        print(f"   - {pdf_file.name}")
    
    return True

def install_dependencies():
    """安裝依賴包"""
    print("正在檢查依賴包...")
    
    try:
        import langchain
        import pandas
        import numpy
        import faiss
        print("✅ 主要依賴包已安裝")
        return True
    except ImportError as e:
        print(f"⚠️ 缺少依賴包: {e}")
        print("正在嘗試安裝依賴包...")
        
        try:
            import subprocess
            result = subprocess.run([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'], 
                                  capture_output=True, text=True)
            
            if result.returncode == 0:
                print("✅ 依賴包安裝成功")
                return True
            else:
                print(f"❌ 依賴包安裝失敗: {result.stderr}")
                return False
                
        except Exception as e:
            print(f"❌ 自動安裝失敗: {e}")
            print("請手動運行: pip install -r requirements.txt")
            return False

def run_system_check():
    """運行系統檢查"""
    print("📊 ESG報告書提取器 - 系統檢查")
    print("=" * 50)
    
    checks = [
        ("Python版本", check_python_version),
        ("依賴包", install_dependencies),
        ("目錄結構", check_directories),
        ("API Key", check_api_key),
        ("PDF文件", check_pdf_files),
    ]
    
    all_passed = True
    
    for check_name, check_func in checks:
        print(f"\n🔍 檢查{check_name}...")
        if not check_func():
            all_passed = False
    
    print("\n" + "=" * 50)
    
    if all_passed:
        print("🎉 系統檢查通過！可以開始使用ESG報告書提取器")
        print("\n下一步:")
        print("1. 運行: python main.py")
        print("2. 選擇功能1執行ESG數據提取")
        print("3. 選擇功能3執行結果彙整")
    else:
        print("⚠️ 系統檢查未完全通過，請解決上述問題後重新運行")
        print("\n常見解決方案:")
        print("1. 設置API Key: 編輯.env文件")
        print("2. 安裝依賴: pip install -r requirements.txt")
        print("3. 添加PDF: 將ESG報告放入data目錄")
    
    return all_passed

def show_usage_guide():
    """顯示使用指南"""
    guide = """
📚 ESG報告書提取器使用指南

🚀 快速開始流程：
1. 準備PDF文件
   - 將ESG報告PDF放入data目錄
   - 支援中文ESG報告書

2. 設置API Key
   - 編輯.env文件
   - 設置GOOGLE_API_KEY=your_actual_key

3. 執行提取
   - 運行: python main.py
   - 選擇功能1: 執行ESG數據提取

4. 查看結果
   - 結果保存在results目錄
   - Excel格式，包含提取的數值和統計

🎯 主要功能：
• 自動識別再生塑膠相關數據
• 支援批量處理多份報告
• 智能關鍵字匹配和數值提取
• 多公司多年度結果彙整

📊 輸出內容：
• 再生塑膠使用量（億支、萬噸等）
• 回收產能和減碳效益
• 循環經濟相關數據
• 環保效益統計

⚙️ 配置調整：
• 提高精度: 增加CONFIDENCE_THRESHOLD到0.7
• 提高覆蓋: 降低CONFIDENCE_THRESHOLD到0.5
• 控制成本: 調整MAX_DOCS_PER_RUN

💡 使用技巧：
• 確保PDF為可搜索格式（非掃描版）
• 建議單次處理3-5份報告
• 提取結果需要人工檢查確認

📞 需要幫助？
• 檢查README.md文件
• 確認PDF格式和內容質量
• 調整配置參數重新嘗試
"""
    print(guide)

def main():
    """主函數"""
    import argparse
    
    parser = argparse.ArgumentParser(description="ESG報告書提取器快速開始工具")
    parser.add_argument("--check", action="store_true", help="執行系統檢查")
    parser.add_argument("--guide", action="store_true", help="顯示使用指南")
    parser.add_argument("--setup", action="store_true", help="初始化設置")
    
    args = parser.parse_args()
    
    if args.guide:
        show_usage_guide()
    elif args.setup:
        print("🔧 初始化ESG報告書提取器...")
        check_directories()
        create_env_file()
        print("\n✅ 初始化完成！")
        print("下一步: 編輯.env文件設置API Key，然後運行 python quick_start.py --check")
    elif args.check or len(sys.argv) == 1:
        run_system_check()
    else:
        parser.print_help()

if __name__ == "__main__":
    main()