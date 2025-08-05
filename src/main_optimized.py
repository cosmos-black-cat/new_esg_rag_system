#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ESG資料提取系統 - 優化版 v3.0
專注於再生塑膠相關數據的精確提取
"""

import os
import sys
import shutil
import argparse
from pathlib import Path
from datetime import datetime

# 添加當前目錄到路徑
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

try:
    from config import *
    CONFIG_LOADED = True
except ImportError as e:
    print(f"⚠️ 配置載入失敗: {e}")
    CONFIG_LOADED = False

def check_pdf_files():
    """檢查PDF文件"""
    if not CONFIG_LOADED:
        return False, None
    
    try:
        data_dir = Path(DATA_PATH)
        pdf_files = list(data_dir.glob("*.pdf"))
        
        if not pdf_files:
            print(f"❌ 在 {DATA_PATH} 目錄中找不到PDF文件")
            return False, None
        
        pdf_file = pdf_files[0]
        print(f"✅ 找到PDF文件: {pdf_file.name}")
        return True, pdf_file
        
    except Exception as e:
        print(f"❌ 檢查PDF文件失敗: {e}")
        return False, None

def check_vector_database():
    """檢查向量資料庫"""
    if not CONFIG_LOADED:
        return False
    
    try:
        if os.path.exists(VECTOR_DB_PATH):
            required_files = ['index.faiss', 'index.pkl']
            missing_files = []
            
            for file in required_files:
                file_path = os.path.join(VECTOR_DB_PATH, file)
                if not os.path.exists(file_path):
                    missing_files.append(file)
            
            if missing_files:
                print(f"⚠️ 向量資料庫不完整，缺少: {missing_files}")
                return False
            
            print(f"✅ 向量資料庫存在: {VECTOR_DB_PATH}")
            return True
        else:
            print(f"❌ 向量資料庫不存在: {VECTOR_DB_PATH}")
            return False
            
    except Exception as e:
        print(f"❌ 檢查向量資料庫失敗: {e}")
        return False

def run_preprocessing(force=False):
    """執行PDF預處理"""
    if not CONFIG_LOADED:
        print("❌ 配置未載入")
        return False
    
    try:
        from preprocess import preprocess_documents
        
        if force and os.path.exists(VECTOR_DB_PATH):
            print("🗑️ 刪除舊的向量資料庫...")
            shutil.rmtree(VECTOR_DB_PATH)
        
        if not force and check_vector_database():
            print("ℹ️ 向量資料庫已存在，跳過預處理")
            return True
        
        has_pdf, pdf_file = check_pdf_files()
        if not has_pdf:
            return False
        
        print("🔄 開始PDF預處理...")
        preprocess_documents(str(pdf_file))
        
        if check_vector_database():
            print("✅ PDF預處理完成")
            return True
        else:
            print("❌ 預處理完成但向量資料庫驗證失敗")
            return False
            
    except Exception as e:
        print(f"❌ 預處理失敗: {e}")
        return False

def run_extraction():
    """執行資料提取"""
    try:
        from esg_extractor_optimized import ESGExtractorOptimized
        
        print("🚀 初始化ESG資料提取器...")
        extractor = ESGExtractorOptimized()
        
        print("🔍 開始資料提取...")
        extractions, summary, excel_path = extractor.run_complete_extraction()
        
        return extractions, summary, excel_path
        
    except Exception as e:
        print(f"❌ 資料提取失敗: {e}")
        return None

def main():
    """主函數"""
    print("🏢 ESG資料提取系統 v3.0 (優化版)")
    print("專注於再生塑膠數據的精確提取")
    print("=" * 50)
    
    parser = argparse.ArgumentParser(description="ESG資料提取系統 - 優化版")
    parser.add_argument("--auto", action="store_true", help="自動執行完整流程")
    parser.add_argument("--preprocess", action="store_true", help="預處理PDF文件")
    parser.add_argument("--extract", action="store_true", help="執行資料提取")
    
    args = parser.parse_args()
    
    if not CONFIG_LOADED:
        print("❌ 配置載入失敗")
        return
    
    if args.auto:
        # 自動執行完整流程
        print("🚀 自動執行模式")
        
        if not check_vector_database():
            print("執行預處理...")
            if not run_preprocessing():
                return
        
        print("執行資料提取...")
        result = run_extraction()
        if result:
            extractions, summary, excel_path = result
            print(f"✅ 完成！結果已保存至: {excel_path}")
        
    elif args.preprocess:
        if run_preprocessing(force=True):
            print("✅ 預處理完成")
        
    elif args.extract:
        result = run_extraction()
        if result:
            extractions, summary, excel_path = result
            print(f"✅ 提取完成: {excel_path}")
    
    else:
        # 互動模式
        while True:
            print("\n" + "🔷" * 15)
            print("🏢 ESG資料提取系統 v3.0")
            print("🔷" * 15)
            print("1. 📊 執行完整資料提取")
            print("2. 🔄 重新預處理PDF")
            print("3. 🚪 退出系統")
            
            choice = input("\n請選擇功能 (1-3): ").strip()
            
            if choice == "1":
                if not check_vector_database():
                    print("🔄 向量資料庫不存在，需要先預處理PDF")
                    if run_preprocessing():
                        print("✅ 預處理完成，繼續提取...")
                    else:
                        print("❌ 預處理失敗")
                        continue
                
                result = run_extraction()
                if result:
                    extractions, summary, excel_path = result
                    print(f"\n🎉 提取完成！")
                    print(f"📁 結果已保存: {excel_path}")
                    print(f"📊 共提取 {len(extractions)} 個相關數據")
                
            elif choice == "2":
                confirm = input("這將刪除現有向量資料庫，確定繼續？(y/n): ").strip().lower()
                if confirm == 'y':
                    if run_preprocessing(force=True):
                        print("✅ 預處理完成")
                
            elif choice == "3":
                print("👋 感謝使用ESG資料提取系統！")
                break
                
            else:
                print("❌ 無效選擇，請輸入1-3之間的數字")

if __name__ == "__main__":
    main()