#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ESG資料彙整模組 v1.0
將多個Excel檔案彙整成一個總覽表
"""

import os
import re
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime
import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from difflib import SequenceMatcher

class CompanyNameStandardizer:
    """公司名稱標準化處理器"""
    
    def __init__(self):
        # 常見的公司後綴詞（按優先級排序）
        self.company_suffixes = [
            "股份有限公司", "有限公司", "股份", "公司", 
            "工業股份有限公司", "工業有限公司", "工業股份", "工業公司", "工業",
            "化學工業股份有限公司", "化學工業有限公司", "化學工業股份", "化學工業", "化學",
            "塑膠工業股份有限公司", "塑膠工業有限公司", "塑膠工業股份", "塑膠工業", "塑膠",
            "電子股份有限公司", "電子有限公司", "電子股份", "電子公司", "電子",
            "科技股份有限公司", "科技有限公司", "科技股份", "科技公司", "科技"
        ]
        
        # 標準化映射表（手動定義的特殊案例）
        self.manual_mappings = {
            # 可以在這裡添加特殊的手動映射
            # "原始名稱": "標準名稱"
        }
        
        # 公司名稱緩存
        self.standardization_cache = {}
    
    def extract_core_name(self, company_name: str) -> str:
        """提取公司核心名稱（去除後綴）"""
        if not company_name or company_name == "未知公司":
            return company_name
        
        # 清理空白
        name = company_name.strip()
        
        # 檢查手動映射
        if name in self.manual_mappings:
            return self.manual_mappings[name]
        
        # 按優先級去除後綴
        for suffix in self.company_suffixes:
            if name.endswith(suffix):
                core_name = name[:-len(suffix)].strip()
                if core_name:  # 確保去除後綴後還有內容
                    return core_name
        
        return name
    
    def calculate_similarity(self, name1: str, name2: str) -> float:
        """計算兩個公司名稱的相似度"""
        if not name1 or not name2:
            return 0.0
        
        # 提取核心名稱
        core1 = self.extract_core_name(name1)
        core2 = self.extract_core_name(name2)
        
        # 完全匹配
        if core1 == core2:
            return 1.0
        
        # 一個是另一個的子字符串
        if core1 in core2 or core2 in core1:
            return 0.9
        
        # 計算字符串相似度
        similarity = SequenceMatcher(None, core1, core2).ratio()
        
        return similarity
    
    def find_best_match(self, target_name: str, existing_names: List[str], threshold: float = 0.8) -> Tuple[str, float]:
        """找到最佳匹配的現有公司名稱"""
        if not existing_names:
            return target_name, 1.0
        
        best_match = target_name
        best_similarity = 0.0
        
        for existing_name in existing_names:
            similarity = self.calculate_similarity(target_name, existing_name)
            if similarity > best_similarity and similarity >= threshold:
                best_similarity = similarity
                best_match = existing_name
        
        return best_match, best_similarity
    
    def choose_standard_name(self, similar_names: List[str]) -> str:
        """從相似的名稱中選擇標準名稱（最完整的）"""
        if not similar_names:
            return "未知公司"
        
        if len(similar_names) == 1:
            return similar_names[0]
        
        # 優先選擇最長的名稱（通常最完整）
        longest_name = max(similar_names, key=len)
        
        # 檢查是否有包含"股份有限公司"的完整名稱
        for name in similar_names:
            if "股份有限公司" in name:
                return name
        
        # 檢查是否有包含"有限公司"的名稱
        for name in similar_names:
            if "有限公司" in name:
                return name
        
        return longest_name

class ESGDataConsolidator:
    """
    ESG資料彙整器
    
    功能：
    - 自動掃描results目錄下的所有Excel檔案
    - 排除檔名包含'無提取'的檔案
    - 智能識別同一公司的不同命名方式
    - 按年度和公司分組生成彙整報告
    - 包含統計摘要和美化格式
    """
    
    def __init__(self, results_path: str):
        self.results_path = Path(results_path)
        self.name_standardizer = CompanyNameStandardizer()
        self.company_mapping = {}  # 原始名稱 -> 標準名稱的映射
        
        print(f"📊 初始化ESG資料彙整器")
        print(f"📁 結果目錄: {self.results_path}")
        print(f"⚠️ 注意：檔名包含'無提取'的檔案將被自動排除")
        print(f"🏢 智能識別：同一公司的不同命名將自動統一")
    
    def _standardize_company_names(self, all_data: List[Dict]) -> List[Dict]:
        """標準化所有公司名稱，將相似的公司名稱統一"""
        if not all_data:
            return all_data
        
        print("🏢 開始標準化公司名稱...")
        
        # 收集所有唯一的公司名稱
        unique_companies = list(set(item['company_name'] for item in all_data if item['company_name'] != "未知公司"))
        
        if not unique_companies:
            return all_data
        
        print(f"   發現 {len(unique_companies)} 個不同的公司名稱")
        
        # 建立標準化映射
        processed_companies = []
        
        for company in unique_companies:
            if company in self.company_mapping:
                continue  # 已處理過
            
            # 尋找相似的公司名稱
            similar_companies = [company]
            
            for other_company in unique_companies:
                if other_company != company and other_company not in processed_companies:
                    similarity = self.name_standardizer.calculate_similarity(company, other_company)
                    if similarity >= 0.8:  # 相似度閾值
                        similar_companies.append(other_company)
            
            # 選擇標準名稱
            standard_name = self.name_standardizer.choose_standard_name(similar_companies)
            
            # 建立映射
            for similar_company in similar_companies:
                self.company_mapping[similar_company] = standard_name
                processed_companies.append(similar_company)
                
                # 如果有統一，顯示信息
                if similar_company != standard_name:
                    print(f"   🔗 {similar_company} → {standard_name}")
        
        # 應用標準化
        standardized_data = []
        for item in all_data:
            new_item = item.copy()
            original_name = item['company_name']
            if original_name in self.company_mapping:
                new_item['company_name'] = self.company_mapping[original_name]
                # 保留原始名稱用於追踪
                new_item['original_company_name'] = original_name
            standardized_data.append(new_item)
        
        # 統計標準化結果
        final_companies = set(item['company_name'] for item in standardized_data if item['company_name'] != "未知公司")
        print(f"   ✅ 標準化完成: {len(unique_companies)} → {len(final_companies)} 個公司")
        
        return standardized_data
        """彙整所有結果到一個Excel檔案，排除'無提取'檔案"""
        print("\n🚀 開始彙整所有ESG提取結果...")
        print("=" * 60)
        
        # 1. 掃描並分析所有Excel檔案（排除'無提取'）
        excel_files = self._scan_excel_files()
        if not excel_files:
            print("❌ 未找到任何有效的Excel結果檔案")
            print("💡 提示：包含'無提取'的檔案已自動排除")
            return None
        
        print(f"📄 找到 {len(excel_files)} 個有效Excel結果檔案（已排除'無提取'檔案）")
        
        # 2. 解析檔案信息
        parsed_files = self._parse_file_info(excel_files)
        print(f"✅ 成功解析 {len(parsed_files)} 個檔案")
        
        # 3. 載入所有資料
        all_data = self._load_all_data(parsed_files)
        print(f"📚 載入完成，共 {len(all_data)} 筆資料")
        
        # 4. 生成彙整報告
        output_path = self._create_consolidated_excel(all_data, parsed_files)
        
        print(f"✅ 彙整完成！")
        print(f"📊 輸出檔案: {output_path}")
        
        return output_path
    
    def _scan_excel_files(self) -> List[Path]:
        """掃描所有Excel檔案，排除包含'無提取'的檔案"""
        excel_files = []
        excluded_files = []
        
        # 查找所有Excel檔案（包括不同版本）
        patterns = [
            "ESG提取結果_*.xlsx",
            "*平衡版*.xlsx", 
            "*高精度*.xlsx"
        ]
        
        for pattern in patterns:
            files = list(self.results_path.glob(pattern))
            for file in files:
                # 檢查檔名是否包含"無提取"
                if "無提取" in file.name:
                    excluded_files.append(file)
                    print(f"   ⊗ 排除檔案: {file.name} (包含'無提取')")
                else:
                    excel_files.append(file)
        
        # 顯示排除統計
        if excluded_files:
            print(f"📋 排除了 {len(excluded_files)} 個'無提取'檔案")
        
        # 去重並排序
        unique_files = list(set(excel_files))
        unique_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        
        return unique_files
    
    def _parse_file_info(self, excel_files: List[Path]) -> List[Dict]:
        """解析檔案信息"""
        parsed_files = []
        
        for file_path in excel_files:
            try:
                # 從檔名解析基本信息
                filename = file_path.stem
                
                # 解析年度
                year_match = re.search(r'(202[0-9])', filename)
                year = year_match.group(1) if year_match else "未知年度"
                
                # 解析版本
                if "平衡版" in filename:
                    version = "平衡版"
                elif "高精度" in filename:
                    version = "高精度版"
                else:
                    version = "標準版"
                
                # 嘗試從Excel內容中讀取更詳細的信息
                company_name, report_year = self._extract_company_info_from_excel(file_path)
                
                parsed_files.append({
                    'file_path': file_path,
                    'filename': filename,
                    'company_name': company_name,
                    'report_year': report_year or year,
                    'version': version,
                    'file_time': datetime.fromtimestamp(file_path.stat().st_mtime)
                })
                
                print(f"   ✓ {company_name} - {report_year or year} ({version})")
                
            except Exception as e:
                print(f"   ⚠️ 解析失敗 {file_path.name}: {e}")
                continue
        
        return parsed_files
    
    def _extract_company_info_from_excel(self, file_path: Path) -> Tuple[str, str]:
        """從Excel檔案中提取公司名稱和年度"""
        try:
            # 讀取第一個工作表的前幾行
            df = pd.read_excel(file_path, nrows=5)
            
            company_name = "未知公司"
            report_year = ""
            
            # 查找公司信息
            for col in df.columns:
                for idx, row in df.iterrows():
                    cell_value = str(row[col])
                    
                    # 提取公司名稱
                    if "公司:" in cell_value:
                        company_match = re.search(r'公司:\s*(.+)', cell_value)
                        if company_match:
                            company_name = company_match.group(1).strip()
                    
                    # 提取報告年度
                    if "報告年度:" in cell_value:
                        year_match = re.search(r'報告年度:\s*(202[0-9])', cell_value)
                        if year_match:
                            report_year = year_match.group(1)
            
            return company_name, report_year
            
        except Exception as e:
            print(f"   ⚠️ 讀取Excel失敗: {e}")
            return "未知公司", ""
    
    def _load_all_data(self, parsed_files: List[Dict]) -> List[Dict]:
        """載入所有資料"""
        all_data = []
        
        for file_info in parsed_files:
            try:
                file_path = file_info['file_path']
                
                # 嘗試讀取不同的工作表名稱
                sheet_names_to_try = [
                    '平衡版提取結果', '提取結果', 'Sheet1', 0
                ]
                
                df = None
                for sheet_name in sheet_names_to_try:
                    try:
                        df = pd.read_excel(file_path, sheet_name=sheet_name)
                        break
                    except:
                        continue
                
                if df is None:
                    print(f"   ⚠️ 無法讀取 {file_path.name}")
                    continue
                
                # 跳過標題行（前2行通常是公司信息和空行）
                data_start_row = 0
                for idx, row in df.iterrows():
                    if idx < 5:  # 檢查前5行
                        first_cell = str(row.iloc[0]) if len(row) > 0 else ""
                        if "公司:" not in first_cell and first_cell.strip() and first_cell != "nan":
                            data_start_row = idx
                            break
                
                if data_start_row < len(df):
                    df = df.iloc[data_start_row:]
                
                # 為每一行添加檔案信息
                for idx, row in df.iterrows():
                    if len(row) > 0 and str(row.iloc[0]).strip() and str(row.iloc[0]) != "nan":
                        data_row = {
                            'company_name': file_info['company_name'],
                            'report_year': file_info['report_year'],
                            'version': file_info['version'],
                            'file_name': file_info['filename'],
                            'source_file': file_path.name
                        }
                        
                        # 添加原始數據列
                        for col_idx, col_name in enumerate(df.columns):
                            if col_idx < len(row):
                                data_row[col_name] = row.iloc[col_idx]
                        
                        all_data.append(data_row)
                
                print(f"   ✓ 載入 {file_info['company_name']} - {file_info['report_year']} ({len(df)} 筆)")
                
            except Exception as e:
                print(f"   ❌ 載入失敗 {file_info['file_path'].name}: {e}")
                continue
        
        return all_data
    
    def _create_consolidated_excel(self, all_data: List[Dict], parsed_files: List[Dict]) -> str:
        """創建彙整的Excel檔案"""
        output_filename = f"ESG彙整報告.xlsx"
        output_path = self.results_path / output_filename
        
        print(f"📊 生成彙整Excel: {output_filename}")
        
        # 轉換為DataFrame
        df_all = pd.DataFrame(all_data)
        
        if df_all.empty:
            print("❌ 沒有有效數據可彙整")
            return None
        
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            # 按年度分組的工作表
            years = sorted(set(item['report_year'] for item in all_data if item['report_year'] != "未知年度"), reverse=True)
            
            for year in years:
                year_data = [item for item in all_data if item['report_year'] == year]
                if year_data:
                    year_df = pd.DataFrame(year_data)
                    sheet_name = f"{year}年總覽"
                    year_df.to_excel(writer, sheet_name=sheet_name, index=False)
                    print(f"   ✓ 創建工作表: {sheet_name} ({len(year_data)} 筆)")
            
            # 按公司分組的工作表
            companies = sorted(set(item['company_name'] for item in all_data if item['company_name'] != "未知公司"))
            
            for company in companies:
                company_data = [item for item in all_data if item['company_name'] == company]
                if company_data:
                    company_df = pd.DataFrame(company_data)
                    # 按年度排序
                    company_df = company_df.sort_values('report_year', ascending=False)
                    
                    # 清理公司名稱作為工作表名稱
                    safe_company_name = re.sub(r'[^\w\s-]', '', company)[:25]  # Excel工作表名稱限制
                    sheet_name = f"{safe_company_name}總覽"
                    
                    company_df.to_excel(writer, sheet_name=sheet_name, index=False)
                    print(f"   ✓ 創建工作表: {sheet_name} ({len(company_data)} 筆)")
            
            # 總覽摘要工作表
            summary_data = self._create_summary_data(parsed_files, all_data)
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_excel(writer, sheet_name='彙整摘要', index=False)
            print(f"   ✓ 創建工作表: 彙整摘要")
        
        # 美化Excel格式
        self._format_excel(output_path)
        
        return str(output_path)
    
    def _create_summary_data(self, parsed_files: List[Dict], all_data: List[Dict]) -> List[Dict]:
        """創建摘要數據，包含公司名稱標準化信息"""
        summary_data = []
        
        # 總體統計
        total_files = len(parsed_files)
        total_records = len(all_data)
        companies = set(item['company_name'] for item in all_data if item['company_name'] != "未知公司")
        years = set(item['report_year'] for item in all_data if item['report_year'] != "未知年度")
        
        # 公司名稱標準化統計
        original_companies = len(self.company_mapping) if self.company_mapping else 0
        standardized_companies = len(companies)
        mappings_count = len([k for k, v in self.company_mapping.items() if k != v]) if self.company_mapping else 0
        
        # 按公司統計
        company_stats = {}
        for item in all_data:
            company = item['company_name']
            if company != "未知公司":
                if company not in company_stats:
                    company_stats[company] = {'years': set(), 'records': 0, 'original_names': set()}
                company_stats[company]['years'].add(item['report_year'])
                company_stats[company]['records'] += 1
                # 記錄原始名稱
                if 'original_company_name' in item:
                    company_stats[company]['original_names'].add(item['original_company_name'])
                else:
                    company_stats[company]['original_names'].add(company)
        
        # 按年度統計
        year_stats = {}
        for item in all_data:
            year = item['report_year']
            if year != "未知年度":
                if year not in year_stats:
                    year_stats[year] = {'companies': set(), 'records': 0}
                year_stats[year]['companies'].add(item['company_name'])
                year_stats[year]['records'] += 1
        
        # 生成摘要
        summary_data.append({
            '統計項目': '彙整總覽',
            '數值': f'檔案數: {total_files}, 總筆數: {total_records}',
            '詳細說明': f'涵蓋 {standardized_companies} 家公司, {len(years)} 個年度',
            '彙整時間': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        })
        
        # 公司名稱標準化統計
        if mappings_count > 0:
            summary_data.append({
                '統計項目': '名稱標準化',
                '數值': f'{mappings_count} 個名稱變體已統一',
                '詳細說明': f'原始: {original_companies} → 標準化: {standardized_companies}',
                '彙整時間': ''
            })
        
        summary_data.append({'統計項目': '', '數值': '', '詳細說明': '', '彙整時間': ''})
        
        # 公司統計（顯示標準化信息）
        summary_data.append({
            '統計項目': '公司統計', '數值': '年度範圍', '詳細說明': '提取筆數', '彙整時間': '原始名稱'
        })
        
        for company, stats in company_stats.items():
            years_range = f"{min(stats['years'])}-{max(stats['years'])}" if len(stats['years']) > 1 else list(stats['years'])[0]
            original_names = ', '.join(stats['original_names']) if len(stats['original_names']) > 1 else ''
            
            summary_data.append({
                '統計項目': company,
                '數值': years_range,
                '詳細說明': f"{stats['records']} 筆",
                '彙整時間': original_names[:100] + ('...' if len(original_names) > 100 else '') if original_names else f"{len(stats['years'])} 個年度"
            })
        
        summary_data.append({'統計項目': '', '數值': '', '詳細說明': '', '彙整時間': ''})
        
        # 年度統計
        summary_data.append({
            '統計項目': '年度統計', '數值': '公司數量', '詳細說明': '提取筆數', '彙整時間': ''
        })
        
        for year in sorted(year_stats.keys(), reverse=True):
            stats = year_stats[year]
            summary_data.append({
                '統計項目': f"{year}年",
                '數值': f"{len(stats['companies'])} 家公司",
                '詳細說明': f"{stats['records']} 筆",
                '彙整時間': ', '.join(sorted(stats['companies']))[:50] + ('...' if len(', '.join(stats['companies'])) > 50 else '')
            })
        
        # 如果有名稱映射，添加映射詳情
        if self.company_mapping and mappings_count > 0:
            summary_data.append({'統計項目': '', '數值': '', '詳細說明': '', '彙整時間': ''})
            summary_data.append({
                '統計項目': '名稱映射詳情', '數值': '原始名稱', '詳細說明': '標準名稱', '彙整時間': ''
            })
            
            for original, standard in self.company_mapping.items():
                if original != standard:  # 只顯示有變更的
                    summary_data.append({
                        '統計項目': '',
                        '數值': original,
                        '詳細說明': standard,
                        '彙整時間': ''
                    })
        
        return summary_data
    
    def _format_excel(self, output_path: str):
        """美化Excel格式"""
        try:
            workbook = openpyxl.load_workbook(output_path)
            
            # 定義樣式
            header_font = Font(bold=True, color="FFFFFF")
            header_fill = PatternFill(start_color="4472C4", end_color="4472C4", fill_type="solid")
            border = Border(
                left=Side(style='thin'),
                right=Side(style='thin'),
                top=Side(style='thin'),
                bottom=Side(style='thin')
            )
            
            for sheet_name in workbook.sheetnames:
                worksheet = workbook[sheet_name]
                
                # 設置標題行格式
                if worksheet.max_row > 0:
                    for cell in worksheet[1]:
                        cell.font = header_font
                        cell.fill = header_fill
                        cell.alignment = Alignment(horizontal='center')
                        cell.border = border
                    
                    # 自動調整列寬
                    for column in worksheet.columns:
                        max_length = 0
                        column_letter = column[0].column_letter
                        
                        for cell in column:
                            try:
                                if len(str(cell.value)) > max_length:
                                    max_length = len(str(cell.value))
                            except:
                                pass
                        
                        adjusted_width = min(max_length + 2, 50)
                        worksheet.column_dimensions[column_letter].width = adjusted_width
            
            workbook.save(output_path)
            print(f"✨ Excel格式美化完成")
            
        except Exception as e:
            print(f"⚠️ Excel格式化失敗: {e}")

def consolidate_esg_results(results_path: str) -> str:
    """彙整ESG結果的主函數"""
    consolidator = ESGDataConsolidator(results_path)
    return consolidator.consolidate_all_results()

# =============================================================================
# 測試功能
# =============================================================================

def test_consolidation():
    """測試彙整功能"""
    print("🧪 測試ESG資料彙整功能")
    
    # 假設有一些測試檔案
    test_results_path = "./test_results"
    
    try:
        consolidator = ESGDataConsolidator(test_results_path)
        result_path = consolidator.consolidate_all_results()
        
        if result_path:
            print(f"✅ 測試成功: {result_path}")
        else:
            print("❌ 測試失敗")
            
    except Exception as e:
        print(f"❌ 測試錯誤: {e}")

def test_company_name_standardization():
    """測試公司名稱標準化功能"""
    print("🧪 測試公司名稱標準化功能")
    print("=" * 50)
    
    standardizer = CompanyNameStandardizer()
    
    # 測試案例
    test_companies = [
        ["三芳", "三芳化學", "三芳化學工業", "三芳化學工業股份有限公司"],
        ["台灣塑膠工業", "台灣塑膠工業股份有限公司", "台灣塑膠"],
        ["南亞塑膠工業", "南亞塑膠", "南亞"],
        ["台積電", "台灣積體電路製造", "台灣積體電路製造股份有限公司"],
        ["中華電信", "中華電信股份有限公司"],
    ]
    
    for i, company_group in enumerate(test_companies, 1):
        print(f"\n測試組 {i}:")
        print(f"原始名稱: {company_group}")
        
        # 計算相似度矩陣
        print("相似度分析:")
        for j, name1 in enumerate(company_group):
            for k, name2 in enumerate(company_group):
                if j < k:  # 只計算上三角
                    similarity = standardizer.calculate_similarity(name1, name2)
                    print(f"  {name1} ↔ {name2}: {similarity:.2f}")
        
        # 選擇標準名稱
        standard_name = standardizer.choose_standard_name(company_group)
        print(f"標準名稱: {standard_name}")
        
        # 顯示映射
        for name in company_group:
            if name != standard_name:
                print(f"  🔗 {name} → {standard_name}")

def test_consolidation():
    """測試彙整功能"""
    print("🧪 測試ESG資料彙整功能")
    
    # 假設有一些測試檔案
    test_results_path = "./test_results"
    
    try:
        consolidator = ESGDataConsolidator(test_results_path)
        result_path = consolidator.consolidate_all_results()
        
        if result_path:
            print(f"✅ 測試成功: {result_path}")
        else:
            print("❌ 測試失敗")
            
    except Exception as e:
        print(f"❌ 測試錯誤: {e}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "--test-names":
            test_company_name_standardization()
        elif sys.argv[1] == "--test-consolidation":
            test_consolidation()
        else:
            print("用法:")
            print("  python consolidator.py --test-names      # 測試名稱標準化")
            print("  python consolidator.py --test-consolidation  # 測試彙整功能")
    else:
        test_company_name_standardization()