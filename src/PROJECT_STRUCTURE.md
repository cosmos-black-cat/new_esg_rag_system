# 📁 ESG報告書提取器 - 項目結構說明

## 🎯 項目概述

ESG報告書提取器 v1.0 是一個專門用於從ESG永續報告中提取再生塑膠相關數據的智能工具。經過全面簡化，現在專注於核心功能，提供更清晰的架構和更簡單的使用體驗。

## 📂 項目目錄結構

```
ESG報告書提取器/
├── 📋 核心程式文件
│   ├── main.py                    # 主程式入口
│   ├── esg_extractor.py          # ESG數據提取核心模組
│   ├── preprocess.py             # PDF預處理模組
│   ├── consolidator.py           # 結果彙整模組
│   ├── config.py                 # 系統配置模組
│   └── api_manager.py            # API管理模組
│
├── ⚙️ 配置和依賴文件
│   ├── .env                      # 環境變數配置（包含API Key）
│   ├── requirements.txt          # Python依賴包清單
│   └── quick_start.py           # 快速開始腳本
│
├── 📚 說明文件
│   ├── README.md                 # 主要使用說明
│   ├── PROJECT_STRUCTURE.md     # 本項目結構文件
│   └── CHANGELOG.md             # 更新記錄（如需要）
│
├── 📁 數據目錄
│   ├── data/                     # PDF輸入目錄
│   │   └── *.pdf                 # ESG報告PDF文件
│   │
│   ├── results/                  # Excel輸出目錄
│   │   ├── ESG提取結果_*.xlsx     # 個別公司結果
│   │   └── ESG彙整報告.xlsx       # 多公司彙整結果
│   │
│   └── vector_db/               # 向量資料庫目錄
│       └── esg_db_*/            # 各PDF對應的向量資料庫
│
└── 🗂️ 臨時和緩存目錄
    ├── __pycache__/             # Python編譯緩存
    └── .git/                    # Git版本控制（如使用）
```

## 🔧 核心模組說明

### 1. main.py - 主程式入口
- **功能**: 系統主入口，提供互動式選單和命令行介面
- **特色**: 
  - 清晰的用戶界面
  - 支援自動化執行
  - 完整的錯誤處理
  - 進度顯示和狀態反饋

### 2. esg_extractor.py - ESG數據提取核心
- **功能**: ESG報告中再生塑膠數據的智能提取
- **核心類**:
  - `ESGExtractor`: 主提取器類
  - `ESGMatcher`: 關鍵字與數值匹配引擎
  - `KeywordConfig`: 關鍵字配置管理
- **特色**:
  - 精確的關鍵字-數值關聯分析
  - 智能排除無關內容
  - 頁面級去重確保品質
  - 多層次相關性檢查

### 3. preprocess.py - PDF預處理模組
- **功能**: PDF文件解析和向量資料庫建立
- **核心類**:
  - `DocumentMetadataExtractor`: 文檔元數據提取器
- **特色**:
  - 智能公司名稱和年度識別
  - 高效的文本分割策略
  - 向量化文檔表示

### 4. consolidator.py - 結果彙整模組
- **功能**: 多公司多年度結果智能彙整
- **特色**:
  - 自動公司名稱標準化
  - 按年度和公司分組
  - 智能排除無效結果
  - 專業Excel報表生成

### 5. config.py - 系統配置模組
- **功能**: 系統參數和環境配置管理
- **特色**:
  - 環境變數自動載入
  - 配置驗證和錯誤檢查
  - 目錄自動創建

### 6. api_manager.py - API管理模組
- **功能**: Google Gemini API的智能管理
- **特色**:
  - 多API Key輪換
  - 自動重試和錯誤處理
  - 使用量統計和監控

## ⚙️ 配置文件說明

### .env 環境配置文件
```bash
# 必填配置
GOOGLE_API_KEY=your_api_key_here    # Google API Key

# 可選配置（有預設值）
CONFIDENCE_THRESHOLD=0.6            # 信心分數閾值
MAX_DOCS_PER_RUN=300               # 最大處理文檔數
ENABLE_LLM_ENHANCEMENT=true        # 是否啟用LLM增強
```

### requirements.txt 依賴包清單
包含所有必要的Python依賴包：
- AI/ML框架: langchain, transformers, torch
- 數據處理: pandas, numpy, scikit-learn
- 文件處理: pypdf, openpyxl
- API相關: google-generativeai

## 🚀 使用流程

### 1. 快速開始
```bash
# 系統檢查和設置
python quick_start.py --setup
python quick_start.py --check

# 運行主程式
python main.py
```

### 2. 典型工作流程
1. **準備**: 將ESG報告PDF放入`data/`目錄
2. **提取**: 執行功能1進行數據提取
3. **彙整**: 執行功能3生成彙整報告
4. **查看**: 在`results/`目錄查看Excel結果

### 3. 命令行模式
```bash
# 自動執行完整流程
python main.py --auto

# 分步執行
python main.py --preprocess  # 預處理
python main.py --extract     # 提取
python main.py --consolidate # 彙整
```

## 📊 輸出結果說明

### 個別公司結果文件
- **檔名**: `ESG提取結果_{公司名}_{年度}.xlsx`
- **工作表**:
  - `提取結果`: 主要提取數據
  - `統計摘要`: 關鍵字統計
  - `處理摘要`: 處理信息

### 彙整報告文件
- **檔名**: `ESG彙整報告.xlsx`
- **工作表**:
  - `2024年總覽`, `2023年總覽`等: 按年度分組
  - `{公司名}總覽`: 按公司分組
  - `彙整摘要`: 統計和映射信息

## 🎯 系統特色

### 1. 專業性
- **專注領域**: 僅針對再生塑膠相關數據
- **精準提取**: 高精度的關鍵字-數值匹配
- **智能過濾**: 自動排除職災、賽事等無關內容

### 2. 易用性
- **一鍵開始**: 快速設置腳本
- **直觀介面**: 清晰的互動式選單
- **自動化**: 支援命令行批量處理

### 3. 可靠性
- **錯誤處理**: 完善的異常處理機制
- **數據驗證**: 多層次的結果驗證
- **去重機制**: 精確和頁面級去重

### 4. 可維護性
- **模組化**: 清晰的模組分工
- **配置化**: 靈活的參數調整
- **文檔化**: 詳細的說明文件

## 🔧 開發和維護

### 代碼結構
- **單一職責**: 每個模組負責特定功能
- **低耦合**: 模組間依賴最小化
- **高內聚**: 相關功能集中在同一模組

### 擴展性
- **新關鍵字**: 在`esg_extractor.py`的`KeywordConfig`中添加
- **新規則**: 在匹配引擎中添加新的判斷邏輯
- **新功能**: 可以新增模組而不影響現有功能

### 配置調優
- **提高精度**: 增加`CONFIDENCE_THRESHOLD`
- **提高覆蓋**: 降低`CONFIDENCE_THRESHOLD`
- **控制成本**: 調整`MAX_DOCS_PER_RUN`

---

📊 **ESG報告書提取器 v1.0** - 簡潔、專業、高效的ESG數據提取解決方案