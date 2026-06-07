# Linear Regression Practice - CRISP-DM 中部空氣品質迴歸建模與異常偵測

本專案為**第四份職前培訓實務作業**，是一個基於 `Python` 與 `Streamlit` 的互動式線性迴歸分析與異常偵測展示工具。專案採用 **CRISP-DM**（跨行業數據挖掘標準流程）框架，系統性整合「迴歸模擬器」與「中部空氣品質實例（AQI 建模）」，實踐以迴歸殘差進行污染異常觀測篩選的決策支持系統原型。

🔗 [**Live Demo 線上展示**](https://linear-regression-practice-dec591nyc.streamlit.app/) | 💻 [**GitHub 專案倉庫**](https://github.com/dec591nyc/Linear-Regression-Practice)

---

## 專案 Infography

| 面向 | 內容 |
| --- | --- |
| 專案定位 | 線性迴歸實作、模型指標解讀與異常觀測偵測 |
| 基礎模組 | 以公式 `y = ax + b + noise` 示範資料，觀察迴歸行為，找出前 10 個殘差離群點 |
| 實際案例 | 以台中、彰化、南投空氣品質指標 (AQI) / PM2.5 數值資料進行污染指標預測與異常觀測排序 |
| 資料來源 | 優先使用環境部 `AQX_P_488` 歷史 AQI；Kaggle Taiwan AQI 作為歷史參考 |
| 介面功能 | 中英文切換、深淺色主題、資料下載、互動圖表 |
| 核心技術 | Streamlit、NumPy、Pandas、scikit-learn、Plotly |
| 展示重點 | 互動參數、迴歸線、殘差、Top 10 outliers、資料來源替代性評估 |

---

## 核心功能

- **模擬線性迴歸實驗**：透過側邊欄調整 `n`、`a`、`b`、`var` 與 random seed，建立示範資料；雜訊變異數與隨機種子同時提供友善輸入模式。
- **線性模型擬合**：迴歸模擬器與 AQI 案例皆可選擇 OLS、Ridge、Lasso、ElasticNet；
- **離群值偵測**：計算每筆資料的 residual 與 absolute residual，列出距離迴歸線最遠的前 10 筆觀測。
- **互動視覺化**：圖表同時呈現生成資料、真實線、迴歸線與 Top 10 outliers。
- **中部空氣品質指標 (AQI) 案例**：目前使用 `data/central_taiwan_aqi_sample.csv`，共 82,034 筆台中、彰化、南投空氣品質資料，支援目標、特徵、地區與線性模型切換。
- **雙語與主題切換**：支援繁體中文 / English，以及 light / dark theme。
- **CRISP-DM 實踐說明**：以 Business Understanding、Data Understanding、Data Preparation、Modeling、Evaluation、Deployment 六階段說明分析流程。
- **資料與報告下載**：使用者可下載 AQI 頁目前實際使用的 CSV；CRISP-DM markdown 報告會依目前中英文模式輸出。
- **Data Source**：說明目前使用資料、Simulator 示範資料、Kaggle 歷史參考與環境部官方來源。

---

## 資料來源

Simulator 分頁依公式模擬一份示範資料：`x` 從 -100 到 100 的均勻分布抽樣，`noise` 來自平均值 0、標準差 `sqrt(var)` 的常態分布，`y` 則由 `a*x + b + noise` 計算而來。左側的樣本數、斜率、截距、雜訊變異數與隨機種子都是資料生成條件。

AQI 分頁目前使用 `central_taiwan_aqi_sample.csv`，共有 82,034 筆，範圍包含台中市、彰化縣與南投縣。

這份資料用來檢視實際空氣品質案例的回歸建模、殘差排序與異常觀測。

本專案保留 Kaggle 的 [Taiwan Air Quality Index Data 2016~2024](https://www.kaggle.com/datasets/taweilo/taiwan-air-quality-data-20162024) 作為歷史資料參考。

正式資料以環境部官方歷史資料 [AQX_P_488](https://data.moenv.gov.tw/en/dataset/detail/aqx_p_488) 為優先。欄位包含測站名稱、縣市、AQI、SO2、CO、O3、PM10、PM2.5、NO2、風速、風向、發布時間與經緯度。


---

## 本機執行

建立虛擬環境後安裝套件：

```bash
pip install -r requirements.txt
```

啟動 Streamlit：

```bash
streamlit run app.py
```

啟動後在瀏覽器開啟：

```text
http://localhost:8501
```

---

## 目錄結構

```text
Linear-Regression-Practice/
├── app.py
├── requirements.txt
├── README.md
└── data/
    ├── central_taiwan_aqi_sample.csv
    └── CentraArea_Data.csv
```

---

## 技術重點

- **Streamlit**：建立互動式資料分析頁面與 tab 分頁。
- **Pandas**：讀取 CSV、整理欄位、篩選台中、彰化與南投資料。
- **NumPy**：產生模擬資料、計算殘差與 least-squares fallback。
- **scikit-learn**：建立線性迴歸模型並取得模型係數。
- **Plotly**：呈現互動散佈圖、迴歸線與模型預測結果。

---

## 開發收穫與實務價值

本專案作為第四份職前培訓實作作業，核心目標在於運用統計學與機器學習迴歸模型，模擬並建構符合實務情境的決策支持機制。本專案完整落實了以下分析流程與技術驗證：

- **模擬資料生成與參數控制**：透過合成數據公式深入探討隨機噪音、斜率與樣本量對迴歸線穩定度的影響。
- **線性模型擬合與多模型比較**：探討 OLS、Ridge、Lasso 與 ElasticNet 模型的共線性處理與係數特徵收斂。
- **殘差排序與異常值偵測**：將預測誤差最大化排序，作為資料清洗、硬體異常監控或突發空污源篩檢的第一道防線。
- **資料來源替代性評估**：對齊環境部監測標準（AQX_P_488 與 AQX_P_432）及封存方案，評估在不同環境下資料取得的穩健性。

透過將線性迴歸模式與空氣品質指標（AQI）建模實踐於中部地區（中彰投測站），我們成功以機器學習迴歸模型與殘差分析對應以下實務應用情境：
1. **公共事務（環境治理）**：輔助監測部門優先複查異常排放或測站儀器漂移。
2. **戶外活動安排（健康防護指引）**：分析特定污染物對 AQI 主導性，提早進行群眾健康風險提示。
3. **營運風險提醒（高污染預警）**：結合實時資料為工業區或園區營運提供早期空污預警，優化應變工作流。
