from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

try:
    from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
    from sklearn.metrics import r2_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
except Exception:  # pragma: no cover - fallback keeps the demo readable without sklearn.
    ElasticNet = None
    Lasso = None
    LinearRegression = None
    Ridge = None
    StandardScaler = None
    make_pipeline = None


PROJECT_ROOT = Path(__file__).resolve().parent
PRIMARY_AQI_PATH = PROJECT_ROOT / "data" / "central_taiwan_aqi_sample.csv"
LOCAL_CENTRAL_AQI_PATH = PROJECT_ROOT / "data" / "CentraArea_Data.csv"
SAMPLE_AQI_PATH = (
    PRIMARY_AQI_PATH
    if PRIMARY_AQI_PATH.exists()
    else LOCAL_CENTRAL_AQI_PATH
    if LOCAL_CENTRAL_AQI_PATH.exists()
    else None
)
AQI_DATA_IS_PRIMARY = SAMPLE_AQI_PATH == PRIMARY_AQI_PATH
AQI_DATA_IS_LOCAL_HISTORICAL = SAMPLE_AQI_PATH == LOCAL_CENTRAL_AQI_PATH
AQI_DATA_AVAILABLE = SAMPLE_AQI_PATH is not None
CENTRAL_COUNTIES = ["臺中市", "台中市", "彰化縣", "南投縣"]
REGION_FILTERS = {
    "central": ["臺中市", "台中市", "Taichung City", "彰化縣", "Changhua County", "南投縣", "Nantou County"],
    "taichung": ["臺中市", "台中市", "Taichung City"],
    "changhua": ["彰化縣", "Changhua County"],
    "nantou": ["南投縣", "Nantou County"],
}
DEFAULT_SOURCE_NAME = "Kaggle Taiwan Air Quality Index Data 2016~2024"
DEFAULT_SOURCE_URL = "https://www.kaggle.com/datasets/taweilo/taiwan-air-quality-data-20162024"
OFFICIAL_SOURCE_NAME = "MOENV AQX_P_488"
OFFICIAL_SOURCE_URL = "https://data.moenv.gov.tw/en/dataset/detail/aqx_p_488"
REALTIME_SOURCE_NAME = "MOENV AQX_P_432"
REALTIME_SOURCE_URL = "https://data.moenv.gov.tw/dataset/detail/aqx_p_432"
SAMPLE_AQI_ROWS = max(0, sum(1 for _ in SAMPLE_AQI_PATH.open(encoding="utf-8")) - 1) if SAMPLE_AQI_PATH else 0

TEXT = {
    "en": {
        "page_title": "Linear Regression Practice",
        "hero_title": "Linear Regression Practice",
        "hero_subtitle": "Lab uses synthetic data to explain regression mechanics; AQI uses central Taiwan air-quality records for the practical case.",
        "theme": "Theme",
        "language": "Language",
        "sidebar_toggle": "Toggle sidebar",
        "to_zh": "🌐 繁中",
        "to_en": "🌐 EN",
        "light": "Light",
        "dark": "Dark",
        "theme_light_btn": "☀️",
        "theme_dark_btn": "🌙",
        "regression_tab": "Regression Sandbox",
        "aqi_tab": "Central AQI",
        "crisp_tab": "CRISP-DM Report",
        "source_tab": "Data Source",
        "nav_lab": "Lab",
        "nav_aqi": "AQI",
        "nav_report": "Report",
        "nav_data": "Data Source",
        "synthetic_title": "Regression Sandbox",
        "synthetic_note": "This sandbox keeps the required numeric regression workflow: generate data with n, a, b, and variance, fit a line, then rank the top residual outliers.",
        "synthetic_source_title": "Regression sandbox data source",
        "synthetic_source_body": "This model uses synthetic numeric data generated in memory by app.py from y = a*x + b + noise. It has no CSV file. It is not teacher-provided classroom data, not Kaggle data, and not MOENV records. The parameters in the sidebar are the data source for Lab.",
        "synthetic_purpose": "Design purpose: the sandbox isolates the core regression requirement before moving to real air-quality data. It helps a non-technical user see that residuals are simply the gaps between a model's guess and the actual generated value.",
        "synthetic_params": "Synthetic Parameters",
        "sample_size": "Sample size (n)",
        "slope": "True slope (a)",
        "intercept": "True intercept (b)",
        "variance_preset": "Noise preset",
        "variance": "Noise variance (var)",
        "seed_preset": "Seed preset",
        "seed": "Random seed",
        "r2": "R-squared",
        "rmse": "RMSE",
        "mae": "MAE",
        "model": "Model",
        "model_choice": "Linear model",
        "model_comparison": "Linear model comparison",
        "model_profile_title": "Model reading",
        "model_ols": "OLS Linear Regression",
        "model_ridge": "Ridge Regression",
        "model_lasso": "Lasso Regression",
        "model_elasticnet": "ElasticNet Regression",
        "model_ols_reading": "OLS is the clearest baseline. It is easy to explain, but coefficients can become unstable when input features move together.",
        "model_ridge_reading": "Ridge keeps all selected features but shrinks their influence. It is useful when several input fields are correlated.",
        "model_lasso_reading": "Lasso can shrink weak feature effects toward zero. It is useful for feature screening, but it may be too aggressive on a small sample.",
        "model_elasticnet_reading": "ElasticNet balances Ridge and Lasso. It is a practical compromise when features are correlated but some simplification is still useful.",
        "coefficient_note": "For Ridge, Lasso and ElasticNet, coefficients are based on standardized inputs, so compare direction and relative size rather than raw units.",
        "equation_regularized_note": "For this regularized model, the displayed coefficients are standardized-model coefficients. Use the fitted line, metrics and residuals for interpretation instead of reading the formula as raw x/y units.",
        "equation": "Estimated equation",
        "equation_note": "The equation is shown for every selected linear model. For Ridge, Lasso and ElasticNet, coefficients are based on standardized inputs, so use it as a model-reading aid rather than a raw-unit business formula.",
        "generated_data": "Generated data",
        "outliers": "Top 10 residual outliers",
        "outlier_plain_note": "Residual means the gap between the actual value and the model's prediction. The top 10 residual rows are the observations where the model was most wrong, so they are useful for checking unusual records first.",
        "true_line": "True line",
        "regression_line": "Regression line",
        "top_outliers": "Top residual observations",
        "outlier_scope_note": "Showing the top 10 rows ranked by absolute residual. The full table is not shown because the active AQI dataset can contain tens of thousands of rows; this view focuses on the observations most worth manual review.",
        "aqi_title": "Central Taiwan Air Quality Index (AQI) Modeling",
        "aqi_note": "Air Quality Index (AQI) is a single number that summarizes air-pollution level. This case prioritizes recent Taichung, Changhua and Nantou observations, then predicts a numeric target and surfaces observations that the baseline model cannot explain well.",
        "aqi_params": "AQI Controls",
        "region": "Region",
        "region_central": "Central Taiwan",
        "region_taichung": "Taichung",
        "region_changhua": "Changhua",
        "region_nantou": "Nantou",
        "active_data": "Active data",
        "data_rows": "Rows",
        "data_scope": "Scope",
        "data_period": "Period",
        "recent_sample": "MOENV central Taiwan AQI sample",
        "local_historical_sample": "Local Taichung historical sample",
        "local_historical_warning": "The active CSV has enough rows, but it is a Taichung-only historical file from 2016-2026. After filtering to 2018 or later, it has 966 rows, so it does not fully satisfy the requested recent central-Taiwan sample condition.",
        "no_active_data": "No usable AQI CSV is available. The tiny demo fallback has been removed by project decision. Add a larger public dataset or switch to another research topic.",
        "target": "Target variable",
        "features": "Feature variables",
        "source_run": "Data source in this run",
        "complete_rows": "Complete rows used",
        "bundled_sample": "Bundled central Taiwan sample",
        "coefficients": "Coefficients",
        "actual_predicted": "Actual vs Predicted",
        "perfect_prediction": "Perfect prediction",
        "source_panel_title": "Active data source",
        "source_panel_body": "AQI modeling reads the local CSV shown below. The active file is a central Taiwan MOENV-style Air Quality Index dataset with Taichung, Changhua and Nantou records, kept within the requested 20,000-100,000 row range. Kaggle is kept as a reproducible historical reference, not the active app file.",
        "sample_scope_note": "Sample scope: the active CSV has {rows} rows. Tiny demo data is not used.",
        "download_data": "Download active AQI CSV",
        "download_report": "Download CRISP-DM report",
        "source_title": "Data Source",
        "source_body": """
        **What this page answers**

        - **AQI case:** uses `data/central_taiwan_aqi_sample.csv`, the active central Taiwan AQI CSV in this repository.
        - **Lab:** does not read any CSV. It generates synthetic values inside `app.py` from `y = a*x + b + noise`, so Lab is not teacher data, Kaggle data, or MOENV data.
        - **Why Kaggle is listed:** Kaggle `Taiwan Air Quality Index Data 2016~2024` is a reproducible historical reference if the project needs a public archived source.
        - **Why MOENV is listed:** MOENV `AQX_P_488` is the official historical AQI source, and `AQX_P_432` is the real-time hourly reference.
        - **Why there is no tiny fallback:** 24-row demo data was removed because it cannot support model conclusions.
        - **What the download button does:** it downloads the exact CSV currently used by the AQI page.
        """,
        "too_few_cols": "AQI data needs at least two numeric columns after cleaning.",
        "select_feature": "Select at least one feature variable.",
        "too_few_rows": "The selected columns have too few complete rows for a useful regression.",
        "metric_status_title": "Metric status",
        "metric_r2_strong": "R-squared is strong. The model explains most of the target variation in this run.",
        "metric_r2_ok": "R-squared is moderate. The model captures part of the pattern, but residual checks still matter.",
        "metric_r2_weak": "R-squared is weak. Treat this as a baseline model, not a reliable predictor.",
        "metric_error_note": "RMSE and MAE are average error indicators. Lower values mean the prediction is closer to the actual target.",
        "metric_outlier_note": "Rows with the largest absolute residuals are the observations most worth checking manually.",
        "chart_plain_title": "Plain-language reading",
        "chart_plain_aqi": "Each dot is one Air Quality Index (AQI) record or selected numeric air-quality target. The horizontal position is what the model guessed, and the vertical position is the real value. Dots close to the diagonal line mean the model guessed well. Dots far from the line are records worth checking because the pollutant pattern looked unusual.",
        "chart_plain_stats": "In this run, the average {target} is {avg:.1f}. The largest miss is about {residual:.1f}. This does not automatically mean the air is dangerous; it means the reading does not match the simple pattern learned from the selected pollutants.",
        "chart_plain_caution": "Because this is a small baseline demo, use it to understand patterns and suspicious records, not as an official air-quality forecast.",
        "crisp_title": "CRISP-DM practice report",
        "crisp_intro": "CRISP-DM turns this project from a charting demo into a repeatable data-mining workflow: first define the air-quality question, then understand data fields, prepare numeric inputs, train a baseline model, evaluate residuals, and package the result for reuse.",
        "crisp_business": "Business understanding: identify air-quality observations that do not follow the usual pollutant pattern, so a user can inspect unusual AQI readings faster.",
        "crisp_data": "Data understanding: use station-level Air Quality Index (AQI) fields such as PM2.5, PM10, O3, NO2, CO, SO2 and wind speed. The preferred app dataset is recent central Taiwan data after 2018; tiny demo data is not used.",
        "crisp_prep": "Data preparation: normalize column names, keep numeric fields, remove incomplete rows, and let users choose the prediction target and features.",
        "crisp_model": "Modeling: train a simple linear regression baseline and produce predicted values.",
        "crisp_eval": "Evaluation: use R-squared, RMSE, MAE and residual ranking. The largest residuals become the records worth checking manually.",
        "crisp_deploy": "Deployment: provide a bilingual Streamlit interface, active AQI CSV download, source notes and a downloadable CRISP-DM report. External upload is intentionally removed until a schema validator is defined.",
    },
    "zh": {
        "page_title": "線性迴歸實作",
        "hero_title": "線性迴歸實作",
        "hero_subtitle": "Lab 用合成資料理解迴歸；AQI 頁才使用中部空氣品質資料做實際案例。",
        "theme": "主題",
        "language": "語言",
        "sidebar_toggle": "切換側邊欄",
        "to_zh": "🌐 繁中",
        "to_en": "🌐 EN",
        "light": "淺色",
        "dark": "深色",
        "theme_light_btn": "☀️",
        "theme_dark_btn": "🌙",
        "regression_tab": "迴歸模擬器",
        "aqi_tab": "中部 AQI",
        "crisp_tab": "CRISP-DM 報告",
        "source_tab": "Data Source",
        "nav_lab": "Lab",
        "nav_aqi": "AQI",
        "nav_report": "Report",
        "nav_data": "Data Source",
        "synthetic_title": "迴歸模擬器",
        "synthetic_note": "此區保留數值迴歸的核心流程：用 n、a、b、變異數產生資料，擬合迴歸線，再用殘差排序找出異常觀測。",
        "synthetic_source_title": "迴歸模擬器資料來源",
        "synthetic_source_body": "這個模型使用 app.py 依照 y = a*x + b + noise 在記憶體即時產生的合成數值資料，沒有對應 CSV 檔。它不是老師課堂提供資料、不是 Kaggle，也不是環境部資料。Lab 的資料來源就是左側參數：樣本數、斜率、截距、雜訊變異數與隨機種子。",
        "synthetic_purpose": "設計用意：模擬器先隔離出回歸作業的核心流程，再接到真實空氣品質案例。對非技術使用者來說，殘差就是模型猜測值和實際值之間的差距。",
        "synthetic_params": "模擬參數",
        "sample_size": "樣本數 (n)",
        "slope": "真實斜率 (a)",
        "intercept": "真實截距 (b)",
        "variance_preset": "雜訊區段",
        "variance": "雜訊變異數 (var)",
        "seed_preset": "隨機種子區段",
        "seed": "隨機種子",
        "r2": "R-squared",
        "rmse": "RMSE",
        "mae": "MAE",
        "model": "模型",
        "model_choice": "線性模型",
        "model_comparison": "線性模型比較",
        "model_profile_title": "模型解讀",
        "model_ols": "OLS 普通最小平方法",
        "model_ridge": "Ridge 迴歸",
        "model_lasso": "Lasso 迴歸",
        "model_elasticnet": "ElasticNet 迴歸",
        "model_ols_reading": "OLS 是最清楚的 baseline，容易解釋；但輸入特徵彼此高度相關時，係數可能不穩定。",
        "model_ridge_reading": "Ridge 會保留所有選取特徵，但降低個別特徵的影響力；當多個輸入欄位彼此相關時，通常較穩定。",
        "model_lasso_reading": "Lasso 會把較弱的特徵影響壓向 0，適合做特徵篩選；但在小型 sample 上可能過度簡化。",
        "model_elasticnet_reading": "ElasticNet 折衷 Ridge 與 Lasso，適合特徵彼此相關、但仍希望模型稍微簡化的情境。",
        "coefficient_note": "Ridge、Lasso、ElasticNet 的係數來自標準化輸入，適合比較方向與相對影響，不宜直接解讀為原始單位。",
        "equation_regularized_note": "此正規化模型的係數是標準化後的模型係數；解讀時請優先看擬合線、指標與殘差，不要把公式直接當成原始 x/y 單位。",
        "equation": "估計方程式",
        "equation_note": "每一種線性模型都會顯示方程式，維持呈現一致。Ridge、Lasso、ElasticNet 的係數來自標準化輸入，適合輔助理解模型，不建議直接當成原始單位的商業公式。",
        "generated_data": "模擬資料",
        "outliers": "殘差前 10 名",
        "outlier_plain_note": "殘差就是實際值和模型預測值之間的差距。殘差前 10 名代表模型最猜不準的 10 筆資料，適合優先人工檢查是否有異常或特殊情況。",
        "true_line": "真實線",
        "regression_line": "迴歸線",
        "top_outliers": "殘差最高觀測",
        "outlier_scope_note": "此處只顯示依絕對殘差排序前 10 筆。完整資料可能有數萬筆，全部列出會干擾閱讀；這個區塊聚焦最值得人工檢查的觀測。",
        "aqi_title": "中部空氣品質指標 (AQI) 建模",
        "aqi_note": "空氣品質指標 (Air Quality Index, AQI) 是把空氣污染狀況整理成單一數字的指標。此區優先使用台中、彰化、南投的近期觀測，預測數值目標，並找出基礎模型較難解釋的污染觀測。",
        "aqi_params": "AQI 控制項",
        "region": "地區",
        "region_central": "中部",
        "region_taichung": "台中",
        "region_changhua": "彰化",
        "region_nantou": "南投",
        "active_data": "目前資料",
        "data_rows": "資料筆數",
        "data_scope": "資料範圍",
        "data_period": "資料期間",
        "recent_sample": "環境部中部 AQI sample",
        "local_historical_sample": "本機台中歷史 sample",
        "local_historical_warning": "目前啟用 CSV 的筆數足夠，但它是 2016-2026 的台中歷史資料；若只保留 2018 以後，只剩 966 筆，因此不完全符合近期中部 sample 條件。",
        "no_active_data": "目前沒有可用的 AQI CSV。依專案決策，微型 demo fallback 已移除；請加入較大的公開資料，或改用其他研究主題。",
        "target": "預測目標",
        "features": "特徵欄位",
        "source_run": "本次資料來源",
        "complete_rows": "完整資料筆數",
        "bundled_sample": "內建中彰 sample",
        "coefficients": "模型係數",
        "actual_predicted": "實際值與預測值",
        "perfect_prediction": "理想預測線",
        "source_panel_title": "目前使用資料",
        "source_panel_body": "AQI 建模讀取下方列出的本機 CSV。啟用檔案是台中、彰化、南投的中部空氣品質指標資料，樣本數控制在 2 萬到 10 萬筆。Kaggle 保留為可重現歷史參考，不是目前 app 實際讀取檔案。",
        "sample_scope_note": "樣本範圍：目前啟用 CSV 有 {rows} 筆。24 筆微型 demo 不再使用。",
        "download_data": "下載目前 AQI CSV",
        "download_report": "下載 CRISP-DM 報告",
        "source_title": "Data Source",
        "source_body": """
        **這頁回答三件事**

        - **AQI 案例用什麼：**使用 `data/central_taiwan_aqi_sample.csv`，也就是此 repo 目前啟用的中部 AQI CSV。
        - **Lab 用什麼：**Lab 不讀取任何 CSV，而是在 `app.py` 依照 `y = a*x + b + noise` 即時合成資料，所以不是老師資料、不是 Kaggle、也不是環境部資料。
        - **Kaggle 為何列出：**`Taiwan Air Quality Index Data 2016~2024` 是可重現的歷史參考來源，當需要公開 archived data 時可以回頭採用。
        - **環境部為何列出：**`AQX_P_488` 是官方歷史 AQI 來源，`AQX_P_432` 是即時每小時 AQI 參考。
        - **為何沒有 24 筆 fallback：**24 筆 demo 無法支撐模型結論，因此已移除。
        - **下載按鈕做什麼：**下載 AQI 頁面目前實際使用的同一份 CSV。
        """,
        "too_few_cols": "清理後的 AQI 資料至少需要兩個數值欄位。",
        "select_feature": "請至少選擇一個特徵欄位。",
        "too_few_rows": "所選欄位的完整資料筆數太少，不適合建立迴歸模型。",
        "metric_status_title": "指標狀況",
        "metric_r2_strong": "R-squared 表現佳，代表本次模型能解釋多數目標變化。",
        "metric_r2_ok": "R-squared 表現中等，模型抓到部分規律，但仍需要看殘差。",
        "metric_r2_weak": "R-squared 偏弱，這次結果適合視為 baseline，不宜當作可靠預測模型。",
        "metric_error_note": "RMSE 與 MAE 是平均誤差指標，數值越低代表預測越接近實際目標。",
        "metric_outlier_note": "絕對殘差最大的資料列，就是最值得人工檢查的異常觀測。",
        "chart_plain_title": "白話解讀",
        "chart_plain_aqi": "每一個點代表一筆空氣品質指標 (Air Quality Index, AQI) 或目前選定的空氣品質數值目標。橫軸是模型猜出的數值，縱軸是真實數值。點越靠近斜線，代表模型猜得越準；離斜線越遠，代表這筆資料和一般污染物規律不太一樣，值得再檢查。",
        "chart_plain_stats": "本次資料的平均 {target} 約為 {avg:.1f}，最大誤差約為 {residual:.1f}。這不一定代表空氣很危險，而是代表這筆觀測不太符合目前模型學到的簡單規律。",
        "chart_plain_caution": "因為這是小型 baseline demo，適合用來理解污染趨勢與可疑觀測，不適合作為正式空氣品質預報。",
        "crisp_title": "CRISP-DM 實踐報告",
        "crisp_intro": "CRISP-DM 讓這個專案不只是畫圖工具，而是一個可重複使用的數據探勘流程：先定義空氣品質問題，再理解資料欄位、整理數值特徵、建立 baseline 模型、評估殘差，最後包裝成可操作的分析介面。",
        "crisp_business": "商業理解：找出不符合一般污染物規律的空氣品質觀測，協助使用者更快注意可疑 AQI 讀數。",
        "crisp_data": "資料理解：使用測站層級空氣品質指標 (AQI) 欄位，例如 PM2.5、PM10、O3、NO2、CO、SO2 與風速；理想 app 資料是 2018 以後近期中部資料，微型 demo 資料不再使用。",
        "crisp_prep": "資料準備：正規化欄位名稱、保留數值欄位、移除不完整資料，並讓使用者選擇預測目標與特徵。",
        "crisp_model": "模型建立：訓練簡單線性迴歸 baseline，產生每筆觀測的預測值。",
        "crisp_eval": "模型評估：使用 R-squared、RMSE、MAE 與殘差排序；殘差最大的資料就是最值得人工檢查的觀測。",
        "crisp_deploy": "部署應用：提供雙語 Streamlit 介面、目前 AQI CSV 下載、資料來源說明與可下載 CRISP-DM 報告。外部上傳在尚未定義 schema 驗證前先移除。",
    },
}


st.set_page_config(
    page_title="Linear Regression Practice",
    page_icon="AQI",
    layout="wide",
    initial_sidebar_state="expanded",
)

if "theme" not in st.session_state:
    st.session_state.theme = "light"
if "locale" not in st.session_state:
    st.session_state.locale = "zh"
if "sidebar_collapsed" not in st.session_state:
    st.session_state.sidebar_collapsed = False
if "active_page" not in st.session_state:
    st.session_state.active_page = "lab"

t = TEXT[st.session_state.locale].get

THEMES = {
    "light": {
        "bg": "#f4f6fa",
        "panel": "#ffffff",
        "panel_alt": "#f8fafc",
        "text": "#111827",
        "muted": "#4b5563",
        "border": "#cfd8e3",
        "accent": "#0f766e",
        "accent_text": "#ffffff",
        "note_bg": "#ecfdf5",
        "metric_bg": "#ffffff",
        "metric_value": "#0f172a",
        "plot_bg": "#ffffff",
        "plot_text": "#111827",
        "plot_muted": "#334155",
        "plot_grid": "#cbd5e1",
        "plot_axis": "#475569",
        "control_bg": "#ffffff",
        "control_text": "#111827",
        "control_muted": "#6b7280",
        "button_bg": "#ffffff",
        "button_text": "#111827",
        "button_hover_bg": "#f1f5f9",
        "button_border": "#94a3b8",
        "button_primary_bg": "#0f766e",
        "button_primary_text": "#ffffff",
        "table_header_bg": "#e2e8f0",
        "table_header_text": "#0f172a",
    },
    "dark": {
        "bg": "#0f172a",
        "panel": "#182235",
        "panel_alt": "#111827",
        "text": "#edf2ff",
        "muted": "#a8b3c7",
        "border": "#2f3b52",
        "accent": "#38bdf8",
        "accent_text": "#0f172a",
        "note_bg": "#14243a",
        "metric_bg": "#182235",
        "metric_value": "#f8fafc",
        "plot_bg": "#111827",
        "plot_text": "#f8fafc",
        "plot_muted": "#cbd5e1",
        "plot_grid": "#334155",
        "plot_axis": "#94a3b8",
        "control_bg": "#111827",
        "control_text": "#f8fafc",
        "control_muted": "#a8b3c7",
        "button_bg": "#1f2937",
        "button_text": "#f8fafc",
        "button_hover_bg": "#263244",
        "button_border": "#475569",
        "button_primary_bg": "#38bdf8",
        "button_primary_text": "#0f172a",
        "table_header_bg": "#111827",
        "table_header_text": "#f8fafc",
    },
}
theme = THEMES[st.session_state.theme]
plot_template = "plotly_dark" if st.session_state.theme == "dark" else "plotly_white"
plot_config = {"displayModeBar": False, "responsive": True}
sidebar_visibility_css = ""
active_nav_css = f"""
    [class*="st-key-nav_{st.session_state.active_page}"] button {{
        background: {theme["button_bg"]} !important;
        background-color: {theme["button_bg"]} !important;
        color: {theme["accent"]} !important;
        border-color: {theme["button_primary_bg"]} !important;
        box-shadow: inset 0 -3px 0 {theme["button_primary_bg"]} !important;
    }}
    [class*="st-key-nav_{st.session_state.active_page}"] button * {{
        color: {theme["accent"]} !important;
        fill: currentColor !important;
    }}
"""
if st.session_state.sidebar_collapsed:
    sidebar_visibility_css = """
    [data-testid="stSidebar"],
    section[data-testid="stSidebar"] {
        display: none !important;
        width: 0 !important;
        min-width: 0 !important;
    }
    [data-testid="stMain"],
    [data-testid="stMainBlockContainer"] {
        margin-left: 0 !important;
    }
    """

st.markdown(
    f"""
    <style>
    .stAppToolbar, #MainMenu, footer {{
        display: none !important;
        visibility: hidden !important;
        height: 0 !important;
    }}
    [data-testid="stSidebarCollapseButton"] {{
        display: none !important;
        visibility: hidden !important;
        pointer-events: none !important;
    }}
    html,
    body,
    [data-testid="stAppViewContainer"],
    [data-testid="stAppViewBlockContainer"],
    [data-testid="stMain"],
    [data-testid="stMainBlockContainer"] {{
        background: {theme["bg"]} !important;
        color: {theme["text"]} !important;
    }}
    [data-testid="stApp"], .stApp {{
        background: {theme["bg"]} !important;
        color: {theme["text"]} !important;
    }}
    .block-container {{
        max-width: 1180px;
        padding-top: 1.75rem;
        padding-bottom: 2.25rem;
    }}
    [data-testid="stHeader"],
    .stAppHeader {{
        background: transparent !important;
        height: 0 !important;
        min-height: 0 !important;
        pointer-events: none !important;
    }}
    [data-testid="stHeader"] * {{
        color: {theme["text"]} !important;
    }}
    [data-testid="stToolbar"],
    [data-testid="stToolbar"] * {{
        visibility: visible !important;
        opacity: 1 !important;
        pointer-events: auto !important;
        color: {theme["text"]} !important;
        fill: {theme["text"]} !important;
    }}
    [data-testid="stToolbar"] {{
        top: 0.55rem !important;
        right: 0.75rem !important;
        pointer-events: auto !important;
    }}
    [class*="st-key-sidebar_state_toggle"] {{
        position: fixed !important;
        top: 12px !important;
        left: 12px !important;
        width: 36px !important;
        height: 36px !important;
        z-index: 2147483647 !important;
    }}
    [class*="st-key-sidebar_state_toggle"] button {{
        width: 36px !important;
        height: 36px !important;
        min-height: 36px !important;
        padding: 0 !important;
        border-radius: 8px !important;
        border: 1px solid {theme["button_border"]} !important;
        background: {theme["button_bg"]} !important;
        background-color: {theme["button_bg"]} !important;
        color: {theme["button_text"]} !important;
        box-shadow: 0 8px 24px rgba(15, 23, 42, 0.18) !important;
        font-size: 18px !important;
        font-weight: 700 !important;
        line-height: 1 !important;
    }}
    [data-testid="stSidebar"],
    section[data-testid="stSidebar"] {{
        background: {theme["panel"]} !important;
        border-right: 1px solid {theme["border"]} !important;
    }}
    [data-testid="column"],
    [data-testid="stColumn"],
    [data-testid="stHorizontalBlock"],
    [data-testid="stVerticalBlock"],
    [data-testid="stElementContainer"] {{
        background: transparent !important;
        color: {theme["text"]} !important;
    }}
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"],
    [data-testid="stAppViewContainer"] label,
    [data-testid="stAppViewContainer"] [data-testid="stMarkdownContainer"],
    [data-testid="stAppViewContainer"] [data-testid="stCaptionContainer"] {{
        color: {theme["text"]} !important;
    }}
    [data-testid="stSidebar"] small,
    [data-testid="stSidebar"] [data-testid="stWidgetLabel"],
    [data-testid="stAppViewContainer"] [data-testid="stWidgetLabel"] {{
        color: {theme["muted"]} !important;
    }}
    .stTabs [data-baseweb="tab"] {{
        color: {theme["muted"]} !important;
        font-size: 0.94rem !important;
    }}
    .stTabs [data-baseweb="tab"][aria-selected="true"],
    .stTabs [data-baseweb="tab"][aria-selected="true"] * {{
        color: {theme["accent"]} !important;
    }}
    [data-baseweb="select"] > div,
    [data-baseweb="input"] > div,
    [data-baseweb="textarea"] > div,
    [data-testid="stNumberInput"] input {{
        background: {theme["control_bg"]} !important;
        color: {theme["control_text"]} !important;
        border-color: {theme["border"]} !important;
    }}
    [data-baseweb="select"] span,
    [data-baseweb="select"] svg,
    [data-baseweb="input"] input,
    [data-testid="stNumberInput"] input {{
        color: {theme["control_text"]} !important;
        fill: {theme["control_text"]} !important;
    }}
    [data-testid="stNumberInput"] button {{
        background: {theme["control_bg"]} !important;
        background-color: {theme["control_bg"]} !important;
        color: {theme["control_text"]} !important;
        border-color: {theme["border"]} !important;
    }}
    [data-testid="stNumberInput"] button svg {{
        color: {theme["control_text"]} !important;
        fill: {theme["control_text"]} !important;
    }}
    [data-baseweb="popover"],
    [data-baseweb="popover"] [role="listbox"],
    [data-baseweb="popover"] [role="option"] {{
        background: {theme["control_bg"]} !important;
        color: {theme["control_text"]} !important;
    }}
    [data-baseweb="popover"] [role="option"]:hover {{
        background: {theme["note_bg"]} !important;
        color: {theme["control_text"]} !important;
    }}
    [data-baseweb="tag"] {{
        background: #ef6359 !important;
        color: #ffffff !important;
    }}
    [data-baseweb="tag"] span,
    [data-baseweb="tag"] svg {{
        color: #ffffff !important;
        fill: #ffffff !important;
    }}
    [data-testid="stFileUploaderDropzone"] {{
        background: {theme["control_bg"]} !important;
        border: 1px solid {theme["button_border"]} !important;
        border-radius: 8px !important;
    }}
    [data-testid="stFileUploaderDropzone"] * {{
        color: {theme["control_text"]} !important;
    }}
    [data-testid="stFileUploaderDropzone"] small {{
        color: {theme["control_muted"]} !important;
    }}
    div.stButton > button,
    div[data-testid="stDownloadButton"] > button,
    button[data-testid^="stBaseButton"] {{
        background: {theme["button_bg"]} !important;
        background-color: {theme["button_bg"]} !important;
        color: {theme["button_text"]} !important;
        border: 1px solid {theme["button_border"]} !important;
        border-radius: 8px !important;
        font-weight: 650 !important;
    }}
    div.stButton > button *,
    div[data-testid="stDownloadButton"] > button *,
    button[data-testid^="stBaseButton"] * {{
        color: inherit !important;
        fill: currentColor !important;
    }}
    div.stButton > button:hover,
    div[data-testid="stDownloadButton"] > button:hover,
    button[data-testid^="stBaseButton"]:hover {{
        border-color: {theme["accent"]} !important;
        background: {theme["button_hover_bg"]} !important;
        background-color: {theme["button_hover_bg"]} !important;
        color: {theme["accent"]} !important;
    }}
    [class*="st-key-nav_"] button,
    [class*="st-key-nav_"] button:hover {{
        min-height: 40px !important;
        border-radius: 8px !important;
    }}
    [class*="st-key-nav_"] button[kind="primary"] {{
        background: {theme["button_primary_bg"]} !important;
        background-color: {theme["button_primary_bg"]} !important;
        color: {theme["button_primary_text"]} !important;
        border-color: {theme["button_primary_bg"]} !important;
    }}
    [class*="st-key-nav_"] button[kind="primary"] * {{
        color: {theme["button_primary_text"]} !important;
        fill: currentColor !important;
    }}
    [class*="st-key-nav_"] button[kind="secondary"] {{
        background: {theme["button_bg"]} !important;
        background-color: {theme["button_bg"]} !important;
        color: {theme["button_text"]} !important;
        border-color: {theme["button_border"]} !important;
    }}
    {active_nav_css}
    div[data-testid="stSlider"] * {{
        color: {theme["text"]} !important;
    }}
    div[data-testid="stSlider"] [role="slider"] {{
        background-color: #ef6359 !important;
        border-color: #ef6359 !important;
    }}
    [data-testid="stMetric"] {{
        background: {theme["metric_bg"]};
        border: 1px solid {theme["border"]};
        border-radius: 8px;
        padding: 0.85rem 0.95rem;
        min-height: 96px;
    }}
    [data-testid="stMetricLabel"] {{
        color: {theme["muted"]} !important;
        font-size: 0.82rem !important;
    }}
    [data-testid="stMetricValue"] {{
        color: {theme["metric_value"]} !important;
        font-size: 1.35rem !important;
        line-height: 1.25 !important;
    }}
    [data-testid="stMetric"], [data-testid="stDataFrame"], .stTabs [data-baseweb="tab-list"] {{
        color: {theme["text"]};
    }}
    [data-testid="stTable"],
    [data-testid="stDataFrame"],
    [data-testid="stDataFrameResizable"],
    .stDataFrame,
    .stTable {{
        background: {theme["panel"]} !important;
        color: {theme["text"]} !important;
        border: 1px solid {theme["border"]} !important;
        border-radius: 8px !important;
    }}
    [data-testid="stTable"] table,
    [data-testid="stTable"] tbody,
    [data-testid="stTable"] tbody tr,
    [data-testid="stTable"] tbody th,
    [data-testid="stTable"] td {{
        background: {theme["panel"]} !important;
        color: {theme["text"]} !important;
        border-color: {theme["border"]} !important;
    }}
    [data-testid="stTable"] thead,
    [data-testid="stTable"] thead tr,
    [data-testid="stTable"] thead th {{
        background: {theme["table_header_bg"]} !important;
        color: {theme["table_header_text"]} !important;
        border-color: {theme["border"]} !important;
    }}
    [data-testid="stDataFrame"] [role="grid"],
    [data-testid="stDataFrame"] [role="gridcell"] {{
        background: {theme["panel"]} !important;
        color: {theme["text"]} !important;
        border-color: {theme["border"]} !important;
    }}
    [data-testid="stDataFrame"] [role="columnheader"] {{
        background: {theme["table_header_bg"]} !important;
        color: {theme["table_header_text"]} !important;
        border-color: {theme["border"]} !important;
    }}
    .st-emotion-cache-znj1k1,
    .st-emotion-cache-6ms01g {{
        color: {theme["text"]} !important;
        background: transparent !important;
    }}
    h2 {{
        font-size: 1.28rem !important;
        line-height: 1.35 !important;
    }}
    h3 {{
        font-size: 1.08rem !important;
        line-height: 1.35 !important;
    }}
    p, li, label, .stMarkdown, [data-testid="stCaptionContainer"] {{
        font-size: 0.95rem;
        line-height: 1.65;
    }}
    .main-title {{
        font-size: 1.85rem;
        line-height: 1.2;
        font-weight: 750;
        margin-bottom: 0.45rem;
        color: {theme["text"]};
    }}
    .brand-row {{
        display: flex;
        align-items: center;
        gap: 0.86rem;
        margin-bottom: 0.42rem;
    }}
    .brand-mark {{
        width: 52px;
        height: 52px;
        border-radius: 10px;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        background: {theme["button_primary_bg"]};
        color: {theme["button_primary_text"]};
        box-shadow: 0 12px 28px rgba(15, 23, 42, 0.18);
        flex: 0 0 auto;
    }}
    .brand-mark svg {{
        width: 38px;
        height: 38px;
        stroke: currentColor;
    }}
    .subtitle {{
        color: {theme["muted"]};
        margin-bottom: 1.15rem;
        line-height: 1.6;
        font-size: 0.98rem;
        max-width: 820px;
    }}
    .note {{
        border-left: 4px solid {theme["accent"]};
        background: {theme["note_bg"]};
        padding: 0.8rem 1rem;
        margin: 0.8rem 0 1.2rem;
        color: {theme["text"]};
        font-size: 0.94rem;
        line-height: 1.65;
    }}
    .metric-explain {{
        background: {theme["panel"]};
        border: 1px solid {theme["border"]};
        border-radius: 8px;
        padding: 0.85rem 1rem;
        margin: 0.85rem 0 1.1rem;
        color: {theme["text"]};
        font-size: 0.93rem;
        line-height: 1.65;
    }}
    .metric-explain strong {{
        color: {theme["accent"]};
    }}
    .equation-card code {{
        display: block;
        margin-top: 0.35rem;
        font-size: 1.05rem;
        line-height: 1.8;
        color: {theme["text"]};
        white-space: normal;
        word-break: break-word;
    }}
    .plain-reading {{
        background: {theme["panel"]};
        border: 1px solid {theme["border"]};
        border-left: 4px solid {theme["accent"]};
        border-radius: 8px;
        padding: 1rem 1.1rem;
        margin: 1rem 0 1.25rem;
        color: {theme["text"]};
        font-size: 0.95rem;
        line-height: 1.7;
    }}
    .plain-reading h4 {{
        margin: 0 0 0.45rem 0;
        font-size: 1.02rem;
        color: {theme["text"]};
    }}
    .plain-reading p {{
        margin: 0.35rem 0;
        color: {theme["text"]};
    }}
    .source-card,
    .crisp-card {{
        background: {theme["panel"]};
        border: 1px solid {theme["border"]};
        border-radius: 8px;
        padding: 1rem 1.1rem;
        margin: 0.8rem 0 1rem;
        color: {theme["text"]};
        line-height: 1.65;
    }}
    .source-card h4,
    .crisp-card h4 {{
        margin: 0 0 0.5rem;
        color: {theme["text"]};
        font-size: 1.03rem;
    }}
    .source-card a,
    .crisp-card a {{
        color: {theme["accent"]} !important;
        font-weight: 650;
    }}
    .learning-step {{
        background: {theme["panel_alt"]};
        border: 1px solid {theme["border"]};
        border-radius: 8px;
        padding: 0.9rem 1rem;
        margin: 0.6rem 0;
        color: {theme["text"]};
    }}
    .learning-step strong {{
        color: {theme["accent"]};
    }}
    .download-panel {{
        background: {theme["panel_alt"]};
        border: 1px solid {theme["border"]};
        border-radius: 8px;
        padding: 0.95rem 1rem;
        margin: 0.4rem 0 0.8rem;
        color: {theme["text"]};
    }}
    .download-panel .file-name {{
        font-weight: 750;
        color: {theme["text"]};
    }}
    .download-panel .file-meta {{
        color: {theme["muted"]};
        font-size: 0.9rem;
    }}
    .js-plotly-plot .modebar {{
        display: none !important;
    }}
    .js-plotly-plot .main-svg text {{
        fill: {theme["plot_text"]} !important;
        opacity: 1 !important;
    }}
    .js-plotly-plot .legendtext,
    .js-plotly-plot .cbtitle,
    .js-plotly-plot .cbaxis text {{
        fill: {theme["plot_muted"]} !important;
        opacity: 1 !important;
    }}
    {sidebar_visibility_css}
    </style>
    """,
    unsafe_allow_html=True,
)

t = TEXT[st.session_state.locale].get


sidebar_toggle_label = "☰" if st.session_state.sidebar_collapsed else "‹"
if st.button(sidebar_toggle_label, key="sidebar_state_toggle", help=t("sidebar_toggle")):
    st.session_state.sidebar_collapsed = not st.session_state.sidebar_collapsed
    st.rerun()

if not st.session_state.sidebar_collapsed:
    with st.sidebar:
        col_lang, col_theme = st.columns(2)
        with col_lang:
            lang_label = t("to_en") if st.session_state.locale == "zh" else t("to_zh")
            if st.button(lang_label, width="stretch", key="lang_toggle", help=t("language")):
                st.session_state.locale = "en" if st.session_state.locale == "zh" else "zh"
                st.rerun()
        with col_theme:
            theme_label = t("theme_dark_btn") if st.session_state.theme == "light" else t("theme_light_btn")
            if st.button(theme_label, width="stretch", key="theme_toggle", help=t("theme")):
                st.session_state.theme = "dark" if st.session_state.theme == "light" else "light"
                st.rerun()


MODEL_OPTIONS = {
    "ols": {"label_key": "model_ols", "reading_key": "model_ols_reading"},
    "ridge": {"label_key": "model_ridge", "reading_key": "model_ridge_reading"},
    "lasso": {"label_key": "model_lasso", "reading_key": "model_lasso_reading"},
    "elasticnet": {"label_key": "model_elasticnet", "reading_key": "model_elasticnet_reading"},
}

NOISE_PRESETS = {
    "very_low": {"value": 100.0, "label": {"en": "Very low - 100", "zh": "很低 - 100"}},
    "low": {"value": 1000.0, "label": {"en": "Low - 1,000", "zh": "低 - 1,000"}},
    "medium": {"value": 10000.0, "label": {"en": "Medium - 10,000", "zh": "中 - 10,000"}},
    "high": {"value": 50000.0, "label": {"en": "High - 50,000", "zh": "高 - 50,000"}},
    "very_high": {"value": 100000.0, "label": {"en": "Very high - 100,000", "zh": "很高 - 100,000"}},
}

SEED_PRESETS = {
    "zero": {"value": 0, "label": {"en": "Fixed start - 0", "zh": "固定起點 - 0"}},
    "small": {"value": 7, "label": {"en": "Small test - 7", "zh": "小型測試 - 7"}},
    "demo": {"value": 42, "label": {"en": "Common demo - 42", "zh": "常用示範 - 42"}},
    "year": {"value": 2024, "label": {"en": "Year marker - 2024", "zh": "年度示範 - 2024"}},
    "alternate": {"value": 1001, "label": {"en": "Alternate sample - 1001", "zh": "替代樣本 - 1001"}},
}


def model_label(model_key: str) -> str:
    return t(MODEL_OPTIONS.get(model_key, MODEL_OPTIONS["ols"])["label_key"])


def model_reading(model_key: str) -> str:
    return t(MODEL_OPTIONS.get(model_key, MODEL_OPTIONS["ols"])["reading_key"])


def preset_label(presets: dict, preset_key: str) -> str:
    return presets[preset_key]["label"][st.session_state.locale]


def sync_noise_preset() -> None:
    st.session_state.synthetic_var = NOISE_PRESETS[st.session_state.synthetic_var_preset]["value"]


def sync_seed_preset() -> None:
    st.session_state.synthetic_seed = SEED_PRESETS[st.session_state.synthetic_seed_preset]["value"]


def build_sklearn_model(model_key: str):
    if model_key == "ridge" and Ridge is not None:
        return make_pipeline(StandardScaler(), Ridge(alpha=1.0)), model_label("ridge")
    if model_key == "lasso" and Lasso is not None:
        return make_pipeline(StandardScaler(), Lasso(alpha=0.03, max_iter=10000)), model_label("lasso")
    if model_key == "elasticnet" and ElasticNet is not None:
        return make_pipeline(
            StandardScaler(),
            ElasticNet(alpha=0.03, l1_ratio=0.45, max_iter=10000),
        ), model_label("elasticnet")
    return LinearRegression(), model_label("ols")


def extract_coefficients(model, model_key: str) -> tuple[np.ndarray, float]:
    if model_key in {"ridge", "lasso", "elasticnet"} and hasattr(model, "named_steps"):
        estimator = model.named_steps[model_key]
        return estimator.coef_, float(estimator.intercept_)
    return model.coef_, float(model.intercept_)


def fit_linear_model(feature_df: pd.DataFrame, target: pd.Series, model_key: str = "ols"):
    x = feature_df.to_numpy(dtype=float)
    y = target.to_numpy(dtype=float)

    if LinearRegression is not None:
        model, model_name = build_sklearn_model(model_key)
        model.fit(x, y)
        y_pred = model.predict(x)
        coefficients, intercept = extract_coefficients(model, model_key)
    else:
        x_design = np.column_stack([np.ones(len(x)), x])
        params, *_ = np.linalg.lstsq(x_design, y, rcond=None)
        intercept = float(params[0])
        coefficients = params[1:]
        y_pred = x_design @ params
        model = None
        model_name = "NumPy least-squares fallback"

    residuals = y - y_pred
    return {
        "model_key": model_key if LinearRegression is not None else "ols",
        "model_name": model_name,
        "intercept": intercept,
        "coefficients": coefficients,
        "predictions": y_pred,
        "residuals": residuals,
        "r2": safe_r2(y, y_pred),
        "mse": float(np.mean((y - y_pred) ** 2)),
        "rmse": float(np.sqrt(np.mean((y - y_pred) ** 2))),
        "mae": float(np.mean(np.abs(y - y_pred))),
        "estimator": model,
    }


def predict_linear_model(result: dict, feature_df: pd.DataFrame) -> np.ndarray:
    x = feature_df.to_numpy(dtype=float)
    estimator = result.get("estimator")
    if estimator is not None:
        return estimator.predict(x)
    x_design = np.column_stack([np.ones(len(x)), x])
    params = np.array([result["intercept"], *result["coefficients"]], dtype=float)
    return x_design @ params


def theme_table(df: pd.DataFrame):
    return (
        df.style.set_properties(
            **{
                "background-color": theme["panel"],
                "color": theme["text"],
                "border-color": theme["border"],
            }
        )
        .set_table_styles(
            [
                {
                    "selector": "thead th",
                    "props": [
                        ("background-color", theme["table_header_bg"]),
                        ("color", theme["table_header_text"]),
                        ("border-color", theme["border"]),
                        ("font-weight", "700"),
                    ],
                },
                {
                    "selector": "tbody th",
                    "props": [
                        ("background-color", theme["panel"]),
                        ("color", theme["text"]),
                        ("border-color", theme["border"]),
                    ],
                },
                {
                    "selector": "td",
                    "props": [
                        ("background-color", theme["panel"]),
                        ("color", theme["text"]),
                        ("border-color", theme["border"]),
                    ],
                },
            ]
        )
    )


def safe_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) < 2 or np.allclose(y_true, y_true[0]):
        return 0.0
    if LinearRegression is not None:
        return float(r2_score(y_true, y_pred))
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    return 1 - ss_res / ss_tot if ss_tot else 0.0


def apply_plot_style(fig: go.Figure, height: int, title: str | None = None) -> go.Figure:
    layout = {
        "height": height,
        "template": plot_template,
        "paper_bgcolor": theme["plot_bg"],
        "plot_bgcolor": theme["plot_bg"],
        "font": {"color": theme["plot_text"], "size": 13},
        "legend": {
            "font": {"color": theme["plot_text"], "size": 12},
            "bgcolor": "rgba(0,0,0,0)",
            "bordercolor": "rgba(0,0,0,0)",
        },
        "hoverlabel": {
            "bgcolor": theme["panel"],
            "bordercolor": theme["border"],
            "font": {"color": theme["text"], "size": 13},
        },
        "coloraxis_colorbar": {
            "title": {"font": {"color": theme["plot_text"], "size": 13}},
            "tickfont": {"color": theme["plot_text"], "size": 12},
        },
        "margin": {"l": 72, "r": 44, "t": 60 if title else 36, "b": 64},
    }
    if title:
        layout["title"] = {"text": title}
        layout["title_font"] = {"color": theme["plot_text"], "size": 18}
    else:
        layout["title_text"] = ""
    fig.update_layout(**layout)
    fig.update_xaxes(
        title_font={"color": theme["plot_text"], "size": 14},
        tickfont={"color": theme["plot_text"], "size": 12},
        gridcolor=theme["plot_grid"],
        zerolinecolor=theme["plot_grid"],
        linecolor=theme["plot_axis"],
        showline=True,
    )
    fig.update_yaxes(
        title_font={"color": theme["plot_text"], "size": 14},
        tickfont={"color": theme["plot_text"], "size": 12},
        gridcolor=theme["plot_grid"],
        zerolinecolor=theme["plot_grid"],
        linecolor=theme["plot_axis"],
        showline=True,
    )
    return fig


def normalize_aqi_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {
        "SiteName": "sitename",
        "site_name": "sitename",
        "County": "county",
        "AQI": "aqi",
        "PM2.5": "pm25",
        "PM2.5_AVG": "pm25_avg",
        "PM10": "pm10",
        "PM10_AVG": "pm10_avg",
        "SO2": "so2",
        "CO": "co",
        "O3": "o3",
        "NO2": "no2",
        "NOx": "nox",
        "NO": "no",
        "WIND_SPEED": "wind_speed",
        "WindSpeed": "wind_speed",
        "windspeed": "wind_speed",
        "WIND_DIREC": "wind_direc",
        "WindDirec": "wind_direc",
        "winddirec": "wind_direc",
        "Longitude": "longitude",
        "Latitude": "latitude",
        "publishtime": "publish_time",
        "PublishTime": "publish_time",
        "datacreationdate": "publish_time",
        "DataCreationDate": "publish_time",
        "pm2.5": "pm25",
        "pm2.5_avg": "pm25_avg",
    }
    normalized = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns}).copy()

    text_columns = {"sitename", "county", "pollutant", "status", "publish_time"}
    for col in normalized.columns:
        if col not in text_columns:
            converted = pd.to_numeric(normalized[col], errors="coerce")
            if converted.notna().sum() > 0:
                normalized[col] = converted
    return normalized


def get_numeric_columns(df: pd.DataFrame) -> list[str]:
    return [
        col
        for col in df.columns
        if pd.api.types.is_numeric_dtype(df[col]) and df[col].notna().sum() >= 3
    ]


def show_metrics(result: dict) -> None:
    cols = st.columns(4)
    cols[0].metric(t("r2"), f"{result['r2']:.3f}")
    cols[1].metric(t("rmse"), f"{result['rmse']:.2f}")
    cols[2].metric(t("mae"), f"{result['mae']:.2f}")
    cols[3].metric(t("model"), result["model_name"])


def metric_explanation(result: dict) -> None:
    if result["r2"] >= 0.8:
        r2_text = t("metric_r2_strong")
    elif result["r2"] >= 0.5:
        r2_text = t("metric_r2_ok")
    else:
        r2_text = t("metric_r2_weak")

    st.markdown(
        f"""
        <div class="metric-explain">
            <strong>{t("metric_status_title")}：</strong>{r2_text}<br>
            {t("metric_error_note")}<br>
            {t("metric_outlier_note")}
        </div>
        """,
        unsafe_allow_html=True,
    )


def model_profile(result: dict) -> None:
    note = t("coefficient_note") if result["model_key"] != "ols" else ""
    st.markdown(
        f"""
        <div class="metric-explain">
            <strong>{t("model_profile_title")}：</strong>{model_reading(result["model_key"])}
            {f"<br>{note}" if note else ""}
        </div>
        """,
        unsafe_allow_html=True,
    )


def equation_text(result: dict, feature_names: list[str], target_name: str = "y") -> str:
    terms = [f"{result['intercept']:.4f}"]
    for feature, coefficient in zip(feature_names, result["coefficients"]):
        sign = "+" if coefficient >= 0 else "-"
        terms.append(f"{sign} {abs(coefficient):.4f} * {feature}")
    return f"{target_name} = " + " ".join(terms)


def show_model_equation(result: dict, feature_names: list[str], target_name: str = "y") -> None:
    st.markdown(
        f"""
        <div class="metric-explain equation-card">
            <strong>{t("equation")}:</strong><br>
            <code>{equation_text(result, feature_names, target_name)}</code><br>
            {t("equation_note") if result["model_key"] != "ols" else ""}
        </div>
        """,
        unsafe_allow_html=True,
    )


def residual_scope_note() -> None:
    st.markdown(
        f"""
        <div class="metric-explain">
            <strong>{t("top_outliers")}:</strong> {t("outlier_plain_note")}<br>
            {t("outlier_scope_note")}
        </div>
        """,
        unsafe_allow_html=True,
    )


def compare_linear_models(feature_df: pd.DataFrame, target: pd.Series) -> pd.DataFrame:
    rows = []
    available_models = ["ols"] if LinearRegression is None else list(MODEL_OPTIONS.keys())
    for model_key in available_models:
        result = fit_linear_model(feature_df, target, model_key)
        rows.append(
            {
                t("model"): result["model_name"],
                t("r2"): round(result["r2"], 3),
                t("rmse"): round(result["rmse"], 2),
                t("mae"): round(result["mae"], 2),
            }
        )
    return pd.DataFrame(rows)


def show_dataframe(df: pd.DataFrame, **kwargs) -> None:
    hide_index = bool(kwargs.pop("hide_index", False))
    kwargs.pop("width", None)
    styled = theme_table(df)
    if hide_index:
        styled = styled.hide(axis="index")
    st.table(styled)


def data_label() -> str:
    if AQI_DATA_IS_PRIMARY:
        return t("recent_sample")
    if AQI_DATA_IS_LOCAL_HISTORICAL:
        return t("local_historical_sample")
    return t("no_active_data")


def dataset_status_html(row_count: int, filtered_count: int | None = None) -> str:
    filtered = "" if filtered_count is None else f"<p><strong>{t('complete_rows')}:</strong> {filtered_count:,}</p>"
    scope = "Taichung City historical records" if AQI_DATA_IS_LOCAL_HISTORICAL else f"{t('region_central')} / Taichung, Changhua, Nantou"
    return f"""
    <div class="source-card">
        <h4>{t("active_data")}</h4>
        <p><strong>{t("source_run")}:</strong> {data_label()}</p>
        <p><strong>{t("data_rows")}:</strong> {row_count:,}</p>
        {filtered}
        <p><strong>{t("data_scope")}:</strong> {scope}</p>
    </div>
    """


def region_label(region_key: str) -> str:
    return t(f"region_{region_key}")


def filter_by_region(df: pd.DataFrame, region_key: str) -> pd.DataFrame:
    counties = REGION_FILTERS.get(region_key, REGION_FILTERS["central"])
    if "county" not in df.columns:
        return df.copy()
    return df[df["county"].isin(counties)].copy()


def page_navigation() -> str:
    options = ["lab", "aqi", "data", "report"]
    labels = {
        "lab": t("nav_lab"),
        "aqi": t("nav_aqi"),
        "report": t("nav_report"),
        "data": t("nav_data"),
    }
    icons = {
        "lab": ":material/science:",
        "aqi": ":material/air:",
        "report": ":material/description:",
        "data": ":material/dataset:",
    }
    cols = st.columns(len(options), gap="small")
    for col, option in zip(cols, options):
        with col:
            if st.button(
                labels[option],
                width="stretch",
                key=f"nav_{option}",
                type="secondary",
                icon=icons[option],
            ):
                st.session_state.active_page = option
                st.rerun()
    return st.session_state.active_page


def report_markdown(locale: str | None = None) -> str:
    active_locale = locale or st.session_state.locale
    active_file = SAMPLE_AQI_PATH.name if SAMPLE_AQI_PATH else "none"
    zh_scope = (
        "台中、彰化、南投中部 AQI 記錄；目前樣本數落在 2 萬到 10 萬筆的目標範圍內。"
        if AQI_DATA_IS_PRIMARY
        else "目前使用 2016-2026 的台中歷史樣本，總筆數足夠，但 2018 以後只剩 966 筆，未完全符合近期中部資料條件。"
        if AQI_DATA_IS_LOCAL_HISTORICAL
        else "目前沒有可用的大型 AQI CSV；微型 demo fallback 已移除，應更換公開資料或研究主題。"
    )
    en_scope = (
        "central Taiwan AQI records for Taichung, Changhua and Nantou; the current row count fits the requested 20,000 to 100,000 row range."
        if AQI_DATA_IS_PRIMARY
        else "the app is using a 2016-2026 Taichung historical sample. The total row count is sufficient, but filtering to 2018 or later leaves 966 rows, so it does not fully satisfy the recent central-Taiwan condition."
        if AQI_DATA_IS_LOCAL_HISTORICAL
        else "no larger AQI CSV is available. The tiny demo fallback has been removed, so the project should switch to another public dataset or research topic."
    )
    if active_locale == "zh":
        return f"""# CRISP-DM 報告：Linear Regression Practice

## 1. 商業理解

本專案用空氣品質指標 (Air Quality Index, AQI) 與污染物欄位建立線性回歸 baseline。目標不是取代官方預報，而是協助使用者快速看出哪些觀測不符合一般污染物規律，值得進一步檢查。

## 2. 資料理解

目前 app 實際使用資料：{data_label()}

實際檔案：`data/{active_file}`

樣本數：{SAMPLE_AQI_ROWS:,} 筆

資料範圍：{zh_scope}

主要欄位：測站、縣市、發布時間、空氣品質指標 (Air Quality Index, AQI)、PM2.5、PM10、O3、NO2、CO、SO2、風速、風向、經緯度。

官方來源：{OFFICIAL_SOURCE_NAME}

MOENV URL：{OFFICIAL_SOURCE_URL}

即時資料參考：{REALTIME_SOURCE_NAME} - {REALTIME_SOURCE_URL}

Kaggle 歷史參考：{DEFAULT_SOURCE_NAME} - {DEFAULT_SOURCE_URL}

Lab 來源邊界：Lab 不使用上述任何 CSV 或公開資料。Lab 由 `app.py` 依照 `y = a*x + b + noise` 在記憶體產生合成資料，左側參數就是資料生成條件。

## 3. 資料準備

- 正規化 AQI、污染物、風速與時間欄位名稱。
- 將可建模欄位轉成數值。
- 依 sidebar 選擇中部、台中、彰化或南投。
- 移除目標欄位與特徵欄位缺值資料列。
- 保留資料來源、樣本數與篩選範圍，避免把 demo 資料誤解成正式模型資料。

## 4. 模型建立

使用 OLS、Ridge、Lasso 與 ElasticNet 四種線性模型。OLS 作為最容易解釋的 baseline；Ridge 適合污染物高度相關時降低係數波動；Lasso 可用於特徵篩選；ElasticNet 則折衷穩定性與簡化。

## 5. 模型評估

使用 R-squared、RMSE、MAE 與殘差排序。R-squared 用來看模型解釋力；RMSE 與 MAE 用來看平均誤差；絕對殘差最大的資料列，是最值得人工檢查的異常觀測。

## 6. 部署與使用

結果包裝成雙語 Streamlit app，支援明暗主題、動態 sidebar、資料下載、白話解讀與依目前語言輸出的 CRISP-DM 報告。外部 CSV 上傳在尚未定義 schema 驗證前先移除。
"""

    return f"""# CRISP-DM Report: Linear Regression Practice

## 1. Business Understanding

This project uses Air Quality Index (AQI) and pollutant fields to build a linear-regression baseline. The goal is not to replace official forecasts, but to help users notice observations that do not follow the usual pollutant pattern.

## 2. Data Understanding

Current app data: {data_label()}

Actual file: `data/{active_file}`

Rows: {SAMPLE_AQI_ROWS:,}

Scope: {en_scope}

Main fields: station, county, publish time, Air Quality Index (AQI), PM2.5, PM10, O3, NO2, CO, SO2, wind speed, wind direction, longitude and latitude.

Official source: {OFFICIAL_SOURCE_NAME}

MOENV URL: {OFFICIAL_SOURCE_URL}

Realtime data reference: {REALTIME_SOURCE_NAME} - {REALTIME_SOURCE_URL}

Kaggle historical reference: {DEFAULT_SOURCE_NAME} - {DEFAULT_SOURCE_URL}

Lab source boundary: Lab does not use any of the CSV or public sources above. It generates synthetic data in memory from `y = a*x + b + noise` inside `app.py`; the sidebar parameters are the data-generation conditions.

## 3. Data Preparation

- Normalize AQI, pollutant, wind and time column names.
- Convert model-ready fields into numeric values.
- Filter Central Taiwan, Taichung, Changhua or Nantou through the sidebar.
- Drop rows missing the selected target or feature fields.
- Keep source, row count and scope visible so demo data is not confused with a formal dataset.

## 4. Modeling

The app uses four linear model variants: OLS, Ridge, Lasso and ElasticNet. OLS is the clearest baseline; Ridge helps when pollutant fields are correlated; Lasso helps with feature screening; ElasticNet balances stability and simplification.

## 5. Evaluation

The app reports R-squared, RMSE, MAE and residual ranking. Large absolute residuals are the observations most worth checking manually.

## 6. Deployment

The result is packaged as a bilingual Streamlit app with light/dark themes, dynamic sidebar controls, data download, plain-language explanations and a CRISP-DM report generated in the currently selected language. External upload is intentionally removed until a schema validator is defined.
"""


def source_download_panel(panel_id: str, include_report: bool = True) -> None:
    if not AQI_DATA_AVAILABLE:
        st.error(t("no_active_data"))
        return
    sample_bytes = SAMPLE_AQI_PATH.read_bytes()
    report_text = report_markdown(st.session_state.locale)
    report_file_name = f"crisp_dm_report_{st.session_state.locale}.md"
    sample_file_name = SAMPLE_AQI_PATH.name
    limitation_note = ""
    if AQI_DATA_IS_LOCAL_HISTORICAL:
        limitation_note = f"<p>{t('local_historical_warning')}</p>"

    st.markdown(
        f"""
        <div class="source-card">
            <h4>{t("source_panel_title")}</h4>
            <p><strong>{t("source_run")}:</strong> {data_label()}</p>
            <p><strong>CSV:</strong> data/{sample_file_name}</p>
            <p>{t("source_panel_body")}</p>
            <p>{t("sample_scope_note").format(rows=SAMPLE_AQI_ROWS)}</p>
            {limitation_note}
            <p><strong>{DEFAULT_SOURCE_NAME}</strong><br>
            <a href="{DEFAULT_SOURCE_URL}" target="_blank">{DEFAULT_SOURCE_URL}</a></p>
            <p><strong>{OFFICIAL_SOURCE_NAME}</strong><br>
            <a href="{OFFICIAL_SOURCE_URL}" target="_blank">{OFFICIAL_SOURCE_URL}</a></p>
            <p><strong>{REALTIME_SOURCE_NAME}</strong><br>
            <a href="{REALTIME_SOURCE_URL}" target="_blank">{REALTIME_SOURCE_URL}</a></p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if include_report:
        col_data, col_report = st.columns(2)
    else:
        col_data = st.container()
        col_report = None
    file_meta = (
        f"{data_label()} · {SAMPLE_AQI_ROWS:,} 筆 · AQI 頁目前使用資料"
        if st.session_state.locale == "zh"
        else f"{data_label()} · {SAMPLE_AQI_ROWS:,} rows · AQI page active dataset"
    )
    with col_data:
        st.markdown(
            f"""
            <div class="download-panel">
                <div class="file-name">data/{sample_file_name}</div>
                <div class="file-meta">{file_meta}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.download_button(
            t("download_data"),
            data=sample_bytes,
            file_name=sample_file_name,
            mime="text/csv",
            width="stretch",
            key=f"download_sample_{panel_id}",
        )
    if col_report is not None:
        with col_report:
            st.download_button(
                t("download_report"),
                data=report_text.encode("utf-8"),
                file_name=report_file_name,
                mime="text/markdown",
                width="stretch",
                key=f"download_report_{panel_id}",
            )


def crisp_report_tab() -> None:
    report_text = report_markdown(st.session_state.locale)
    st.subheader(t("crisp_title"))
    st.markdown(f'<div class="note">{t("crisp_intro")}</div>', unsafe_allow_html=True)
    st.markdown(report_text)
    st.download_button(
        t("download_report"),
        data=report_text.encode("utf-8"),
        file_name=f"crisp_dm_report_{st.session_state.locale}.md",
        mime="text/markdown",
        width="stretch",
        key="download_report_crisp_only",
    )


def plain_aqi_reading(model_df: pd.DataFrame, target_col: str) -> None:
    avg_target = float(model_df[target_col].mean())
    max_residual = float(model_df["abs_residual"].max())
    st.markdown(
        f"""
        <div class="plain-reading">
            <h4>{t("chart_plain_title")}</h4>
            <p>{t("chart_plain_aqi")}</p>
            <p>{t("chart_plain_stats").format(target=target_col, avg=avg_target, residual=max_residual)}</p>
            <p>{t("chart_plain_caution")}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def beginner_learning_panel() -> None:
    if st.session_state.locale == "zh":
        title = "線性迴歸小白學習路線"
        steps = [
            ("先看 x 和 y", "x 是輸入線索，y 是模型想猜的答案。Lab 的 y 由 app.py 依照 y = a*x + b + noise 即時合成。"),
            ("一次只改一個參數", "樣本數 n 影響穩定性；斜率 a 影響 x 和 y 的關係強度；截距 b 會把整條線往上或往下移。"),
            ("理解雜訊與種子", "雜訊變異數越高，點越散，模型越難猜準；隨機種子讓同一組設定可以重現同一批合成資料。"),
            ("看殘差", "殘差是實際值和預測值的差距。殘差前 10 名不是一定錯，而是最值得人工檢查。"),
            ("再切到 AQI", "AQI 頁使用真實空氣品質 CSV。Lab 的斜率、截距、雜訊和種子不會改動 AQI 資料。"),
        ]
    else:
        title = "Beginner learning path"
        steps = [
            ("Start with x and y", "x is the input clue, and y is the answer the model tries to estimate. Lab y is generated in app.py from y = a*x + b + noise."),
            ("Change one parameter at a time", "Sample size n affects stability; slope a changes the strength of the x/y pattern; intercept b moves the line up or down."),
            ("Read noise and seed", "Higher noise variance spreads the points and makes prediction harder. The random seed makes the same settings reproducible."),
            ("Check residuals", "A residual is actual value minus predicted value. The top 10 residuals are not automatically wrong; they are the first records to inspect."),
            ("Then move to AQI", "The AQI page uses the real air-quality CSV. Lab slope, intercept, noise and seed do not modify AQI records."),
        ]
    st.markdown(f"#### {title}")
    for heading, body in steps:
        st.markdown(
            f"""
            <div class="learning-step">
                <strong>{heading}</strong><br>
                {body}
            </div>
            """,
            unsafe_allow_html=True,
        )


def synthetic_lab() -> None:
    st.subheader(t("synthetic_title"))
    st.markdown(
        f'<div class="note">{t("synthetic_note")}</div>',
        unsafe_allow_html=True,
    )

    if "synthetic_var_preset" not in st.session_state:
        st.session_state.synthetic_var_preset = "medium"
    if "synthetic_seed_preset" not in st.session_state:
        st.session_state.synthetic_seed_preset = "demo"
    if "synthetic_var" not in st.session_state:
        st.session_state.synthetic_var = NOISE_PRESETS[st.session_state.synthetic_var_preset]["value"]
    if "synthetic_seed" not in st.session_state:
        st.session_state.synthetic_seed = SEED_PRESETS[st.session_state.synthetic_seed_preset]["value"]
    available_models = ["ols"] if LinearRegression is None else list(MODEL_OPTIONS.keys())

    if not st.session_state.sidebar_collapsed:
        with st.sidebar:
            st.markdown(f"### {t('synthetic_params')}")
            model_key = st.selectbox(
                t("model_choice"),
                available_models,
                format_func=model_label,
                key="synthetic_model_choice",
            )
            n = st.slider(t("sample_size"), 50, 1000, 300, 50, key="synthetic_n")
            a = st.slider(t("slope"), -50.0, 50.0, 8.0, 0.5, key="synthetic_a")
            b = st.slider(t("intercept"), -100.0, 100.0, 40.0, 1.0, key="synthetic_b")
            st.selectbox(
                t("variance_preset"),
                list(NOISE_PRESETS.keys()),
                format_func=lambda key: preset_label(NOISE_PRESETS, key),
                key="synthetic_var_preset",
                on_change=sync_noise_preset,
            )
            var_kwargs = {
                "label": t("variance"),
                "min_value": 0.0,
                "max_value": 100000.0,
                "step": 500.0,
                "key": "synthetic_var",
            }
            if "synthetic_var" not in st.session_state:
                var_kwargs["value"] = 10000.0
            var = st.number_input(**var_kwargs)
            st.selectbox(
                t("seed_preset"),
                list(SEED_PRESETS.keys()),
                format_func=lambda key: preset_label(SEED_PRESETS, key),
                key="synthetic_seed_preset",
                on_change=sync_seed_preset,
            )
            seed_kwargs = {
                "label": t("seed"),
                "min_value": 0,
                "max_value": 999999,
                "step": 1,
                "key": "synthetic_seed",
            }
            if "synthetic_seed" not in st.session_state:
                seed_kwargs["value"] = 42
            seed = st.number_input(**seed_kwargs)
    else:
        model_key = st.session_state.get("synthetic_model_choice", "ols")
        n = st.session_state.get("synthetic_n", 300)
        a = st.session_state.get("synthetic_a", 8.0)
        b = st.session_state.get("synthetic_b", 40.0)
        var = st.session_state.get("synthetic_var", 10000.0)
        seed = st.session_state.get("synthetic_seed", 42)

    rng = np.random.default_rng(seed)
    x = rng.uniform(-100, 100, n)
    noise = rng.normal(0, np.sqrt(var), n)
    y = a * x + b + noise
    df = pd.DataFrame({"x": x, "y": y})

    result = fit_linear_model(df[["x"]], df["y"], model_key)
    df["predicted_y"] = result["predictions"]
    df["residual"] = result["residuals"]
    df["abs_residual"] = np.abs(result["residuals"])
    df["rank"] = df["abs_residual"].rank(method="first", ascending=False).astype(int)
    df["is_outlier"] = df["rank"] <= 10

    x_line = np.linspace(df["x"].min(), df["x"].max(), 200)
    fitted_line = predict_linear_model(result, pd.DataFrame({"x": x_line}))
    true_line = a * x_line + b

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["x"], y=df["y"], mode="markers", name=t("generated_data")))
    fig.add_trace(
        go.Scatter(
            x=df.loc[df["is_outlier"], "x"],
            y=df.loc[df["is_outlier"], "y"],
            mode="markers",
            name=t("outliers"),
            marker={"size": 12, "color": "#ef4444", "symbol": "circle-open", "line": {"width": 2}},
        )
    )
    fig.add_trace(go.Scatter(x=x_line, y=true_line, mode="lines", name=t("true_line"), line={"dash": "dash"}))
    fig.add_trace(go.Scatter(x=x_line, y=fitted_line, mode="lines", name=t("regression_line")))
    apply_plot_style(fig, height=500)
    fig.update_layout(xaxis_title="x", yaxis_title="y", legend_orientation="h")
    st.plotly_chart(fig, width="stretch", config=plot_config)

    show_metrics(result)
    metric_explanation(result)
    model_profile(result)
    show_model_equation(result, ["x"], "y")

    st.markdown(f"#### {t('outliers')}")
    residual_scope_note()
    show_dataframe(
        df.sort_values("abs_residual", ascending=False)
        .head(10)[["rank", "x", "y", "predicted_y", "residual", "abs_residual"]],
        width="stretch",
    )
    beginner_learning_panel()
    st.markdown(
        f"""
        <div class="source-card">
            <h4>{t("synthetic_source_title")}</h4>
            <p>{t("synthetic_source_body")}</p>
            <p>{t("synthetic_purpose")}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def aqi_case() -> None:
    st.subheader(t("aqi_title"))
    st.markdown(
        f'<div class="note">{t("aqi_note")}</div>',
        unsafe_allow_html=True,
    )
    if not AQI_DATA_AVAILABLE:
        st.error(t("no_active_data"))
        return

    raw_df = pd.read_csv(SAMPLE_AQI_PATH)
    df = normalize_aqi_columns(raw_df)
    full_row_count = len(df)

    numeric_cols = get_numeric_columns(df)
    if len(numeric_cols) < 2:
        st.error(t("too_few_cols"))
        show_dataframe(df.head(20), width="stretch")
        return

    default_target = "aqi" if "aqi" in numeric_cols else numeric_cols[-1]
    feature_options = [col for col in numeric_cols if col != default_target]
    default_features = [
        col
        for col in ["pm25", "pm25_avg", "pm10", "pm10_avg", "o3", "no2", "co", "so2", "wind_speed"]
        if col in feature_options
    ]
    if not default_features:
        default_features = feature_options[: min(5, len(feature_options))]

    available_models = ["ols"] if LinearRegression is None else list(MODEL_OPTIONS.keys())
    if not st.session_state.sidebar_collapsed:
        with st.sidebar:
            st.markdown(f"### {t('aqi_params')}")
            region_key = st.selectbox(
                t("region"),
                list(REGION_FILTERS.keys()),
                format_func=region_label,
                key="aqi_region",
            )
            target_col = st.selectbox(
                t("target"),
                numeric_cols,
                index=numeric_cols.index(default_target),
                key="aqi_target",
            )
            feature_options = [col for col in numeric_cols if col != target_col]
            selected_features = st.multiselect(
                t("features"),
                feature_options,
                default=[col for col in default_features if col in feature_options],
                key="aqi_features",
            )
            model_key = st.selectbox(
                t("model_choice"),
                available_models,
                format_func=model_label,
                key="aqi_model_choice",
            )
    else:
        region_key = st.session_state.get("aqi_region", "central")
        target_col = st.session_state.get("aqi_target", default_target)
        feature_options = [col for col in numeric_cols if col != target_col]
        selected_features = st.session_state.get(
            "aqi_features",
            [col for col in default_features if col in feature_options],
        )
        model_key = st.session_state.get("aqi_model_choice", "ols")

    df = filter_by_region(df, region_key)

    if not selected_features:
        st.warning(t("select_feature"))
        return

    model_df = df[[target_col, *selected_features]].dropna().copy()
    if len(model_df) < 10:
        st.warning(t("too_few_rows"))
        return

    result = fit_linear_model(model_df[selected_features], model_df[target_col], model_key)
    model_df["predicted"] = result["predictions"]
    model_df["residual"] = result["residuals"]
    model_df["abs_residual"] = np.abs(result["residuals"])
    model_df["rank"] = model_df["abs_residual"].rank(method="first", ascending=False).astype(int)

    display_df = df.loc[model_df.index].copy()
    for col in ["predicted", "residual", "abs_residual", "rank"]:
        display_df[col] = model_df[col]

    coefficient_df = pd.DataFrame(
        {"feature": selected_features, "coefficient": result["coefficients"]}
    ).sort_values("coefficient", key=lambda s: s.abs(), ascending=False)

    st.markdown(
        f"""
        <div class="source-card">
            <h4>{t("model_choice")}</h4>
            <p><strong>{t("region")}:</strong> {region_label(region_key)}</p>
            <p><strong>{t("target")}:</strong> {target_col}</p>
            <p><strong>{t("features")}:</strong> {", ".join(selected_features)}</p>
            <p><strong>{t("model")}:</strong> {result["model_name"]}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    left, right = st.columns([1.5, 1])
    with left:
        fig = px.scatter(
            model_df,
            x="predicted",
            y=target_col,
            color="abs_residual",
            hover_data=selected_features,
            color_continuous_scale="Reds",
            title=t("actual_predicted"),
        )
        min_val = float(min(model_df["predicted"].min(), model_df[target_col].min()))
        max_val = float(max(model_df["predicted"].max(), model_df[target_col].max()))
        fig.add_trace(
            go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode="lines",
                name=t("perfect_prediction"),
                line={"dash": "dash", "color": "#475569"},
            )
        )
        apply_plot_style(fig, height=440, title=t("actual_predicted"))
        st.plotly_chart(fig, width="stretch", config=plot_config)
    with right:
        st.markdown(f"#### {t('coefficients')}")
        if result["model_key"] != "ols":
            st.caption(t("coefficient_note"))
        show_dataframe(coefficient_df, width="stretch", hide_index=True)

    show_metrics(result)
    plain_aqi_reading(model_df, target_col)
    metric_explanation(result)
    model_profile(result)
    show_model_equation(result, selected_features, target_col)
    st.markdown(f"#### {t('model_comparison')}")
    show_dataframe(
        compare_linear_models(model_df[selected_features], model_df[target_col]),
        width="stretch",
        hide_index=True,
    )

    st.markdown(f"#### {t('top_outliers')}")
    residual_scope_note()
    preferred_cols = [
        col
        for col in [
            "rank",
            "county",
            "sitename",
            "publish_time",
            target_col,
            "predicted",
            "residual",
            "abs_residual",
            *selected_features,
        ]
        if col in display_df.columns
    ]
    show_dataframe(
        display_df.sort_values("abs_residual", ascending=False).head(10)[preferred_cols],
        width="stretch",
        hide_index=True,
    )
    if not AQI_DATA_IS_PRIMARY:
        st.warning(t("local_historical_warning") if AQI_DATA_IS_LOCAL_HISTORICAL else t("no_active_data"))
    st.markdown(dataset_status_html(full_row_count, len(model_df)), unsafe_allow_html=True)


st.markdown(
    f"""
    <div class="brand-row">
        <div class="brand-mark" aria-hidden="true">
            <svg viewBox="0 0 32 32" fill="none" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round">
                <path d="M6 25H27" opacity="0.72" />
                <path d="M6 25V6" opacity="0.72" />
                <path d="M8 22C12.8 18.8 17.3 14.5 25 9" />
                <circle cx="10" cy="20" r="2.2" fill="currentColor" stroke="none" />
                <circle cx="15" cy="17" r="2.2" fill="currentColor" stroke="none" />
                <circle cx="20" cy="13" r="2.2" fill="currentColor" stroke="none" />
                <circle cx="25" cy="9" r="2.2" fill="currentColor" stroke="none" />
            </svg>
        </div>
        <div class="main-title">{t("hero_title")}</div>
    </div>
    """,
    unsafe_allow_html=True,
)
st.markdown(
    f'<div class="subtitle">{t("hero_subtitle")}</div>',
    unsafe_allow_html=True,
)

active_page = page_navigation()

if active_page == "lab":
    synthetic_lab()
elif active_page == "aqi":
    aqi_case()
elif active_page == "data":
    st.subheader(t("source_title"))
    st.markdown(t("source_body"))
    source_download_panel("source", include_report=False)
else:
    crisp_report_tab()
