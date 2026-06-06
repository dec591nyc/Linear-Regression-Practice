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
SAMPLE_AQI_PATH = PROJECT_ROOT / "data" / "central_taiwan_aqi_sample.csv"
CENTRAL_COUNTIES = ["臺中市", "台中市", "彰化縣"]
DEFAULT_SOURCE_NAME = "Kaggle Taiwan Air Quality Index Data 2016~2024"
DEFAULT_SOURCE_URL = "https://www.kaggle.com/datasets/taweilo/taiwan-air-quality-data-20162024"
OFFICIAL_SOURCE_NAME = "MOENV AQX_P_432"
OFFICIAL_SOURCE_URL = "https://data.moenv.gov.tw/dataset/detail/aqx_p_432"

TEXT = {
    "en": {
        "page_title": "Linear Regression Practice",
        "hero_title": "Linear Regression Practice",
        "hero_subtitle": "A compact regression dashboard for testing baseline modeling, residual ranking, and central Taiwan air-quality use cases.",
        "theme": "Theme",
        "language": "Language",
        "to_zh": "🌐 繁",
        "to_en": "🌐 EN",
        "light": "Light",
        "dark": "Dark",
        "theme_light_btn": "☀️",
        "theme_dark_btn": "🌙",
        "regression_tab": "Regression Sandbox",
        "aqi_tab": "Central Taiwan AQI",
        "crisp_tab": "CRISP-DM Report",
        "source_tab": "Source Evaluation",
        "synthetic_title": "Regression Sandbox",
        "synthetic_note": "This sandbox keeps the required numeric regression workflow: generate data with n, a, b, and variance, fit a line, then rank the top residual outliers.",
        "synthetic_params": "Synthetic Parameters",
        "sample_size": "Sample size (n)",
        "slope": "True slope (a)",
        "intercept": "True intercept (b)",
        "variance": "Noise variance (var)",
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
        "model_ols_reading": "OLS is the clearest baseline. It is easy to explain, but it can become unstable when pollutant fields move together.",
        "model_ridge_reading": "Ridge keeps all selected features but shrinks their influence. It is useful when PM2.5, PM10 and other pollutants are correlated.",
        "model_lasso_reading": "Lasso can shrink weak feature effects toward zero. It is useful for feature screening, but it may be too aggressive on a small sample.",
        "model_elasticnet_reading": "ElasticNet balances Ridge and Lasso. It is a practical compromise when features are correlated but some simplification is still useful.",
        "coefficient_note": "For Ridge, Lasso and ElasticNet, coefficients are based on standardized inputs, so compare direction and relative size rather than raw units.",
        "equation": "Estimated equation",
        "generated_data": "Generated data",
        "outliers": "Top 10 residual outliers",
        "true_line": "True line",
        "regression_line": "Regression line",
        "top_outliers": "Top residual observations",
        "aqi_title": "Central Taiwan AQI Case",
        "aqi_note": "This case uses Taichung and Changhua air-quality readings to predict a numeric target and surface observations that the baseline model cannot explain well.",
        "target": "Target variable",
        "features": "Feature variables",
        "source_run": "Data source in this run",
        "complete_rows": "Complete rows used",
        "bundled_sample": "Bundled central Taiwan sample",
        "coefficients": "Coefficients",
        "actual_predicted": "Actual vs Predicted",
        "perfect_prediction": "Perfect prediction",
        "source_panel_title": "Default data source",
        "source_panel_body": "Current app run uses the bundled central Taiwan AQI sample. It is a compact classroom demo dataset with fields aligned to Taiwan AQI open data, not a live API pull. Kaggle is the recommended reproducible dataset reference; MOENV AQX_P_432 is the official replacement source for production use.",
        "download_data": "Download sample CSV",
        "download_report": "Download CRISP-DM report",
        "source_title": "Data Source Evaluation",
        "source_body": """
        - Current app run uses the bundled `data/central_taiwan_aqi_sample.csv` sample for modeling and charting.
        - Kaggle `Taiwan Air Quality Index Data 2016~2024` is useful for reproducible modeling practice because the data has already been collected and shaped for analysis.
        - The official replacement source is MOENV `AQX_P_432`, which provides hourly station-level AQI and pollutant fields.
        - Kaggle is convenient for a portfolio demo; the official API is better for production because it is closer to the source of truth.
        - The synthetic sandbox stays separate from the AQI case so the baseline regression workflow remains clear.
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
        "chart_plain_aqi": "Each dot is one air-quality record. The horizontal position is what the model guessed, and the vertical position is the real value. Dots close to the diagonal line mean the model guessed well. Dots far from the line are records worth checking because the pollutant pattern looked unusual.",
        "chart_plain_stats": "In this run, the average {target} is {avg:.1f}. The largest miss is about {residual:.1f}. This does not automatically mean the air is dangerous; it means the reading does not match the simple pattern learned from the selected pollutants.",
        "chart_plain_caution": "Because this is a small baseline demo, use it to understand patterns and suspicious records, not as an official air-quality forecast.",
        "crisp_title": "CRISP-DM practice report",
        "crisp_intro": "CRISP-DM turns this project from a charting demo into a repeatable data-mining workflow: first define the air-quality question, then understand data fields, prepare numeric inputs, train a baseline model, evaluate residuals, and package the result for reuse.",
        "crisp_business": "Business understanding: identify air-quality observations that do not follow the usual pollutant pattern, so a user can inspect unusual AQI readings faster.",
        "crisp_data": "Data understanding: use station-level AQI fields such as PM2.5, PM10, O3, NO2, CO, SO2 and wind speed. The default sample focuses on Taichung and Changhua.",
        "crisp_prep": "Data preparation: normalize column names, keep numeric fields, remove incomplete rows, and let users choose the prediction target and features.",
        "crisp_model": "Modeling: train a simple linear regression baseline and produce predicted values.",
        "crisp_eval": "Evaluation: use R-squared, RMSE, MAE and residual ranking. The largest residuals become the records worth checking manually.",
        "crisp_deploy": "Deployment: provide a bilingual Streamlit interface, sample-data download, source notes and a downloadable CRISP-DM report. External upload is intentionally removed until a schema validator is defined.",
    },
    "zh": {
        "page_title": "線性迴歸實作",
        "hero_title": "線性迴歸實作",
        "hero_subtitle": "以台中、彰化空氣品質為情境，展示基礎迴歸建模、殘差排序與異常觀測判讀。",
        "theme": "主題",
        "language": "語言",
        "to_zh": "🌐 繁",
        "to_en": "🌐 EN",
        "light": "淺色",
        "dark": "深色",
        "theme_light_btn": "☀️",
        "theme_dark_btn": "🌙",
        "regression_tab": "迴歸模擬器",
        "aqi_tab": "中彰 AQI 案例",
        "crisp_tab": "CRISP-DM 報告",
        "source_tab": "資料來源評估",
        "synthetic_title": "迴歸模擬器",
        "synthetic_note": "此區保留數值迴歸的核心流程：用 n、a、b、變異數產生資料，擬合迴歸線，再用殘差排序找出異常觀測。",
        "synthetic_params": "模擬參數",
        "sample_size": "樣本數 (n)",
        "slope": "真實斜率 (a)",
        "intercept": "真實截距 (b)",
        "variance": "雜訊變異數 (var)",
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
        "model_ols_reading": "OLS 是最清楚的 baseline，容易解釋，但污染物欄位彼此高度相關時，係數可能不穩定。",
        "model_ridge_reading": "Ridge 會保留所有選取特徵，但降低個別特徵的影響力；當 PM2.5、PM10 等污染物彼此相關時，通常較穩定。",
        "model_lasso_reading": "Lasso 會把較弱的特徵影響壓向 0，適合做特徵篩選；但在小型 sample 上可能過度簡化。",
        "model_elasticnet_reading": "ElasticNet 折衷 Ridge 與 Lasso，適合特徵彼此相關、但仍希望模型稍微簡化的情境。",
        "coefficient_note": "Ridge、Lasso、ElasticNet 的係數來自標準化輸入，適合比較方向與相對影響，不宜直接解讀為原始單位。",
        "equation": "估計方程式",
        "generated_data": "模擬資料",
        "outliers": "殘差前 10 名",
        "true_line": "真實線",
        "regression_line": "迴歸線",
        "top_outliers": "殘差最高觀測",
        "aqi_title": "中彰 AQI 實際案例",
        "aqi_note": "此區以台中與彰化空氣品質資料預測數值目標，並找出基礎模型較難解釋的污染觀測。",
        "target": "預測目標",
        "features": "特徵欄位",
        "source_run": "本次資料來源",
        "complete_rows": "完整資料筆數",
        "bundled_sample": "內建中彰 sample",
        "coefficients": "模型係數",
        "actual_predicted": "實際值與預測值",
        "perfect_prediction": "理想預測線",
        "source_panel_title": "預設資料來源",
        "source_panel_body": "目前 app 實際執行使用的是內建中彰 AQI sample。這是一份小型課堂示範資料，欄位邏輯對齊台灣空氣品質開放資料，但不是即時 API 抓取。Kaggle 是建議用於可重現練習的資料參考；MOENV AQX_P_432 則是正式應用時應改用的官方來源。",
        "download_data": "下載 sample CSV",
        "download_report": "下載 CRISP-DM 報告",
        "source_title": "資料來源評估",
        "source_body": """
        - 目前 app 實際建模與圖表使用的是內建 `data/central_taiwan_aqi_sample.csv` sample。
        - Kaggle `Taiwan Air Quality Index Data 2016~2024` 適合展示與練習，因為資料已整理成可分析格式。
        - 正式替代來源是環境部 `AQX_P_432`，可取得每小時測站 AQI 與污染物欄位。
        - Kaggle 適合作品展示與可重現分析；環境部 API 更接近正式資料源。
        - 模擬迴歸與 AQI 案例分開呈現，讓基礎迴歸流程與實際案例不互相混淆。
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
        "chart_plain_aqi": "每一個點代表一筆空氣品質觀測。橫軸是模型猜出的數值，縱軸是真實數值。點越靠近斜線，代表模型猜得越準；離斜線越遠，代表這筆資料和一般污染物規律不太一樣，值得再檢查。",
        "chart_plain_stats": "本次資料的平均 {target} 約為 {avg:.1f}，最大誤差約為 {residual:.1f}。這不一定代表空氣很危險，而是代表這筆觀測不太符合目前模型學到的簡單規律。",
        "chart_plain_caution": "因為這是小型 baseline demo，適合用來理解污染趨勢與可疑觀測，不適合作為正式空氣品質預報。",
        "crisp_title": "CRISP-DM 實踐報告",
        "crisp_intro": "CRISP-DM 讓這個專案不只是畫圖工具，而是一個可重複使用的數據探勘流程：先定義空氣品質問題，再理解資料欄位、整理數值特徵、建立 baseline 模型、評估殘差，最後包裝成可操作的分析介面。",
        "crisp_business": "商業理解：找出不符合一般污染物規律的空氣品質觀測，協助使用者更快注意可疑 AQI 讀數。",
        "crisp_data": "資料理解：使用測站層級 AQI 欄位，例如 PM2.5、PM10、O3、NO2、CO、SO2 與風速；預設 sample 聚焦台中與彰化。",
        "crisp_prep": "資料準備：正規化欄位名稱、保留數值欄位、移除不完整資料，並讓使用者選擇預測目標與特徵。",
        "crisp_model": "模型建立：訓練簡單線性迴歸 baseline，產生每筆觀測的預測值。",
        "crisp_eval": "模型評估：使用 R-squared、RMSE、MAE 與殘差排序；殘差最大的資料就是最值得人工檢查的觀測。",
        "crisp_deploy": "部署應用：提供雙語 Streamlit 介面、sample 資料下載、資料來源說明與可下載 CRISP-DM 報告。外部上傳在尚未定義 schema 驗證前先移除。",
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

t = TEXT[st.session_state.locale].get

THEMES = {
    "light": {
        "bg": "#f4f6fa",
        "panel": "#ffffff",
        "text": "#111827",
        "muted": "#4b5563",
        "border": "#cfd8e3",
        "accent": "#0f766e",
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
    },
    "dark": {
        "bg": "#0f172a",
        "panel": "#182235",
        "text": "#edf2ff",
        "muted": "#a8b3c7",
        "border": "#2f3b52",
        "accent": "#38bdf8",
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
    },
}
theme = THEMES[st.session_state.theme]
plot_template = "plotly_dark" if st.session_state.theme == "dark" else "plotly_white"
plot_config = {"displayModeBar": False, "responsive": True}

st.markdown(
    f"""
    <style>
    .stAppToolbar, #MainMenu, footer {{
        display: none !important;
        visibility: hidden !important;
        height: 0 !important;
    }}
    html, body, [data-testid="stAppViewContainer"] {{
        background: {theme["bg"]};
        color: {theme["text"]};
    }}
    [data-testid="stApp"], .stApp {{
        background: {theme["bg"]} !important;
        color: {theme["text"]} !important;
    }}
    [data-testid="stAppViewContainer"] *,
    [data-testid="stSidebar"] * {{
        color: {theme["text"]};
    }}
    .block-container {{
        max-width: 1180px;
        padding-top: 1.35rem;
        padding-bottom: 2.25rem;
    }}
    [data-testid="stHeader"] {{
        background: {theme["bg"]} !important;
    }}
    [data-testid="stHeader"] * {{
        color: {theme["text"]} !important;
    }}
    [data-testid="stToolbar"],
    [data-testid="stToolbar"] *,
    [data-testid="collapsedControl"],
    [data-testid="collapsedControl"] *,
    [data-testid="stSidebarCollapseButton"],
    [data-testid="stSidebarCollapseButton"] *,
    [data-testid="stSidebarCollapsedControl"],
    [data-testid="stSidebarCollapsedControl"] * {{
        visibility: visible !important;
        opacity: 1 !important;
        pointer-events: auto !important;
        color: {theme["text"]} !important;
        fill: {theme["text"]} !important;
    }}
    [data-testid="collapsedControl"],
    [data-testid="stSidebarCollapsedControl"] {{
        display: flex !important;
        z-index: 1000 !important;
    }}
    [data-testid="stSidebar"],
    section[data-testid="stSidebar"] {{
        background: {theme["panel"]};
        border-right: 1px solid {theme["border"]};
    }}
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] div,
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
        border: 1px solid {theme["border"]} !important;
        border-radius: 8px !important;
    }}
    [data-testid="stFileUploaderDropzone"] * {{
        color: {theme["control_text"]} !important;
    }}
    [data-testid="stFileUploaderDropzone"] small {{
        color: {theme["control_muted"]} !important;
    }}
    button {{
        color: {theme["control_text"]} !important;
    }}
    div.stButton > button,
    div[data-testid="stDownloadButton"] > button {{
        background: {theme["control_bg"]} !important;
        color: {theme["control_text"]} !important;
        border: 1px solid {theme["border"]} !important;
        border-radius: 8px !important;
        font-weight: 650 !important;
    }}
    div.stButton > button:hover,
    div[data-testid="stDownloadButton"] > button:hover {{
        border-color: {theme["accent"]} !important;
        color: {theme["accent"]} !important;
    }}
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
    </style>
    """,
    unsafe_allow_html=True,
)

t = TEXT[st.session_state.locale].get

with st.sidebar:
    col_lang, col_theme = st.columns(2)
    with col_lang:
        lang_label = t("to_en") if st.session_state.locale == "zh" else t("to_zh")
        if st.button(lang_label, use_container_width=True, key="lang_toggle", help=t("language")):
            st.session_state.locale = "en" if st.session_state.locale == "zh" else "zh"
            st.rerun()
    with col_theme:
        theme_label = t("theme_dark_btn") if st.session_state.theme == "light" else t("theme_light_btn")
        if st.button(theme_label, use_container_width=True, key="theme_toggle", help=t("theme")):
            st.session_state.theme = "dark" if st.session_state.theme == "light" else "light"
            st.rerun()


MODEL_OPTIONS = {
    "ols": {"label_key": "model_ols", "reading_key": "model_ols_reading"},
    "ridge": {"label_key": "model_ridge", "reading_key": "model_ridge_reading"},
    "lasso": {"label_key": "model_lasso", "reading_key": "model_lasso_reading"},
    "elasticnet": {"label_key": "model_elasticnet", "reading_key": "model_elasticnet_reading"},
}


def model_label(model_key: str) -> str:
    return t(MODEL_OPTIONS.get(model_key, MODEL_OPTIONS["ols"])["label_key"])


def model_reading(model_key: str) -> str:
    return t(MODEL_OPTIONS.get(model_key, MODEL_OPTIONS["ols"])["reading_key"])


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
    }


def safe_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) < 2 or np.allclose(y_true, y_true[0]):
        return 0.0
    if LinearRegression is not None:
        return float(r2_score(y_true, y_pred))
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    return 1 - ss_res / ss_tot if ss_tot else 0.0


def apply_plot_style(fig: go.Figure, height: int, title: str | None = None) -> go.Figure:
    fig.update_layout(
        height=height,
        template=plot_template,
        paper_bgcolor=theme["plot_bg"],
        plot_bgcolor=theme["plot_bg"],
        font={"color": theme["plot_text"], "size": 13},
        title={"text": title} if title else None,
        title_font={"color": theme["plot_text"], "size": 18},
        legend={
            "font": {"color": theme["plot_text"], "size": 12},
            "bgcolor": "rgba(0,0,0,0)",
            "bordercolor": "rgba(0,0,0,0)",
        },
        coloraxis_colorbar={
            "title": {"font": {"color": theme["plot_text"], "size": 13}},
            "tickfont": {"color": theme["plot_text"], "size": 12},
        },
        margin={"l": 72, "r": 44, "t": 60 if title else 36, "b": 64},
    )
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
        "WIND_DIREC": "wind_direc",
        "Longitude": "longitude",
        "Latitude": "latitude",
        "publishtime": "publish_time",
        "PublishTime": "publish_time",
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


def report_markdown(locale: str | None = None) -> str:
    active_locale = locale or st.session_state.locale
    if active_locale == "zh":
        return f"""# CRISP-DM 報告：Linear Regression Practice

## 1. 商業理解

本專案使用台中、彰化空氣品質資料，找出不符合一般污染物規律的觀測。實務目標不是取代官方預報，而是協助非技術使用者更快注意可疑 AQI 讀數。

## 2. 資料理解

目前 app 實際使用資料：內建中彰 AQI sample

內建 sample 說明：這是一份小型課堂示範資料，欄位邏輯對齊台灣空氣品質開放資料，保留測站、縣市、發布時間、AQI、PM2.5、PM10、O3、NO2、CO、SO2 與風速等欄位。它適合示範數值建模流程，但不是即時 API 抓取。

建議可重現資料來源：{DEFAULT_SOURCE_NAME}

Kaggle URL：{DEFAULT_SOURCE_URL}

正式替代來源：{OFFICIAL_SOURCE_NAME}

MOENV URL：{OFFICIAL_SOURCE_URL}

資料來源判斷：Kaggle 適合作品展示與練習，因為資料已整理成可分析格式；MOENV AQX_P_432 更接近正式資料源，適合未來要接近真實應用時改用。

## 3. 資料準備

- 正規化常見 AQI 欄位名稱。
- 將污染物與氣象欄位轉成數值。
- 若資料包含縣市欄位，篩選台中與彰化。
- 移除預測目標與特徵欄位缺值的資料列。
- 讓使用者在介面中選擇目標欄位與特徵組合。

## 4. 模型建立

本專案仍維持線性回歸作業範圍，但增加多個線性模型變體：

- OLS 普通最小平方法：最清楚的 baseline，適合說明線性關係。
- Ridge 迴歸：保留所有特徵並降低係數波動，適合污染物欄位彼此相關的情境。
- Lasso 迴歸：可將較弱特徵壓向 0，適合做特徵篩選。
- ElasticNet 迴歸：折衷 Ridge 與 Lasso，兼顧穩定性與簡化。

## 5. 模型評估

介面使用 R-squared、RMSE、MAE 與殘差排序。R-squared 用來看模型抓到多少規律；RMSE 與 MAE 用來看平均誤差；絕對殘差最大的資料列，是最值得人工檢查的異常觀測。

## 6. 部署與使用

結果包裝成雙語 Streamlit app，支援明暗色、sample 資料下載、資料來源說明、白話解讀與依目前語言輸出的 CRISP-DM 報告。外部 CSV 上傳在尚未定義 schema 驗證前先移除。
"""

    return f"""# CRISP-DM Report: Linear Regression Practice

## 1. Business Understanding

This project uses central Taiwan air-quality data to identify observations that do not follow the usual pollutant pattern. The practical goal is to help a non-technical user notice suspicious AQI readings faster.

## 2. Data Understanding

Current app data: bundled central Taiwan AQI sample

Bundled sample note: this is a compact classroom demo dataset with field logic aligned to Taiwan AQI open data. It keeps station, county, publish time, AQI, PM2.5, PM10, O3, NO2, CO, SO2 and wind-speed-like fields. It is useful for demonstrating numeric modeling, but it is not a live API pull.

Recommended reproducible dataset reference: {DEFAULT_SOURCE_NAME}

Kaggle URL: {DEFAULT_SOURCE_URL}

Official replacement source: {OFFICIAL_SOURCE_NAME}

MOENV URL: {OFFICIAL_SOURCE_URL}

Source judgment: Kaggle is suitable for reproducible classroom and portfolio practice because it is already shaped for analysis. MOENV AQX_P_432 is closer to the source of truth and should be used for a future production-like version.

## 3. Data Preparation

- Normalize common AQI column names.
- Convert pollutant and weather fields into numeric values.
- Filter central Taiwan counties when county data exists.
- Drop rows missing the selected target or feature fields.
- Let users choose a target and feature set in the interface.

## 4. Modeling

The app stays within the linear-regression assignment scope but adds multiple linear model variants:

- OLS Linear Regression: the clearest baseline for explaining a linear relationship.
- Ridge Regression: keeps all features while reducing coefficient instability, useful when pollutant fields are correlated.
- Lasso Regression: can shrink weaker feature effects toward zero, useful for feature screening.
- ElasticNet Regression: balances Ridge and Lasso for a practical stability/simplification compromise.

## 5. Evaluation

The app reports R-squared, RMSE and MAE. It also ranks observations by absolute residual. Large residuals are the records where the model's guess differs most from the actual value.

## 6. Deployment

The result is packaged as a bilingual Streamlit app with light/dark themes, sample data download, source notes, plain-language explanations below the chart and a CRISP-DM report generated in the currently selected language. External upload is intentionally removed until a schema validator is defined.
"""


def source_download_panel(panel_id: str) -> None:
    sample_bytes = SAMPLE_AQI_PATH.read_bytes()
    report_text = report_markdown(st.session_state.locale)
    report_file_name = f"crisp_dm_report_{st.session_state.locale}.md"

    st.markdown(
        f"""
        <div class="source-card">
            <h4>{t("source_panel_title")}</h4>
            <p><strong>{t("source_run")}:</strong> {t("bundled_sample")}</p>
            <p>{t("source_panel_body")}</p>
            <p><strong>{DEFAULT_SOURCE_NAME}</strong><br>
            <a href="{DEFAULT_SOURCE_URL}" target="_blank">{DEFAULT_SOURCE_URL}</a></p>
            <p><strong>{OFFICIAL_SOURCE_NAME}</strong><br>
            <a href="{OFFICIAL_SOURCE_URL}" target="_blank">{OFFICIAL_SOURCE_URL}</a></p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    col_data, col_report = st.columns(2)
    with col_data:
        st.download_button(
            t("download_data"),
            data=sample_bytes,
            file_name="central_taiwan_aqi_sample.csv",
            mime="text/csv",
            use_container_width=True,
            key=f"download_sample_{panel_id}",
        )
    with col_report:
        st.download_button(
            t("download_report"),
            data=report_text.encode("utf-8"),
            file_name=report_file_name,
            mime="text/markdown",
            use_container_width=True,
            key=f"download_report_{panel_id}",
        )


def crisp_report_tab() -> None:
    phases = [
        t("crisp_business"),
        t("crisp_data"),
        t("crisp_prep"),
        t("crisp_model"),
        t("crisp_eval"),
        t("crisp_deploy"),
    ]
    st.subheader(t("crisp_title"))
    st.markdown(f'<div class="note">{t("crisp_intro")}</div>', unsafe_allow_html=True)
    for index, phase in enumerate(phases, start=1):
        st.markdown(
            f"""
            <div class="crisp-card">
                <h4>{index}. {phase.split("：", 1)[0].split(":", 1)[0]}</h4>
                <p>{phase}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    source_download_panel("crisp")


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


def synthetic_lab() -> None:
    st.subheader(t("synthetic_title"))
    st.markdown(
        f'<div class="note">{t("synthetic_note")}</div>',
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.markdown(f"### {t('synthetic_params')}")
        n = st.slider(t("sample_size"), 50, 1000, 300, 50)
        a = st.slider(t("slope"), -50.0, 50.0, 8.0, 0.5)
        b = st.slider(t("intercept"), -100.0, 100.0, 40.0, 1.0)
        var = st.number_input(t("variance"), 0.0, 100000.0, 10000.0, 500.0)
        seed = st.number_input(t("seed"), 0, 999999, 42, 1)

    rng = np.random.default_rng(seed)
    x = rng.uniform(-100, 100, n)
    noise = rng.normal(0, np.sqrt(var), n)
    y = a * x + b + noise
    df = pd.DataFrame({"x": x, "y": y})

    result = fit_linear_model(df[["x"]], df["y"])
    df["predicted_y"] = result["predictions"]
    df["residual"] = result["residuals"]
    df["abs_residual"] = np.abs(result["residuals"])
    df["rank"] = df["abs_residual"].rank(method="first", ascending=False).astype(int)
    df["is_outlier"] = df["rank"] <= 10

    show_metrics(result)
    metric_explanation(result)

    st.markdown(
        f"{t('equation')}: `y = {result['coefficients'][0]:.4f} * x + {result['intercept']:.4f}`"
    )

    x_line = np.linspace(df["x"].min(), df["x"].max(), 200)
    fitted_line = result["coefficients"][0] * x_line + result["intercept"]
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
    st.plotly_chart(fig, use_container_width=True, config=plot_config)

    st.markdown(f"#### {t('outliers')}")
    st.dataframe(
        df.sort_values("abs_residual", ascending=False)
        .head(10)[["rank", "x", "y", "predicted_y", "residual", "abs_residual"]],
        use_container_width=True,
    )


def aqi_case() -> None:
    st.subheader(t("aqi_title"))
    st.markdown(
        f'<div class="note">{t("aqi_note")}</div>',
        unsafe_allow_html=True,
    )
    source_download_panel("aqi")

    raw_df = pd.read_csv(SAMPLE_AQI_PATH)
    source_label = t("bundled_sample")

    df = normalize_aqi_columns(raw_df)
    if "county" in df.columns:
        df = df[df["county"].isin(CENTRAL_COUNTIES)].copy()

    numeric_cols = get_numeric_columns(df)
    if len(numeric_cols) < 2:
        st.error(t("too_few_cols"))
        st.dataframe(df.head(20), use_container_width=True)
        return

    default_target = "aqi" if "aqi" in numeric_cols else numeric_cols[-1]
    target_col = st.selectbox(t("target"), numeric_cols, index=numeric_cols.index(default_target))
    feature_options = [col for col in numeric_cols if col != target_col]
    default_features = [
        col
        for col in ["pm25", "pm25_avg", "pm10", "pm10_avg", "o3", "no2", "co", "so2", "wind_speed"]
        if col in feature_options
    ]
    if not default_features:
        default_features = feature_options[: min(5, len(feature_options))]

    selected_features = st.multiselect(t("features"), feature_options, default=default_features)
    if not selected_features:
        st.warning(t("select_feature"))
        return
    available_models = ["ols"] if LinearRegression is None else list(MODEL_OPTIONS.keys())
    model_key = st.selectbox(
        t("model_choice"),
        available_models,
        format_func=model_label,
        key="aqi_model_choice",
    )

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

    st.caption(f"{t('source_run')}: {source_label}. {t('complete_rows')}: {len(model_df)}.")
    show_metrics(result)
    metric_explanation(result)
    model_profile(result)
    st.markdown(f"#### {t('model_comparison')}")
    st.dataframe(compare_linear_models(model_df[selected_features], model_df[target_col]), use_container_width=True, hide_index=True)

    coefficient_df = pd.DataFrame(
        {"feature": selected_features, "coefficient": result["coefficients"]}
    ).sort_values("coefficient", key=lambda s: s.abs(), ascending=False)

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
        st.plotly_chart(fig, use_container_width=True, config=plot_config)
    with right:
        st.markdown(f"#### {t('coefficients')}")
        if result["model_key"] != "ols":
            st.caption(t("coefficient_note"))
        st.dataframe(coefficient_df, use_container_width=True, hide_index=True)

    plain_aqi_reading(model_df, target_col)

    st.markdown(f"#### {t('top_outliers')}")
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
    st.dataframe(
        display_df.sort_values("abs_residual", ascending=False).head(10)[preferred_cols],
        use_container_width=True,
        hide_index=True,
    )


st.markdown(f'<div class="main-title">{t("hero_title")}</div>', unsafe_allow_html=True)
st.markdown(
    f'<div class="subtitle">{t("hero_subtitle")}</div>',
    unsafe_allow_html=True,
)

tab_synthetic, tab_aqi, tab_crisp, tab_sources = st.tabs(
    [t("regression_tab"), t("aqi_tab"), t("crisp_tab"), t("source_tab")]
)

with tab_synthetic:
    synthetic_lab()

with tab_aqi:
    aqi_case()

with tab_crisp:
    crisp_report_tab()

with tab_sources:
    st.subheader(t("source_title"))
    st.markdown(t("source_body"))
    source_download_panel("source")
