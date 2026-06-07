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
        "hero_subtitle": "Simulator uses formula simulation data to understand regression; AQI uses central Taiwan air-quality records for the practical case.",
        "theme": "Theme",
        "language": "Language",
        "sidebar_toggle": "Toggle sidebar",
        "to_zh": "🌐 繁中",
        "to_en": "🌐 EN",
        "light": "Light",
        "dark": "Dark",
        "theme_light_btn": "☀️",
        "theme_dark_btn": "🌙",
        "regression_tab": "Regression Simulator",
        "aqi_tab": "Central AQI",
        "crisp_tab": "CRISP-DM Analysis Report",
        "source_tab": "Data Source",
        "nav_lab": "Simulator",
        "nav_aqi": "AQI",
        "nav_report": "Report",
        "nav_data": "Data Source",
        "synthetic_title": "Regression Simulator",
        "synthetic_note": "This sandbox uses controlled numeric data to show how sample size, slope, intercept, noise and model choice change a regression result.",
        "synthetic_source_title": "Simulator Data Source",
        "synthetic_source_body": "Simulator page simulates a demonstration dataset based on explicit parameters: `x` is sampled uniformly from -100 to 100; `noise` is sampled from a normal distribution with mean 0 and standard deviation `sqrt(var)`; `y` is calculated as `a*x + b + noise`. The sample size, slope, intercept, noise variance, and random seed on the left are the data generation conditions.",
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
        "model_comparison": "Model comparison",
        "model_comparison_note": "This comparison keeps the same target and features, then changes only the linear model. Keep it when you want to see whether a more constrained model improves stability; otherwise treat OLS as the clearest baseline.",
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
        "outlier_plain_note": "Residual means actual value minus predicted value. A large absolute residual does not automatically mean the row is wrong; it means the row deserves review because the model could not explain it well.",
        "true_line": "True line",
        "regression_line": "Regression line",
        "top_outliers": "Top residual observations",
        "outlier_scope_note": "Showing the top 10 rows ranked by absolute residual. The full table is not shown because the active dataset can contain many rows; this view focuses on the observations most worth manual review.",
        "aqi_title": "Central Taiwan Air Quality Index (AQI) Modeling",
        "aqi_note": "Air Quality Index (AQI) is a single number that summarizes air-pollution level. This case prioritizes recent Taichung, Changhua and Nantou observations, then predicts a numeric target and surfaces observations that the baseline model cannot explain well.",
        "aqi_params": "AQI Controls",
        "region": "Region",
        "region_central": "Central Taiwan",
        "region_taichung": "Taichung",
        "region_changhua": "Changhua",
        "region_nantou": "Nantou",
        "active_data": "Reference data",
        "data_rows": "Filtered rows",
        "data_scope": "Scope",
        "data_period": "Period",
        "recent_sample": "MOENV central Taiwan AQI sample",
        "local_historical_sample": "Local Taichung historical sample",
        "local_historical_warning": "The active CSV has enough rows, but it is a Taichung-only historical file from 2016-2026. After filtering to 2018 or later, it has 966 rows, so it does not fully satisfy the requested recent central-Taiwan sample condition.",
        "no_active_data": "No usable AQI CSV is available. The tiny demo fallback has been removed by project decision. Add a larger public dataset or switch to another research topic.",
        "target": "Target variable",
        "features": "Feature variables",
        "source_run": "Data source in this run",
        "complete_rows": "Model-ready rows",
        "bundled_sample": "Bundled central Taiwan sample",
        "coefficients": "Coefficients",
        "actual_predicted": "Actual vs Predicted",
        "perfect_prediction": "Perfect prediction",
        "source_panel_title": "Active dataset",
        "source_panel_body": "The AQI page reads only one built-in CSV file bundled inside the project repository's 'data/' folder on GitHub. Whether running locally or deployed on cloud environments (like Streamlit Community Cloud), the app automatically accesses this bundled file, and users can fully download this real-world dataset via the download button or directly from the GitHub repository folder. Kaggle and MOENV links are reference sources, not extra files loaded at runtime.",
        "sample_scope_note": "Sample scope: the active CSV has {rows} rows. Tiny demo data is not used.",
        "download_data": "Download CSV",
        "download_report": "Download report",
        "source_title": "Data Source",
        "source_intro": "",
        "source_adopt_title": "Data Sources Adopted",
        "source_adopt_body": "This project's AQI modeling page only loads a single built-in real-world dataset bundled inside the GitHub repository: <strong>central_taiwan_aqi_sample.csv</strong> (located in the <code>data/</code> directory). When deployed to a cloud platform (e.g., Streamlit Community Cloud), this bundled file is packaged and deployed alongside the application code. Web users can fully download this CSV via the download button below, or obtain it directly from the GitHub repository folder.<br/><br/>To ensure the modeling and analysis reference clear benchmarks, the <strong>references and adoption conclusions</strong> are as follows:<ol><li><strong>Official Historical Standard (Primary Benchmark)</strong>: The Ministry of Environment's <a href=\"https://data.moenv.gov.tw/en/dataset/detail/aqx_p_488\" target=\"_blank\">MOENV AQX_P_488 Historical Air Quality Records</a> serves as our main historical reference.</li><li><strong>Real-time Standard</strong>: The Ministry of Environment's <a href=\"https://data.moenv.gov.tw/dataset/detail/aqx_p_432\" target=\"_blank\">MOENV AQX_P_432 Real-time AQI</a> serves as our dynamic reference for current observations.</li><li><strong>Archival Option</strong>: The <a href=\"https://www.kaggle.com/datasets/taweilo/taiwan-air-quality-data-20162024\" target=\"_blank\">Kaggle Taiwan Air Quality Data (2016~2024)</a> serves as an alternative archive option for reproducible offline historical experiments.</li></ol>",
        "source_quality_title": "Suitability Statement",
        "source_quality_body": "Practicing linear regression using AQI. Considering sample size and web performance constraints, the data is restricted to Central Taiwan (covering Taichung, Changhua, and Nantou). It is suitable for residual ranking and model interpretation, rather than official forecasting services.",
        "download_hint_data": "Exports the exact CSV loaded by the AQI page.",
        "download_hint_report": "Exports the current-language business analysis report.",
        "too_few_cols": "AQI data needs at least two numeric columns after cleaning.",
        "select_feature": "Select at least one feature variable.",
        "too_few_rows": "The selected columns have too few complete rows for a useful regression.",
        "metric_status_title": "Metric status",
        "metric_r2_strong": "R-squared is strong. The model explains most of the target variation in this run.",
        "metric_r2_ok": "R-squared is moderate. The model captures part of the pattern, but residual checks still matter.",
        "metric_r2_weak": "R-squared is weak. Treat this as a baseline model, not a reliable predictor.",
        "metric_error_note": "RMSE and MAE are average error indicators. Lower values mean the prediction is closer to the actual target.",
        "metric_outlier_note": "Rows with the largest absolute residuals are the observations most worth checking manually.",
        "chart_plain_title": "Data reading",
        "chart_plain_aqi": "Each dot is one Air Quality Index (AQI) record or selected numeric air-quality target. The horizontal position is what the model guessed, and the vertical position is the real value. Dots close to the diagonal line mean the model guessed well. Dots far from the line are records worth checking because the pollutant pattern looked unusual.",
        "chart_plain_stats": "In this run, the average {target} is {avg:.1f}. The largest miss is about {residual:.1f}. This does not automatically mean the air is dangerous; it means the reading does not match the simple pattern learned from the selected pollutants.",
        "chart_plain_caution": "Use this baseline to understand patterns and review candidates, not as an official air-quality forecast.",
        "lab_plain_title": "Data reading",
        "lab_plain_body": "Each dot is one generated observation. The dashed line is the rule used to create the data; the fitted line is what the selected model learned from the points. When noise increases, the points spread out and the fitted line becomes harder to trust.",
        "lab_plain_stats": "",
        "interpretation_title": "Model summary",
        "learning_title": "Regression metric guide",
        "learning_body": "R-squared, also called the coefficient of determination, shows how much target variation the model explains. RMSE means Root Mean Squared Error and is sensitive to larger misses. MAE means Mean Absolute Error and shows the average absolute miss in a more direct way. Residual ranking then identifies the records that deserve review.",
        "crisp_title": "CRISP-DM Analysis Report: AQI Linear Regression and Anomaly Detection in Central Taiwan",
        "crisp_intro": "This report presents a systematic evaluation of the Central Taiwan Air Quality Index (AQI) linear regression modeling and anomaly monitoring prototype, framed under the CRISP-DM framework from the professional perspective and tone of a Business Analyst.",
        "crisp_business": "Business understanding: help local environmental or public-health analysts prioritize air-quality observations that deserve review because they do not follow the usual pollutant pattern.",
        "crisp_data": "Data understanding: use station-level Air Quality Index (AQI) fields such as PM2.5, PM10, O3, NO2, CO, SO2 and wind speed. The preferred app dataset is recent central Taiwan data after 2018; tiny demo data is not used.",
        "crisp_prep": "Data preparation: normalize column names, keep numeric fields, remove incomplete rows, and let users choose the prediction target and features.",
        "crisp_model": "Modeling: train a simple linear regression baseline and produce predicted values.",
        "crisp_eval": "Evaluation: use R-squared, RMSE, MAE and residual ranking. The largest residuals become the records worth checking manually.",
        "crisp_deploy": "Deployment: provide a bilingual Streamlit interface, active AQI CSV download, source notes and a downloadable business analysis report. External upload is intentionally removed until a schema validator is defined.",
        "report_audience_title": "Intended audience",
        "report_audience_body": "Primary: local government environmental and public-health analysts. Secondary: community environmental groups, facility or campus operations teams, and residents who need a readable explanation. This is not a consumer forecast product.",
        "report_question_title": "Decision question",
        "report_question_body": "Which station-time observations in Taichung, Changhua and Nantou look inconsistent with the selected pollutant pattern, and should be reviewed before making operational or communication decisions?",
        "report_value_title": "Business value",
        "report_value_body": "The model does not declare air quality safe or unsafe. It reduces review workload by ranking unusual records first, so analysts can inspect station issues, weather effects, pollutant spikes or data-quality problems sooner.",
        "report_limit_title": "Governance limits",
        "report_limit_body": "Use the output as a screening layer. Official alerts, enforcement, health guidance and resident-facing risk messages still require domain review and official monitoring rules.",
        "github_repo_btn": "GitHub Repository",
    },
    "zh": {
        "page_title": "線性迴歸實作",
        "hero_title": "線性迴歸實作",
        "hero_subtitle": "Simulator 以公式示範資料理解迴歸；AQI 使用中部空氣品質資料做實際案例。",
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
        "crisp_tab": "CRISP-DM 分析報告",
        "source_tab": "Data Source",
        "nav_lab": "Simulator",
        "nav_aqi": "AQI",
        "nav_report": "Report",
        "nav_data": "Data Source",
        "synthetic_title": "迴歸模擬器",
        "synthetic_note": "此區使用可控制的數值資料，觀察樣本數、斜率、截距、雜訊與模型選擇如何改變迴歸結果。",
        "synthetic_source_title": "Simulator 資料來源",
        "synthetic_source_body": "Simulator 分頁僅以公式模擬：`x` 從 -100 到 100 的均勻分布抽樣，`noise` 來自平均值 0、標準差 `sqrt(var)` 的常態分布，`y` 則由 `a*x + b + noise` 計算而來。左側的樣本數、斜率、截距、雜訊變異數與隨機種子都是資料生成條件。",
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
        "model_comparison": "模型比較",
        "model_comparison_note": "此比較固定同一組目標與特徵，只更換線性模型。若想確認正規化模型是否讓結果更穩定，可以保留；若只需要最容易說明的基準模型，OLS 就是最清楚的參考。",
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
        "outlier_plain_note": "殘差是實際值減掉預測值。絕對殘差很大不代表資料一定錯，而是代表模型無法很好解釋這筆資料，適合優先檢查。",
        "true_line": "真實線",
        "regression_line": "迴歸線",
        "top_outliers": "殘差最高觀測",
        "outlier_scope_note": "此處只顯示依絕對殘差排序前 10 筆。完整資料可能很多，全部列出會干擾閱讀；這個區塊聚焦最值得人工檢查的觀測。",
        "aqi_title": "中部空氣品質指標 (AQI) 建模",
        "aqi_note": "空氣品質指標 (Air Quality Index, AQI) 是把空氣污染狀況整理成單一數字的指標。此區優先使用台中、彰化、南投的近期觀測，預測數值目標，並找出基礎模型較難解釋的污染觀測。",
        "aqi_params": "AQI 控制項",
        "region": "地區",
        "region_central": "中部",
        "region_taichung": "台中",
        "region_changhua": "彰化",
        "region_nantou": "南投",
        "active_data": "參考資料",
        "data_rows": "篩選後資料筆數",
        "data_scope": "資料範圍",
        "data_period": "資料期間",
        "recent_sample": "環境部中部 AQI sample",
        "local_historical_sample": "本機台中歷史 sample",
        "local_historical_warning": "目前啟用 CSV 的筆數足夠，但它是 2016-2026 的台中歷史資料；若只保留 2018 以後，只剩 966 筆，因此不完全符合近期中部 sample 條件。",
        "no_active_data": "目前沒有可用的 AQI CSV。依專案決策，微型 demo fallback 已移除；請加入較大的公開資料，或改用其他研究主題。",
        "target": "預測目標",
        "features": "特徵欄位",
        "source_run": "本次資料來源",
        "complete_rows": "建模可用筆數",
        "bundled_sample": "內建中彰 sample",
        "coefficients": "模型係數",
        "actual_predicted": "實際值與預測值",
        "perfect_prediction": "理想預測線",
        "source_panel_title": "啟用資料集",
        "source_panel_body": "AQI 頁面僅讀取一份專案內建的 CSV 檔案（已打包在 GitHub 倉庫的 'data/' 目錄中）。不論是在本機開發或是部署於雲端平台，系統均能自動讀取該目錄下的資料，且使用者皆可點擊「下載 CSV」按鈕正常下載此真實世界數據，或直接在 GitHub 原始碼目錄下載。Kaggle 與環境部連結則是作為資料參考來源。",
        "sample_scope_note": "樣本範圍：目前啟用 CSV 有 {rows} 筆。",
        "download_data": "下載 CSV",
        "download_report": "下載報告",
        "source_title": "Data Source",
        "source_intro": "",
        "source_adopt_title": "採用資料來源",
        "source_adopt_body": "本專案的 AQI 建模僅實際載入一份專案 GitHub 倉庫內建的真實世界資料集：<strong>central_taiwan_aqi_sample.csv</strong>（存放於專案原始碼的 <code>data/</code> 目錄中）。當應用程式部署於雲端平台（如 Streamlit Community Cloud）時，此內建檔案會隨同程式碼打包部署，網頁使用者皆可點擊下方按鈕直接下載該 CSV 檔案，亦可直接至 GitHub 原始碼目錄下載。<br/><br/>為確保模型開發與分析具有可信的參考對照，本專案之<strong>參考來源與採用結論</strong>如下：<ol><li><strong>官方歷史資料 (主要對照基準)</strong>：環境部 <a href=\"https://data.moenv.gov.tw/en/dataset/detail/aqx_p_488\" target=\"_blank\">MOENV AQX_P_488 歷史空氣品質監測資料庫</a>，作為歷史比對之主體。</li><li><strong>即時觀測參考</strong>：環境部 <a href=\"https://data.moenv.gov.tw/dataset/detail/aqx_p_432\" target=\"_blank\">MOENV AQX_P_432 即時空氣品質指標</a>，作為即時監控時的動態參照。</li><li><strong>存檔封存方案</strong>：<a href=\"https://www.kaggle.com/datasets/taweilo/taiwan-air-quality-data-20162024\" target=\"_blank\">Kaggle Taiwan Air Quality Data (2016~2024)</a>，作為需要離線重現歷史封存數據時的公開封存檔案替代方案。</li></ol>",
        "source_quality_title": "適用性說明",
        "source_quality_body": "以 AQI 實踐線性迴歸，考量樣本數及網頁效能限制，將數據控制在中部地區（涵蓋台中、彰化、南投）。本系統適合用於殘差排序與模型判讀，不等於官方預報服務。",
        "download_hint_data": "匯出 AQI 分頁實際載入的 CSV。",
        "download_hint_report": "匯出目前語言版本的 CRISP-DM 分析報告。",
        "too_few_cols": "清理後的 AQI 資料至少需要兩個數值欄位。",
        "select_feature": "請至少選擇一個特徵欄位。",
        "too_few_rows": "所選欄位的建模可用筆數太少，不適合建立迴歸模型。",
        "metric_status_title": "指標狀況",
        "metric_r2_strong": "R-squared 表現佳，代表本次模型能解釋多數目標變化。",
        "metric_r2_ok": "R-squared 表現中等，模型抓到部分規律，但仍需要看殘差。",
        "metric_r2_weak": "R-squared 偏弱，這次結果適合視為 baseline，不宜當作可靠預測模型。",
        "metric_error_note": "RMSE 與 MAE 是平均誤差指標，數值越低代表預測越接近實際目標。",
        "metric_outlier_note": "絕對殘差最大的資料列，就是最值得人工檢查的異常觀測。",
        "chart_plain_title": "數據解讀",
        "chart_plain_aqi": "每一個點代表一筆空氣品質指標 (Air Quality Index, AQI) 或目前選定的空氣品質數值目標。橫軸是模型猜出的數值，縱軸是真實數值。點越靠近斜線，代表模型猜得越準；離斜線越遠，代表這筆資料和一般污染物規律不太一樣，值得再檢查。",
        "chart_plain_stats": "本次資料的平均 {target} 約為 {avg:.1f}，最大誤差約為 {residual:.1f}。這不一定代表空氣很危險，而是代表這筆觀測不太符合目前模型學到的簡單規律。",
        "chart_plain_caution": "這個 baseline 適合用來理解污染趨勢與可疑觀測，不適合作為正式空氣品質預報。",
        "lab_plain_title": "數據解讀",
        "lab_plain_body": "每個點代表一筆產生出的觀測。虛線是資料生成規則，迴歸線是模型從點位中學到的結果。雜訊越高，點越分散，模型線也越不容易穩定。",
        "lab_plain_stats": "",
        "interpretation_title": "模型摘要",
        "learning_title": "迴歸判讀說明",
        "learning_body": "R-squared 又稱判定係數，用來看模型能解釋多少目標變化。RMSE 是 Root Mean Squared Error，中文可理解為均方根誤差，對大型誤差比較敏感。MAE 是 Mean Absolute Error，中文可理解為平均絕對誤差，代表平均猜錯多少。殘差排序則用來找出最值得複查的資料列。",
        "crisp_title": "CRISP-DM 分析報告：中部空氣品質線性迴歸與異常監測",
        "crisp_intro": "本報告以商業分析的視角並結合 CRISP-DM 框架系統性說明中部地區空氣品質指標 (AQI) 線性迴歸建模與異常監測的原型評估。",
        "crisp_business": "商業理解：協助地方環境或公共衛生分析人員，優先找出不符合一般污染物規律、值得複查的空氣品質觀測。",
        "crisp_data": "資料理解：使用測站層級空氣品質指標 (AQI) 欄位，例如 PM2.5、PM10、O3、NO2、CO、SO2 與風速；理想 app 資料是 2018 以後近期中部資料，微型 demo 資料不再使用。",
        "crisp_prep": "資料準備：正規化欄位名稱、保留數值欄位、移除不完整資料，並讓使用者選擇預測目標與特徵。",
        "crisp_model": "模型建立：訓練簡單線性迴歸 baseline，產生每筆觀測的預測值。",
        "crisp_eval": "模型評估：使用 R-squared、RMSE、MAE 與殘差排序；殘差最大的資料就是最值得人工檢查的觀測。",
        "crisp_deploy": "部署應用：提供雙語 Streamlit 介面、目前 AQI CSV 下載、資料來源說明與可下載商業分析報告。外部上傳在尚未定義 schema 驗證前先移除。",
        "report_audience_title": "目標受眾",
        "report_audience_body": "主要受眾是地方政府環境與公共衛生分析人員；次要受眾是社區環保倡議者、園區或校園營運管理者，以及想要了解空氣品質的居民。",
        "report_question_title": "決策問題",
        "report_question_body": "在台中、彰化、南投的測站紀錄中，哪些站點與時間的觀測值不符合目前污染物組合的通常規律，需要先複查再進一步做營運或溝通判斷？",
        "report_value_title": "分析價值",
        "report_value_body": "模型不判定空氣安全或危險，而是把最不容易被 baseline 解釋的紀錄排到前面，協助分析人員更快檢查測站異常、天候影響、污染物突增或資料品質問題。",
        "report_limit_title": "治理限制",
        "report_limit_body": "此結果適合作為篩選層。正式警示、稽查、健康建議與居民風險溝通，仍需結合專業審查與官方監測規範。",
        "github_repo_btn": "GitHub 專案倉庫",
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
    st.session_state.active_page = "simulator"

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
    [data-testid="stAppViewContainer"] svg,
    [data-testid="stSidebar"] svg,
    .st-emotion-cache-28gi3v {{
        color: {theme["text"]} !important;
        fill: currentColor !important;
    }}
    div.stButton svg,
    div[data-testid="stDownloadButton"] svg,
    button[data-testid^="stBaseButton"] svg {{
        color: inherit !important;
        fill: currentColor !important;
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
    .metric-explain h4 {{
        margin: 0 0 0.55rem 0;
        font-size: 1.03rem;
        color: {theme["text"]};
    }}
    .metric-explain p {{
        margin: 0.45rem 0;
    }}
    .equation-card code {{
        display: block;
        margin-top: 0.35rem;
        font-size: 1.12rem;
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
    .page-intro {{
        color: {theme["muted"]};
        max-width: 860px;
        margin: 0.15rem 0 1.05rem;
        line-height: 1.65;
        font-size: 0.96rem;
    }}
    .insight-grid {{
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 0.85rem;
        margin: 0.85rem 0 1rem;
    }}
    .insight-card {{
        background: {theme["panel"]};
        border: 1px solid {theme["border"]};
        border-radius: 8px;
        padding: 1rem 1.05rem;
        color: {theme["text"]};
        min-height: 132px;
    }}
    .insight-card h4 {{
        margin: 0 0 0.5rem 0;
        font-size: 1.02rem;
        color: {theme["text"]};
    }}
    .insight-card p {{
        margin: 0;
        color: {theme["muted"]};
        line-height: 1.62;
    }}
    .source-summary {{
        display: grid;
        grid-template-columns: 1.1fr 0.9fr;
        gap: 0.85rem;
        margin: 0.9rem 0 1rem;
    }}
    .source-summary .source-card {{
        margin: 0;
    }}
    .source-meta {{
        display: flex;
        flex-wrap: wrap;
        gap: 0.5rem;
        margin-top: 0.7rem;
    }}
    .source-pill {{
        display: inline-flex;
        align-items: center;
        border: 1px solid {theme["border"]};
        border-radius: 999px;
        padding: 0.32rem 0.62rem;
        color: {theme["text"]};
        background: {theme["panel_alt"]};
        font-size: 0.86rem;
        font-weight: 650;
    }}
    .reference-list {{
        display: grid;
        gap: 0.55rem;
        margin-top: 0.6rem;
    }}
    .reference-item {{
        border: 1px solid {theme["border"]};
        border-radius: 8px;
        padding: 0.7rem 0.8rem;
        background: {theme["panel_alt"]};
    }}
    .reference-item strong {{
        display: block;
        color: {theme["text"]};
        margin-bottom: 0.25rem;
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
    .section-heading {{
        display: flex;
        align-items: center;
        gap: 0.55rem;
        margin: 1.1rem 0 0.45rem;
    }}
    .section-heading h4 {{
        margin: 0;
        font-size: 1.03rem;
        color: {theme["text"]};
    }}
    .info-tip {{
        position: relative;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 22px;
        height: 22px;
        border-radius: 999px;
        border: 1px solid {theme["button_border"]};
        color: {theme["accent"]};
        background: {theme["panel_alt"]};
        font-size: 0.82rem;
        font-weight: 800;
        cursor: help;
    }}
    .info-tip-text {{
        position: absolute;
        left: 0;
        top: 28px;
        z-index: 50;
        display: none;
        min-width: 280px;
        max-width: 420px;
        padding: 0.75rem 0.85rem;
        border-radius: 8px;
        border: 1px solid {theme["border"]};
        background: {theme["panel"]};
        color: {theme["text"]};
        box-shadow: 0 14px 34px rgba(15, 23, 42, 0.18);
        line-height: 1.55;
        font-weight: 500;
    }}
    .info-tip:hover .info-tip-text,
    .info-tip:focus .info-tip-text {{
        display: block;
    }}
    .download-panel {{
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 1rem;
        background: {theme["panel"]};
        border: 1px solid {theme["border"]};
        border-radius: 8px;
        padding: 0.95rem 1rem;
        margin: 0.75rem 0 0.8rem;
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
    .download-panel .file-icon {{
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 38px;
        height: 38px;
        border-radius: 8px;
        background: {theme["note_bg"]};
        color: {theme["accent"]};
        font-weight: 800;
        flex: 0 0 auto;
    }}
    .download-copy {{
        display: flex;
        align-items: center;
        gap: 0.8rem;
        min-width: 0;
    }}
    @media (max-width: 760px) {{
        .insight-grid,
        .source-summary {{
            grid-template-columns: 1fr;
        }}
        .download-panel {{
            align-items: flex-start;
            flex-direction: column;
        }}
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
    "demo": {"value": 42, "label": {"en": "Common reference - 42", "zh": "常用參考 - 42"}},
    "year": {"value": 2024, "label": {"en": "Year marker - 2024", "zh": "年度示範 - 2024"}},
    "alternate": {"value": 1001, "label": {"en": "Alternate sample - 1001", "zh": "替代樣本 - 1001"}},
}


def model_label(model_key: str) -> str:
    return t(MODEL_OPTIONS.get(model_key, MODEL_OPTIONS["ols"])["label_key"])


def model_reading(model_key: str) -> str:
    return t(MODEL_OPTIONS.get(model_key, MODEL_OPTIONS["ols"])["reading_key"])


def preset_label(presets: dict, preset_key: str) -> str:
    return presets[preset_key]["label"][st.session_state.locale]


def localized_key(base_key: str) -> str:
    return f"{base_key}_{st.session_state.locale}"


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
    cols = st.columns(3)
    cols[0].metric(t("r2"), f"{result['r2']:.3f}")
    cols[1].metric(t("rmse"), f"{result['rmse']:.2f}")
    cols[2].metric(t("mae"), f"{result['mae']:.2f}")


def metric_explanation(result: dict) -> None:
    st.markdown(f'<div class="metric-explain">{metric_explanation_html(result)}</div>', unsafe_allow_html=True)


def metric_explanation_html(result: dict) -> str:
    if result["r2"] >= 0.8:
        r2_text = t("metric_r2_strong")
    elif result["r2"] >= 0.5:
        r2_text = t("metric_r2_ok")
    else:
        r2_text = t("metric_r2_weak")

    return (
        f"<p><strong>{t('metric_status_title')}:</strong> {r2_text}<br>"
        f"{t('metric_error_note')}<br>"
        f"{t('metric_outlier_note')}</p>"
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


def model_profile_html(result: dict) -> str:
    note = t("coefficient_note") if result["model_key"] != "ols" else ""
    return (
        f"<p><strong>{t('model_profile_title')}:</strong> {model_reading(result['model_key'])}"
        f"{f'<br>{note}' if note else ''}</p>"
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


def model_equation_html(result: dict, feature_names: list[str], target_name: str = "y") -> str:
    note = t("equation_note") if result["model_key"] != "ols" else ""
    return (
        f"<p><strong>{t('equation')}:</strong>"
        f"<code>{equation_text(result, feature_names, target_name)}</code>"
        f"{f'<br>{note}' if note else ''}</p>"
    )


def interpretation_panel(result: dict, feature_names: list[str], target_name: str = "y") -> None:
    st.markdown(
        f"""
        <div class="metric-explain equation-card">
            <h4>{t("interpretation_title")}</h4>
            {metric_explanation_html(result)}
            {model_profile_html(result)}
            {model_equation_html(result, feature_names, target_name)}
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


def residual_heading_with_tooltip(title_key: str = "top_outliers") -> None:
    st.markdown(
        f"""
        <div class="section-heading">
            <h4>{t(title_key)}</h4>
            <span class="info-tip" tabindex="0">?
                <span class="info-tip-text">{t("outlier_plain_note")}<br>{t("outlier_scope_note")}</span>
            </span>
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
    options = ["simulator", "aqi", "data", "report"]
    labels = {
        "simulator": t("nav_lab"),
        "aqi": t("nav_aqi"),
        "report": t("nav_report"),
        "data": t("nav_data"),
    }
    icons = {
        "simulator": ":material/science:",
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
        return f"""

本報告採用 CRISP-DM（跨行業數據挖掘標準流程）框架，系統性分析中部地區空氣品質指標 (AQI) 線性迴歸建模原型與異常觀測篩選工具。

本專案旨在提供一個科學且可複現的數據驅動流程，以輔助環境監測決策與資源調度。

## 1. 業務理解 (Business Understanding)
*   **業務背景與組織痛點**：空氣品質指標 (AQI) 的異常數值往往與突發性本地污染、短暫天候效應或監測設備損壞密切相關。面對海量測站數據，傳統人工巡檢極為耗時，容易錯過關鍵異常事件。
*   **目標受眾 (Target Audience)**：主要受眾為地方環境管理與公共衛生分析人員；次要受眾為社區監測設備維護人員與環保倡議小組。
*   **核心決策問題 (Core Decision Question)**：在中部測站（台中、彰化、南投）的空污觀測紀錄中，哪些觀測值嚴重偏離了通常的污染物關聯規律，需要被優先進行複查？
*   **專案核心目標**：建立線性迴歸模型作為**決策支持系統 (DSS)** 中的異常檢索引擎。透過污染物規律擬合，篩選出「高絕對殘差」的疑似異常觀測點（實際值與預測值偏差極大者），供環境監測部門優先複查，以迅速定位設備損壞、局部污染源突增或特殊天氣效應。
*   **成功指標 (Success Metrics)**：
    *   模型可解釋性：基準 Ordinary Least Squares (OLS) 的 R-squared 與誤差指標需保持在合理水平以提供可解釋的污染物規律。
    *   業務實用性：能將前 10 筆高絕對殘差的疑似異常觀測自動彙整，顯著降低業務分析人員全面巡檢的時間成本。

## 2. 數據理解 (Data Understanding)
*   **數據來源、範疇與交代 (Data Sources)**：
    *   專案內建資料集：採用託管於 GitHub 專案倉庫目錄下的真實世界資料集 `data/{active_file}`，共計 {SAMPLE_AQI_ROWS:,} 筆。考量地理特徵與效能限制，地理範疇控制在台中、彰化及南投測站數據。
    *   官方歷史參考：環境部官方歷史監測資料庫 `MOENV AQX_P_488`，為歷史大數據比對的主體。
    *   即時觀測參考：環境部官方即時指標 `MOENV AQX_P_432`，作即時動態比對。
    *   封存替代方案：`Kaggle Taiwan Air Quality Data (2016~2024)`，用於離線可重現性分析。
*   **變數分析**：
    *   預測目標 (Target)：如 AQI 或 PM2.5 等單一數值目標。
    *   預測特徵 (Features)：包括 PM2.5、PM10、O3、NO2、CO、SO2 及風速等。
    *   關聯邏輯：例如，臭氧 (O3) 的濃度通常與光化學反應及氣溫有極大關聯；PM2.5 與 PM10 具有高度共線性，但若在特定測站出現兩者比例嚴重失衡，即可能為局部的特殊污染或儀器故障。
*   **數據限制**：測站數據包含設備漂移、突發局部干擾等高頻噪音。線性模型難以完全配適非線性特徵，而這正是該模型作為異常檢測手段的關鍵：模型無法說明的「高殘差」即為最值得人為介入複查的「潛在異常」。

## 3. 數據準備 (Data Preparation)
*   **數據清洗與格式對齊**：統一並正規化各污染物欄位名稱，將欄位轉為數值型態，並剔除包含缺失值 (NaN) 的觀測列以確保模型訓練品質。
*   **數據篩選**：支援使用者依區域（中部、台中、彰化、南投）進行子集篩選，以排除不同縣市微氣候或地理屏障帶來的分析偏差。
*   **特徵工程與標準化**：針對 Ridge、Lasso 及 ElasticNet 等正則化模型，在擬合前先對特徵進行標準化 (StandardScaler)，確保各污染物的係數權重可直接在同一尺度上進行方向與強度的比較。

## 4. 建立模型 (Modeling)
本系統提供四種線性迴歸演算法，用以平衡模型解釋力與泛化能力：
*   **OLS (普通最小平方法)**：作為最直觀、最容易向業務主管解釋的基準模型。
*   **Ridge 迴歸 (脊迴歸)**：引入 L2 懲罰項，在污染物欄位高度相關（共線性）時，能有效穩定模型係數。
*   **Lasso 迴歸 (套索迴歸)**：引入 L1 懲罰項，具有特徵篩選功能，會將微弱的特徵係數直接壓縮為 0，簡化模型結構。
*   **ElasticNet 迴歸**：結合 L1 與 L2 懲罰，適合處理共線性特徵且需適度特徵刪減的情境。

## 5. 模型評估 (Evaluation)
*   **評估指標定義**：
    *   **R-squared (判定係數)**：衡量模型能解釋目標變數變異量的比例。
    *   **RMSE (均方根誤差)**：反映預測值與實際值的平均偏差，因平方懲罰而對大誤差極為敏感。
    *   **MAE (平均絕對誤差)**：呈現預測誤差的直觀物理意義（平均猜錯多少個 AQI 單位）。
*   **殘差分析 (異常判定核心)**：計算 Residual = Actual - Predicted。絕對殘差值最大的前 10 筆觀測代表該時間點的空氣品質特徵嚴重偏離了區域大環境的平均統計規律（例如在低 NO2、CO 的情況下 PM2.5 卻異常暴增），極具人工複查價值。

## 6. 模型部署與治理 (Deployment & Governance)
*   **雲端部署與數據下載性**：
    *   原型系統部署於雲端平台（如 Streamlit Community Cloud），並將 `data/` 目錄與程式碼一同打包。此做法確保了雲端環境下能直接進行高效的本機 CSV 讀取與建模，並且允許使用者隨時下載完整的真實世界 CSV 資料（亦可透過 GitHub 原始碼目錄下載）。
    *   提供決策者一鍵切換特徵、目標、地區與模型的自助式分析平台，並支持乾淨 CSV 數據及本分析報告下載。
*   **分析價值與決策輔助 (Business Value)**：本專案的核心價值在於**異常事件的快速篩選與工作流優化**。它能將最值得注意的高絕對殘差觀測自動排在最前列（前 10 名），大幅縮短監測人員篩查數據的時間，幫助相關團隊提早定位硬體漂移或地方污染事件。
*   **業務治理限制與防線 (Governance Limits)**：
    *   基準迴歸模型僅作輔助篩選（殘差排序與模型判讀），**不等於官方預報服務**。
    *   正式的空氣品質裁決、健康建議與風險通報，仍必須結合監測人員之專業領域審查 (Domain Review) 與政府官方發布之正式指標。
    *   後續建議引入資料版本控制與測站品質標籤，以進一步提升決策DSS的可靠度。
"""

    return f"""

This report utilizes the CRISP-DM (Cross-Industry Standard Process for Data Mining) framework to systematically analyze the Central Taiwan Air Quality Index (AQI) linear regression modeling prototype and residual anomaly screening tool. 

This project provides a scientific and reproducible data-driven pipeline to optimize monitoring decisions.

## 1. Business Understanding
*   **Business Background and Pain Points**: Anomalies in the Air Quality Index (AQI) are typically caused by localized sudden emission sources, transient weather phenomena, or monitoring equipment deterioration. In the face of massive station observations, manual checking is extremely labor-intensive, often leading to delayed reactions.
*   **Target Audience**: Primary: local environmental and public health analysts/decision-makers; Secondary: community monitoring maintenance teams and environmental advocacy groups.
*   **Core Decision Question**: Which station-time observations in Central Taiwan (Taichung, Changhua, Nantou) deviate from normal pollutant correlation patterns and should be prioritized for review?
*   **Core Goal**: Develop a linear regression model as an anomaly screening engine for a **Decision Support System (DSS)**. By fitting pollutant correlation patterns, it isolates observations with high absolute residuals to help analysts quickly identify sensor faults or emission spikes.
*   **Success Metrics**:
    *   Model Interpretability: The baseline Ordinary Least Squares (OLS) model's R-squared and error metrics must remain stable to guarantee explainable pollutant relationships.
    *   Operational Efficiency: Automatically rank and present the top 10 absolute residual observations, dramatically cutting down the time analysts spend on manual screening.

## 2. Data Understanding
*   **Data Sources and Disclosures**:
    *   Bundled Dataset: Bundled real-world dataset `data/{active_file}` (totaling {SAMPLE_AQI_ROWS:,} records) hosted in the project repository on GitHub. Due to geography and performance constraints, the spatial coverage is focused on Taichung, Changhua, and Nantou monitoring stations.
    *   Official Historical Benchmark: The Ministry of Environment's `MOENV AQX_P_488` historical dataset, serving as the main benchmark.
    *   Real-time Reference: The `MOENV AQX_P_432` real-time API, referenced for current measurements.
    *   Archival Option: `Kaggle Taiwan Air Quality Data (2016~2024)`, utilized for offline reproducibility.
*   **Variable Analysis**:
    *   Target Variable: e.g., AQI or PM2.5.
    *   Feature Variables: Includes PM2.5, PM10, O3, NO2, CO, SO2, wind speed, etc.
    *   Correlation Logic: For instance, Ozone (O3) typically correlates with photochemical reactions and temperature; PM2.5 and PM10 are collinear, but a sudden shift in their ratio at a specific station can indicate unique pollution spikes or sensor drift.
*   **Data Constraints**: Station measurements contain high-frequency noise from instrument wear or temporary local events. Linear regression cannot capture all non-linear interactions; however, this is exactly the model's value as an anomaly detector: the high residual that the model fails to explain represents the "potential anomaly" most worthy of manual review.

## 3. Data Preparation
*   **Data Cleaning**: Standardize and normalize column names, convert fields to numeric types, and remove rows containing missing values (NaN) to ensure the quality of model training.
*   **Data Filtering**: Allow users to filter subsets by region (Central, Taichung, Changhua, Nantou) to eliminate analytical biases brought by microclimates or geographical barriers in different counties.
*   **Feature Engineering and Standardization**: For regularized models like Ridge, Lasso, and ElasticNet, scale features using `StandardScaler` prior to fitting. This ensures that the coefficients of various pollutants can be directly compared on the same scale for direction and magnitude.

## 4. Modeling
This system provides four linear regression algorithms to balance model interpretability and generalization ability:
*   **OLS (Ordinary Least Squares)**: Served as the most intuitive baseline model, which is easy to explain to business executives.
*   **Ridge Regression**: Introduces an L2 penalty, which effectively stabilizes model coefficients when pollutant fields are highly correlated (multicollinearity).
*   **Lasso Regression**: Introduces an L1 penalty, which performs automatic feature selection by shrinking weak feature coefficients to exactly zero, simplifying the model structure.
*   **ElasticNet Regression**: Combines L1 and L2 regularization, suitable for scenarios with collinear features where moderate feature elimination is desired.

## 5. Evaluation
*   **Evaluation Metrics Definition**:
    *   **R-squared (Coefficient of Determination)**: Measures the proportion of variance in the target variable explained by the model.
    *   **RMSE (Root Mean Squared Error)**: Reflects the average deviation between predicted and actual values, highly sensitive to large errors due to squaring.
    *   **MAE (Mean Absolute Error)**: Represents the straightforward physical meaning of the prediction error (the average unit of AQI guessed incorrectly).
*   **Residual Analysis (Anomaly Detection Core)**: Calculated as Residual = Actual - Predicted. The top 10 observations with the largest absolute residuals represent times when the air quality characteristics deviated significantly from the regional average statistical pattern (e.g., a sudden spike in PM2.5 despite low NO2 and CO), indicating high value for manual inspection.

## 6. Deployment & Governance
*   **Cloud Deployment and Data Access**:
    *   The prototype is deployed in a cloud environment (e.g., Streamlit Community Cloud) with the `data/` directory bundled alongside the codebase. This ensures highly efficient local CSV reads and modeling under the cloud setup while allowing users to download the complete real-world CSV dataset at any time (also accessible via the GitHub repository directory).
    *   Provides decision-makers with a self-service analysis platform to toggle features, targets, regions, and models, with support for downloading clean CSV data and this report.
*   **Decision Value**: The primary value lies in **rapid anomaly indexing and workflow optimization**. It ranks the top 10 absolute residual observations automatically, reducing manual inspection time and helping teams catch sensor drifts or local spikes earlier.
*   **Governance Limits and Warnings**:
    *   The anomalous observations flagged by this system serve only as "inspection cues" and **do not constitute an official air quality forecast service**.
    *   Official air quality rulings, health advice, and risk communication must still combine professional Domain Review by monitoring experts with official government releases.
    *   Future recommendations include introducing data version control and station quality labels to further enhance the reliability of the decision DSS.
"""


def download_action(panel_id: str, kind: str, label: str, hint: str, file_name: str, data: bytes, mime: str) -> None:
    left, right = st.columns([3.2, 1])
    icon = "CSV" if kind == "csv" else "MD"
    with left:
        st.markdown(
            f"""
            <div class="download-panel">
                <div class="download-copy">
                    <span class="file-icon">{icon}</span>
                    <div>
                        <div class="file-name">{file_name}</div>
                        <div class="file-meta">{hint}</div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with right:
        st.download_button(
            label,
            data=data,
            file_name=file_name,
            mime=mime,
            width="stretch",
            key=f"download_{kind}_{panel_id}",
        )


def source_download_panel(panel_id: str, include_report: bool = True) -> None:
    if not AQI_DATA_AVAILABLE:
        st.error(t("no_active_data"))
        return
    sample_bytes = SAMPLE_AQI_PATH.read_bytes()
    report_text = report_markdown(st.session_state.locale)
    report_file_name = f"business_analysis_report_{st.session_state.locale}.md"
    sample_file_name = SAMPLE_AQI_PATH.name
    limitation_note = ""
    if AQI_DATA_IS_LOCAL_HISTORICAL:
        limitation_note = f"<p>{t('local_historical_warning')}</p>"

    file_meta = (
        f"{data_label()} · {SAMPLE_AQI_ROWS:,} 筆 · {t('download_hint_data')}"
        if st.session_state.locale == "zh"
        else f"{data_label()} · {SAMPLE_AQI_ROWS:,} rows · {t('download_hint_data')}"
    )
    if limitation_note:
        st.warning(t("local_historical_warning"))
    download_action(
        panel_id,
        "csv",
        t("download_data"),
        file_meta,
        sample_file_name,
        sample_bytes,
        "text/csv",
    )
    if include_report:
        download_action(
            panel_id,
            "report",
            t("download_report"),
            t("download_hint_report"),
            report_file_name,
            report_text.encode("utf-8"),
            "text/markdown",
        )


def data_source_tab() -> None:
    if not AQI_DATA_AVAILABLE:
        st.error(t("no_active_data"))
        return
    sample_file_name = SAMPLE_AQI_PATH.name
    st.subheader(t("source_title"))
    st.markdown(
        f"""
        <div class="source-card">
            <h4>{t("source_adopt_title")}</h4>
            <p>{t("source_adopt_body")}</p>
            <div class="source-meta" style="margin-top: 1rem;">
                <span class="source-pill">data/{sample_file_name}</span>
                <span class="source-pill">{SAMPLE_AQI_ROWS:,} rows</span>
                <span class="source-pill">Taichung / Changhua / Nantou</span>
            </div>
        </div>
        <div class="source-card">
            <h4>{t("source_quality_title")}</h4>
            <p>{t("source_quality_body")}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    source_download_panel("source", include_report=False)


def crisp_report_tab() -> None:
    report_text = report_markdown(st.session_state.locale)
    report_file_name = f"crisp_dm_analysis_report_{st.session_state.locale}.md"
    st.subheader(t("crisp_title"))
    st.markdown(report_text)
    download_action(
        "crisp_only",
        "report",
        t("download_report"),
        t("download_hint_report"),
        report_file_name,
        report_text.encode("utf-8"),
        "text/markdown",
    )
    st.markdown('</div>', unsafe_allow_html=True)


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


def plain_lab_reading(n: int, a: float, b: float, var: float, seed: int) -> None:
    stats = t("lab_plain_stats")
    stats_html = f"<p>{stats.format(n=n, a=a, b=b, var=var, seed=seed)}</p>" if stats else ""
    st.markdown(
        f"""
        <div class="plain-reading">
            <h4>{t("lab_plain_title")}</h4>
            <p>{t("lab_plain_body")}</p>
            {stats_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def beginner_learning_panel() -> None:
    st.markdown(
        f"""
        <div class="source-card">
            <h4>{t("learning_title")}</h4>
            <p>{t("learning_body")}</p>
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
            current_model = st.session_state.get("synthetic_model_choice", "ols")
            model_key = st.selectbox(
                t("model_choice"),
                available_models,
                index=available_models.index(current_model) if current_model in available_models else 0,
                format_func=model_label,
                key=localized_key("synthetic_model_choice"),
            )
            st.session_state.synthetic_model_choice = model_key
            n = st.slider(t("sample_size"), 50, 1000, 300, 50, key="synthetic_n")
            a = st.slider(t("slope"), -50.0, 50.0, 8.0, 0.5, key="synthetic_a")
            b = st.slider(t("intercept"), -100.0, 100.0, 40.0, 1.0, key="synthetic_b")
            current_var_preset = st.session_state.get("synthetic_var_preset", "medium")
            selected_var_preset = st.selectbox(
                t("variance_preset"),
                list(NOISE_PRESETS.keys()),
                index=list(NOISE_PRESETS.keys()).index(current_var_preset)
                if current_var_preset in NOISE_PRESETS
                else 2,
                format_func=lambda key: preset_label(NOISE_PRESETS, key),
                key=localized_key("synthetic_var_preset"),
            )
            if selected_var_preset != st.session_state.get("synthetic_var_preset"):
                st.session_state.synthetic_var_preset = selected_var_preset
                sync_noise_preset()
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
            current_seed_preset = st.session_state.get("synthetic_seed_preset", "demo")
            selected_seed_preset = st.selectbox(
                t("seed_preset"),
                list(SEED_PRESETS.keys()),
                index=list(SEED_PRESETS.keys()).index(current_seed_preset)
                if current_seed_preset in SEED_PRESETS
                else 2,
                format_func=lambda key: preset_label(SEED_PRESETS, key),
                key=localized_key("synthetic_seed_preset"),
            )
            if selected_seed_preset != st.session_state.get("synthetic_seed_preset"):
                st.session_state.synthetic_seed_preset = selected_seed_preset
                sync_seed_preset()
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

    plain_lab_reading(n, a, b, var, seed)
    show_metrics(result)
    interpretation_panel(result, ["x"], "y")

    residual_heading_with_tooltip()
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
            region_options = list(REGION_FILTERS.keys())
            current_region = st.session_state.get("aqi_region", "central")
            region_key = st.selectbox(
                t("region"),
                region_options,
                index=region_options.index(current_region) if current_region in region_options else 0,
                format_func=region_label,
                key=localized_key("aqi_region"),
            )
            st.session_state.aqi_region = region_key
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
            current_model = st.session_state.get("aqi_model_choice", "ols")
            model_key = st.selectbox(
                t("model_choice"),
                available_models,
                index=available_models.index(current_model) if current_model in available_models else 0,
                format_func=model_label,
                key=localized_key("aqi_model_choice"),
            )
            st.session_state.aqi_model_choice = model_key
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
    filtered_row_count = len(df)

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
            line={"dash": "dash", "color": theme["plot_axis"]},
        )
    )
    apply_plot_style(fig, height=480, title=t("actual_predicted"))
    st.plotly_chart(fig, width="stretch", config=plot_config)

    plain_aqi_reading(model_df, target_col)
    show_metrics(result)
    interpretation_panel(result, selected_features, target_col)

    with st.expander(t("coefficients")):
        if result["model_key"] != "ols":
            st.caption(t("coefficient_note"))
        show_dataframe(coefficient_df, width="stretch", hide_index=True)

    with st.expander(t("model_comparison")):
        st.markdown(t("model_comparison_note"))
        show_dataframe(
            compare_linear_models(model_df[selected_features], model_df[target_col]),
            width="stretch",
            hide_index=True,
        )

    residual_heading_with_tooltip()
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
    st.markdown(dataset_status_html(filtered_row_count, len(model_df)), unsafe_allow_html=True)


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

if active_page == "simulator":
    synthetic_lab()
elif active_page == "aqi":
    aqi_case()
elif active_page == "data":
    data_source_tab()
else:
    crisp_report_tab()

if not st.session_state.sidebar_collapsed:
    with st.sidebar:
        st.markdown("---")
        st.link_button(
            t("github_repo_btn", "GitHub Repository"),
            "https://github.com/dec591nyc/Linear-Regression-Practice",
            icon="🐙",
            use_container_width=True,
        )
