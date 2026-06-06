from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

try:
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import r2_score
except Exception:  # pragma: no cover - fallback keeps the demo readable without sklearn.
    LinearRegression = None


PROJECT_ROOT = Path(__file__).resolve().parent
SAMPLE_AQI_PATH = PROJECT_ROOT / "data" / "central_taiwan_aqi_sample.csv"
CENTRAL_COUNTIES = ["臺中市", "台中市", "彰化縣"]

TEXT = {
    "en": {
        "page_title": "Linear Regression Practice",
        "hero_title": "Linear Regression Practice",
        "hero_subtitle": "A compact regression dashboard for testing baseline modeling, residual ranking, and central Taiwan air-quality use cases.",
        "theme": "Theme",
        "language": "Language",
        "light": "Light",
        "dark": "Dark",
        "regression_tab": "Regression Sandbox",
        "aqi_tab": "Central Taiwan AQI",
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
        "equation": "Estimated equation",
        "generated_data": "Generated data",
        "outliers": "Top 10 residual outliers",
        "true_line": "True line",
        "regression_line": "Regression line",
        "top_outliers": "Top residual observations",
        "aqi_title": "Central Taiwan AQI Case",
        "aqi_note": "This case uses Taichung and Changhua air-quality readings to predict a numeric target and surface observations that the baseline model cannot explain well.",
        "advanced_data": "Advanced data source",
        "use_external": "Use uploaded AQI CSV",
        "upload_label": "Upload Kaggle or MOENV AQI CSV",
        "upload_help": "Optional. If no file is uploaded, the app uses the bundled central Taiwan AQI sample.",
        "target": "Target variable",
        "features": "Feature variables",
        "source_run": "Data source in this run",
        "complete_rows": "Complete rows used",
        "bundled_sample": "Bundled central Taiwan sample",
        "uploaded_csv": "Uploaded CSV",
        "coefficients": "Coefficients",
        "actual_predicted": "Actual vs Predicted",
        "perfect_prediction": "Perfect prediction",
        "source_title": "Data Source Evaluation",
        "source_body": """
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
    },
    "zh": {
        "page_title": "線性迴歸實作",
        "hero_title": "線性迴歸實作",
        "hero_subtitle": "以台中、彰化空氣品質為情境，展示基礎迴歸建模、殘差排序與異常觀測判讀。",
        "theme": "主題",
        "language": "語言",
        "light": "淺色",
        "dark": "深色",
        "regression_tab": "迴歸模擬器",
        "aqi_tab": "中彰 AQI 案例",
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
        "equation": "估計方程式",
        "generated_data": "模擬資料",
        "outliers": "殘差前 10 名",
        "true_line": "真實線",
        "regression_line": "迴歸線",
        "top_outliers": "殘差最高觀測",
        "aqi_title": "中彰 AQI 實際案例",
        "aqi_note": "此區以台中與彰化空氣品質資料預測數值目標，並找出基礎模型較難解釋的污染觀測。",
        "advanced_data": "進階資料來源",
        "use_external": "使用外部 AQI CSV",
        "upload_label": "上傳 Kaggle 或環境部 AQI CSV",
        "upload_help": "選填。未上傳時會使用內建中彰 AQI sample。",
        "target": "預測目標",
        "features": "特徵欄位",
        "source_run": "本次資料來源",
        "complete_rows": "完整資料筆數",
        "bundled_sample": "內建中彰 sample",
        "uploaded_csv": "上傳 CSV",
        "coefficients": "模型係數",
        "actual_predicted": "實際值與預測值",
        "perfect_prediction": "理想預測線",
        "source_title": "資料來源評估",
        "source_body": """
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

with st.sidebar:
    st.session_state.locale = st.radio(
        "Language / 語言",
        ["zh", "en"],
        index=["zh", "en"].index(st.session_state.locale),
        horizontal=True,
        format_func=lambda value: "繁中" if value == "zh" else "English",
    )

t = TEXT[st.session_state.locale].get

with st.sidebar:
    st.session_state.theme = st.radio(
        t("theme"),
        ["light", "dark"],
        index=["light", "dark"].index(st.session_state.theme),
        horizontal=True,
        format_func=lambda value: t(value),
    )

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
        "control_bg": "#111827",
        "control_text": "#f8fafc",
        "control_muted": "#a8b3c7",
    },
}
theme = THEMES[st.session_state.theme]
plot_template = "plotly_dark" if st.session_state.theme == "dark" else "plotly_white"

st.markdown(
    f"""
    <style>
    [data-testid="stToolbar"], .stAppToolbar, #MainMenu, footer {{
        display: none !important;
        visibility: hidden !important;
        height: 0 !important;
    }}
    html, body, [data-testid="stAppViewContainer"] {{
        background: {theme["bg"]};
        color: {theme["text"]};
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
        background: transparent;
    }}
    [data-testid="stSidebar"] {{
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
    </style>
    """,
    unsafe_allow_html=True,
)


def fit_linear_model(feature_df: pd.DataFrame, target: pd.Series):
    x = feature_df.to_numpy(dtype=float)
    y = target.to_numpy(dtype=float)

    if LinearRegression is not None:
        model = LinearRegression()
        model.fit(x, y)
        y_pred = model.predict(x)
        coefficients = model.coef_
        intercept = float(model.intercept_)
        model_name = "scikit-learn LinearRegression"
    else:
        x_design = np.column_stack([np.ones(len(x)), x])
        params, *_ = np.linalg.lstsq(x_design, y, rcond=None)
        intercept = float(params[0])
        coefficients = params[1:]
        y_pred = x_design @ params
        model_name = "NumPy least-squares fallback"

    residuals = y - y_pred
    return {
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
    fig.update_layout(
        height=500,
        xaxis_title="x",
        yaxis_title="y",
        legend_orientation="h",
        template=plot_template,
        paper_bgcolor=theme["plot_bg"],
        plot_bgcolor=theme["plot_bg"],
        font={"color": theme["text"], "size": 13},
    )
    st.plotly_chart(fig, use_container_width=True)

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

    uploaded = None
    with st.sidebar.expander(t("advanced_data"), expanded=False):
        uploaded = st.file_uploader(t("upload_label"), type=["csv"], help=t("upload_help"))

    if uploaded is not None:
        raw_df = pd.read_csv(uploaded)
        source_label = t("uploaded_csv")
    else:
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

    model_df = df[[target_col, *selected_features]].dropna().copy()
    if len(model_df) < 10:
        st.warning(t("too_few_rows"))
        return

    result = fit_linear_model(model_df[selected_features], model_df[target_col])
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
        fig.update_layout(
            height=440,
            template=plot_template,
            paper_bgcolor=theme["plot_bg"],
            plot_bgcolor=theme["plot_bg"],
            font={"color": theme["text"], "size": 13},
        )
        st.plotly_chart(fig, use_container_width=True)
    with right:
        st.markdown(f"#### {t('coefficients')}")
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

tab_synthetic, tab_aqi, tab_sources = st.tabs(
    [t("regression_tab"), t("aqi_tab"), t("source_tab")]
)

with tab_synthetic:
    synthetic_lab()

with tab_aqi:
    aqi_case()

with tab_sources:
    st.subheader(t("source_title"))
    st.markdown(t("source_body"))
