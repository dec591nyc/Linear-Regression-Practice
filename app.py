from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

try:
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
except Exception:  # pragma: no cover - fallback keeps the demo readable without sklearn.
    LinearRegression = None


PROJECT_ROOT = Path(__file__).resolve().parent
SAMPLE_AQI_PATH = PROJECT_ROOT / "data" / "central_taiwan_aqi_sample.csv"
CENTRAL_COUNTIES = ["臺中市", "台中市", "彰化縣"]


st.set_page_config(
    page_title="HW4 Linear Regression AQI Case",
    page_icon="LR",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown(
    """
    <style>
    .block-container {
        max-width: 1180px;
        padding-top: 2rem;
    }
    .main-title {
        font-size: 2.25rem;
        font-weight: 750;
        margin-bottom: 0.35rem;
    }
    .subtitle {
        color: #5f6673;
        margin-bottom: 1.4rem;
        line-height: 1.7;
    }
    .note {
        border-left: 4px solid #2563eb;
        background: #f8fafc;
        padding: 0.8rem 1rem;
        margin: 0.8rem 0 1.2rem;
        color: #344054;
    }
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
    cols[0].metric("R-squared", f"{result['r2']:.3f}")
    cols[1].metric("RMSE", f"{result['rmse']:.2f}")
    cols[2].metric("MAE", f"{result['mae']:.2f}")
    cols[3].metric("Model", result["model_name"])


def synthetic_lab() -> None:
    st.subheader("Synthetic Linear Regression Lab")
    st.markdown(
        '<div class="note">此模式保留課堂作業核心：用 n、a、b、var 生成數值資料，'
        "套用線性迴歸，並用絕對殘差找出前 10 個離群點。</div>",
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.markdown("### Synthetic Parameters")
        n = st.slider("Sample size (n)", 50, 1000, 300, 50)
        a = st.slider("True slope (a)", -50.0, 50.0, 8.0, 0.5)
        b = st.slider("True intercept (b)", -100.0, 100.0, 40.0, 1.0)
        var = st.number_input("Noise variance (var)", 0.0, 100000.0, 10000.0, 500.0)
        seed = st.number_input("Random seed", 0, 999999, 42, 1)

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

    st.markdown(
        f"Estimated equation: `y = {result['coefficients'][0]:.4f} * x + {result['intercept']:.4f}`"
    )

    x_line = np.linspace(df["x"].min(), df["x"].max(), 200)
    fitted_line = result["coefficients"][0] * x_line + result["intercept"]
    true_line = a * x_line + b

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["x"], y=df["y"], mode="markers", name="Generated data"))
    fig.add_trace(
        go.Scatter(
            x=df.loc[df["is_outlier"], "x"],
            y=df.loc[df["is_outlier"], "y"],
            mode="markers",
            name="Top 10 residual outliers",
            marker={"size": 12, "color": "#ef4444", "symbol": "circle-open", "line": {"width": 2}},
        )
    )
    fig.add_trace(go.Scatter(x=x_line, y=true_line, mode="lines", name="True line", line={"dash": "dash"}))
    fig.add_trace(go.Scatter(x=x_line, y=fitted_line, mode="lines", name="Regression line"))
    fig.update_layout(height=520, xaxis_title="x", yaxis_title="y", legend_orientation="h")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Top 10 residual outliers")
    st.dataframe(
        df.sort_values("abs_residual", ascending=False)
        .head(10)[["rank", "x", "y", "predicted_y", "residual", "abs_residual"]],
        use_container_width=True,
    )


def aqi_case() -> None:
    st.subheader("Central Taiwan AQI Business Case")
    st.markdown(
        '<div class="note">此模式作為實際案例補充：以中部空氣品質資料預測 AQI 或 PM2.5，'
        "再用殘差排序找出模型難以解釋的污染異常觀測。正式資料可改用 Kaggle 整理版或環境部 API。</div>",
        unsafe_allow_html=True,
    )

    uploaded = st.file_uploader("Upload Kaggle or MOENV AQI CSV", type=["csv"])
    if uploaded is not None:
        raw_df = pd.read_csv(uploaded)
        source_label = "Uploaded CSV"
    else:
        raw_df = pd.read_csv(SAMPLE_AQI_PATH)
        source_label = "Bundled central Taiwan sample"

    df = normalize_aqi_columns(raw_df)
    if "county" in df.columns:
        df = df[df["county"].isin(CENTRAL_COUNTIES)].copy()

    numeric_cols = get_numeric_columns(df)
    if len(numeric_cols) < 2:
        st.error("AQI data needs at least two numeric columns after cleaning.")
        st.dataframe(df.head(20), use_container_width=True)
        return

    default_target = "aqi" if "aqi" in numeric_cols else numeric_cols[-1]
    target_col = st.selectbox("Target variable", numeric_cols, index=numeric_cols.index(default_target))
    feature_options = [col for col in numeric_cols if col != target_col]
    default_features = [
        col
        for col in ["pm25", "pm25_avg", "pm10", "pm10_avg", "o3", "no2", "co", "so2", "wind_speed"]
        if col in feature_options
    ]
    if not default_features:
        default_features = feature_options[: min(5, len(feature_options))]

    selected_features = st.multiselect("Feature variables", feature_options, default=default_features)
    if not selected_features:
        st.warning("Select at least one feature variable.")
        return

    model_df = df[[target_col, *selected_features]].dropna().copy()
    if len(model_df) < 10:
        st.warning("The selected columns have too few complete rows for a useful regression.")
        return

    result = fit_linear_model(model_df[selected_features], model_df[target_col])
    model_df["predicted"] = result["predictions"]
    model_df["residual"] = result["residuals"]
    model_df["abs_residual"] = np.abs(result["residuals"])
    model_df["rank"] = model_df["abs_residual"].rank(method="first", ascending=False).astype(int)

    display_df = df.loc[model_df.index].copy()
    for col in ["predicted", "residual", "abs_residual", "rank"]:
        display_df[col] = model_df[col]

    st.caption(f"Data source in this run: {source_label}. Complete rows used: {len(model_df)}.")
    show_metrics(result)

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
            title="Actual vs Predicted",
        )
        min_val = float(min(model_df["predicted"].min(), model_df[target_col].min()))
        max_val = float(max(model_df["predicted"].max(), model_df[target_col].max()))
        fig.add_trace(
            go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode="lines",
                name="Perfect prediction",
                line={"dash": "dash", "color": "#475569"},
            )
        )
        fig.update_layout(height=460)
        st.plotly_chart(fig, use_container_width=True)
    with right:
        st.markdown("#### Coefficients")
        st.dataframe(coefficient_df, use_container_width=True, hide_index=True)

    st.markdown("#### Top residual observations")
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


st.markdown('<div class="main-title">HW4 Linear Regression and Central Taiwan AQI Case</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">作業主線維持簡單線性迴歸與離群值偵測，'
    "並補上一個台中、彰化空氣品質的現實資料案例，讓模型結果能對應公共與商業決策。</div>",
    unsafe_allow_html=True,
)

tab_synthetic, tab_aqi, tab_sources = st.tabs(
    ["Synthetic Lab", "Central Taiwan AQI", "Source Evaluation"]
)

with tab_synthetic:
    synthetic_lab()

with tab_aqi:
    aqi_case()

with tab_sources:
    st.subheader("Data Source Evaluation")
    st.markdown(
        """
        - Kaggle `Taiwan Air Quality Index Data 2016~2024` is suitable for classroom work because it is already collected and shaped for machine-learning practice.
        - The official replacement source is MOENV `AQX_P_432`, which provides hourly station-level AQI and pollutant fields.
        - The Kaggle dataset is convenient for reproducible homework; the official API is better for production because it is closer to the source of truth.
        - This app keeps the synthetic assignment mode separate from the real AQI case, so the extra case does not replace the required `n/a/b/var` workflow.
        """
    )
