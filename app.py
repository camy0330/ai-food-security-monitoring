
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI-Driven Food Security Monitoring",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .block-container {
        padding-top: 1.8rem;
        padding-bottom: 2rem;
    }

    [data-testid="stMetric"] {
        background-color: rgba(128, 128, 128, 0.08);
        border: 1px solid rgba(128, 128, 128, 0.15);
        padding: 15px;
        border-radius: 10px;
    }

    .section-caption {
        color: #777777;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# 2. FILE PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "outputs" / "dashboard"


# ============================================================
# 3. DATA LOADING FUNCTIONS
# ============================================================

@st.cache_data
def load_csv(filename):
    file_path = DATA_DIR / filename

    if not file_path.exists():
        return None

    return pd.read_csv(file_path)


@st.cache_data
def load_metadata():
    file_path = DATA_DIR / "metadata.json"

    if not file_path.exists():
        return {}

    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)


def prepare_dates(df, column="month"):
    if df is not None and column in df.columns:
        df = df.copy()
        df[column] = pd.to_datetime(df[column], errors="coerce")

    return df


# ============================================================
# 4. LOAD PROJECT DATA
# ============================================================

district_df = load_csv("district_vulnerability.csv")
efpi_df = load_csv("efpi_monitoring.csv")
forecast_df = load_csv("efpi_forecast.csv")
performance_df = load_csv("model_performance.csv")
naive_df = load_csv("naive_reference.csv")
predictions_df = load_csv("forecast_test_predictions.csv")

vulnerability_summary = load_csv("vulnerability_summary.csv")
efpi_status = load_csv("current_efpi_status.csv")
forecast_summary = load_csv("forecast_summary.csv")
model_summary = load_csv("model_summary.csv")

food_prices_df = load_csv("food_price_indices.csv")
food_basket_df = load_csv("food_basket.csv")

metadata = load_metadata()

efpi_df = prepare_dates(efpi_df)
forecast_df = prepare_dates(forecast_df)
predictions_df = prepare_dates(predictions_df)

if food_prices_df is not None:
    food_prices_df = prepare_dates(food_prices_df)


# ============================================================
# 5. CHECK REQUIRED DATASETS
# ============================================================

required_files = [
    "district_vulnerability.csv",
    "efpi_monitoring.csv",
    "efpi_forecast.csv",
    "model_performance.csv",
    "naive_reference.csv",
    "forecast_test_predictions.csv",
    "food_price_indices.csv",
    "food_basket.csv",
    "metadata.json"
]

missing_files = [
    filename for filename in required_files
    if not (DATA_DIR / filename).exists()
]

if missing_files:
    st.error(
        "Some dashboard datasets are missing. "
        "Please upload the following files into "
        "`outputs/dashboard/` in your GitHub repository:"
    )

    for filename in missing_files:
        st.write(f"- {filename}")

    st.stop()


# ============================================================
# 6. HELPER FUNCTIONS
# ============================================================

def format_number(value, decimals=2):
    if pd.isna(value):
        return "N/A"

    return f"{value:,.{decimals}f}"


def show_metric(label, value, help_text=None):
    st.metric(
        label=label,
        value=value,
        help=help_text
    )


def get_selected_model():
    return metadata.get("selected_model", "Not available")


def get_latest_efpi():
    if efpi_df is None or efpi_df.empty:
        return np.nan

    latest = efpi_df.sort_values("month").iloc[-1]
    return latest["EFPI"]


def get_latest_month():
    if efpi_df is None or efpi_df.empty:
        return "N/A"

    latest = efpi_df["month"].max()

    if pd.isna(latest):
        return "N/A"

    return latest.strftime("%B %Y")


def chart_theme(fig, title=None):
    fig.update_layout(
        title=title,
        template="plotly_white",
        margin=dict(l=20, r=20, t=60, b=20),
        legend_title_text="",
        hovermode="x unified"
    )

    return fig


# ============================================================
# 7. SIDEBAR NAVIGATION
# ============================================================

st.sidebar.title("📊 Project Dashboard")

st.sidebar.caption(
    "AI-Driven Socioeconomic Vulnerability "
    "and Essential Food Price Monitoring"
)

page = st.sidebar.radio(
    "Navigation",
    [
        "Overview",
        "Socioeconomic Vulnerability",
        "Food Price Monitoring",
        "EFPI Forecasting",
        "Model Evaluation"
    ]
)

st.sidebar.divider()

st.sidebar.caption("Data sources")
st.sidebar.write("• HIES and socioeconomic statistics")
st.sidebar.write("• PriceCatcher / OpenDOSM")

st.sidebar.divider()

st.sidebar.caption(
    "EFPI is a custom research index based on "
    "eight selected food categories. It is not "
    "the official DOSM CPI."
)


# ============================================================
# 8. DASHBOARD HEADER
# ============================================================

st.title("AI-Driven Food Security Monitoring")

st.markdown(
    "An integrated monitoring dashboard for examining "
    "district-level socioeconomic vulnerability and "
    "essential food price movements in Malaysia."
)

st.caption(
    f"Socioeconomic reference year: "
    f"{metadata.get('socio_reference_year', 'N/A')} | "
    f"EFPI period: "
    f"{metadata.get('efpi_start', 'N/A')} to "
    f"{metadata.get('efpi_end', 'N/A')}"
)

st.divider()


# ============================================================
# PAGE 1: OVERVIEW
# ============================================================

if page == "Overview":

    st.subheader("Project Overview")

    latest_efpi = get_latest_efpi()
    selected_model = get_selected_model()

    total_districts = (
        len(district_df)
        if district_df is not None else 0
    )

    total_months = (
        len(efpi_df)
        if efpi_df is not None else 0
    )

    total_categories = (
        food_basket_df["food_category"].nunique()
        if food_basket_df is not None
        and "food_category" in food_basket_df.columns
        else 0
    )

    # KPI cards
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        show_metric(
            "Districts Analysed",
            f"{total_districts:,}"
        )

    with col2:
        show_metric(
            "Latest EFPI",
            format_number(latest_efpi, 2),
            "January 2023 = 100"
        )

    with col3:
        show_metric(
            "Food Categories",
            str(total_categories)
        )

    with col4:
        show_metric(
            "Historical Months",
            str(total_months)
        )

    st.divider()

    # Historical EFPI
    st.subheader("Essential Food Price Index Trend")

    fig = px.line(
        efpi_df.sort_values("month"),
        x="month",
        y="EFPI",
        markers=True,
        labels={
            "month": "Month",
            "EFPI": "EFPI"
        }
    )

    fig.add_hline(
        y=100,
        line_dash="dash",
        annotation_text="Base index = 100"
    )

    st.plotly_chart(
        chart_theme(fig),
        use_container_width=True
    )

    # Vulnerability distribution
    st.subheader("Socioeconomic Vulnerability Distribution")

    col1, col2 = st.columns(2)

    with col1:
        if vulnerability_summary is not None:

            fig = px.pie(
                vulnerability_summary,
                names="vulnerability_level",
                values="district_count",
                hole=0.45,
                title="District Distribution by Vulnerability"
            )

            st.plotly_chart(
                chart_theme(fig),
                use_container_width=True
            )

    with col2:
        if vulnerability_summary is not None:

            fig = px.bar(
                vulnerability_summary,
                x="vulnerability_level",
                y="district_count",
                text="district_count",
                labels={
                    "vulnerability_level": "Vulnerability Level",
                    "district_count": "Number of Districts"
                },
                title="Number of Districts"
            )

            st.plotly_chart(
                chart_theme(fig),
                use_container_width=True
            )

    st.divider()

    st.subheader("Forecasting Summary")

    col1, col2 = st.columns(2)

    with col1:
        show_metric(
            "Selected Forecast Model",
            selected_model
        )

    with col2:
        show_metric(
            "Lowest-RMSE Candidate",
            metadata.get("lowest_rmse_model", "N/A")
        )

    st.info(
        "The selected forecasting model is determined using "
        "the notebook's predefined RMSE tolerance and model "
        "complexity rule. The naive previous-month forecast "
        "is retained as a separate benchmark."
    )


# ============================================================
# PAGE 2: SOCIOECONOMIC VULNERABILITY
# ============================================================

elif page == "Socioeconomic Vulnerability":

    st.subheader("District-Level Socioeconomic Vulnerability")

    st.markdown(
        "Explore the socioeconomic profiles of Malaysian "
        "districts based on five selected indicators."
    )

    st.caption(
        "The vulnerability labels represent relative "
        "socioeconomic profiles from K-Means clustering. "
        "They are not official food-insecurity classifications."
    )

    # Filters
    col1, col2 = st.columns(2)

    states = sorted(district_df["state"].dropna().unique())
    levels = sorted(
        district_df["vulnerability_level"].dropna().unique()
    )

    with col1:
        selected_states = st.multiselect(
            "Select state",
            options=states,
            default=states
        )

    with col2:
        selected_levels = st.multiselect(
            "Select vulnerability level",
            options=levels,
            default=levels
        )

    filtered = district_df[
        district_df["state"].isin(selected_states)
        & district_df["vulnerability_level"].isin(selected_levels)
    ].copy()

    st.divider()

    col1, col2, col3 = st.columns(3)

    with col1:
        show_metric(
            "Districts Displayed",
            str(len(filtered))
        )

    with col2:
        show_metric(
            "States Displayed",
            str(filtered["state"].nunique())
        )

    with col3:
        higher_count = (
            filtered["vulnerability_level"]
            .eq("Higher socioeconomic vulnerability")
            .sum()
        )

        show_metric(
            "Higher Vulnerability Districts",
            str(higher_count)
        )

    # Distribution
    st.subheader("Vulnerability Distribution")

    distribution = (
        filtered["vulnerability_level"]
        .value_counts()
        .reset_index()
    )

    distribution.columns = [
        "vulnerability_level",
        "district_count"
    ]

    fig = px.bar(
        distribution,
        x="vulnerability_level",
        y="district_count",
        color="vulnerability_level",
        text="district_count",
        labels={
            "vulnerability_level": "Vulnerability Level",
            "district_count": "District Count"
        }
    )

    st.plotly_chart(
        chart_theme(fig),
        use_container_width=True
    )

    # Indicator comparison
    st.subheader("Socioeconomic Indicator Analysis")

    indicator = st.selectbox(
        "Choose indicator",
        options=[
            "income_median",
            "poverty",
            "gini",
            "u_rate",
            "piped_water"
        ],
        format_func=lambda x: {
            "income_median": "Median Household Income",
            "poverty": "Poverty Rate",
            "gini": "Gini Coefficient",
            "u_rate": "Unemployment Rate",
            "piped_water": "Access to Piped Water"
        }[x]
    )

    fig = px.box(
        filtered,
        x="vulnerability_level",
        y=indicator,
        color="vulnerability_level",
        points="all",
        labels={
            "vulnerability_level": "Vulnerability Level",
            indicator: indicator.replace("_", " ").title()
        }
    )

    st.plotly_chart(
        chart_theme(fig),
        use_container_width=True
    )

    # District table
    st.subheader("District Details")

    display_columns = [
        "state",
        "district",
        "income_median",
        "poverty",
        "gini",
        "u_rate",
        "piped_water",
        "vulnerability_level"
    ]

    st.dataframe(
        filtered[display_columns],
        use_container_width=True,
        hide_index=True
    )

    csv = filtered.to_csv(index=False).encode("utf-8")

    st.download_button(
        "Download Filtered District Data",
        data=csv,
        file_name="filtered_district_vulnerability.csv",
        mime="text/csv"
    )

    imputed_count = (
        int(filtered["u_rate_imputed"].sum())
        if "u_rate_imputed" in filtered.columns
        else 0
    )

    st.warning(
        f"{imputed_count} displayed district record(s) "
        "have an imputed unemployment value. "
        "Refer to the original analysis for the imputation details."
    )


# ============================================================
# PAGE 3: FOOD PRICE MONITORING
# ============================================================

elif page == "Food Price Monitoring":

    st.subheader("Essential Food Price Monitoring")

    st.markdown(
        "Monitor historical price index movements across "
        "the eight selected essential food categories."
    )

    st.caption(
        "The EFPI uses January 2023 = 100 as its base period "
        "and applies equal weights to the eight food categories."
    )

    latest_efpi = get_latest_efpi()

    previous_efpi = (
        efpi_df.sort_values("month").iloc[-2]["EFPI"]
        if len(efpi_df) >= 2 else np.nan
    )

    monthly_change = (
        (latest_efpi - previous_efpi) / previous_efpi * 100
        if pd.notna(previous_efpi) and previous_efpi != 0
        else np.nan
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        show_metric(
            "Latest EFPI",
            format_number(latest_efpi, 2)
        )

    with col2:
        show_metric(
            "Latest Month",
            get_latest_month()
        )

    with col3:
        show_metric(
            "Month-on-Month Change",
            f"{monthly_change:+.2f}%"
            if pd.notna(monthly_change) else "N/A"
        )

    st.divider()

    # Historical EFPI
    st.subheader("Historical EFPI")

    fig = px.line(
        efpi_df.sort_values("month"),
        x="month",
        y="EFPI",
        markers=True,
        labels={
            "month": "Month",
            "EFPI": "Price Index"
        }
    )

    fig.add_hline(
        y=100,
        line_dash="dash",
        annotation_text="January 2023 = 100"
    )

    st.plotly_chart(
        chart_theme(fig),
        use_container_width=True
    )

    # Food category comparison
    st.subheader("Food Category Price Indices")

    if (
        food_prices_df is not None
        and "food_category" in food_prices_df.columns
        and "price_index" in food_prices_df.columns
    ):

        categories = sorted(
            food_prices_df["food_category"].dropna().unique()
        )

        selected_categories = st.multiselect(
            "Select food categories",
            categories,
            default=categories
        )

        category_data = food_prices_df[
            food_prices_df["food_category"].isin(selected_categories)
        ].copy()

        fig = px.line(
            category_data.sort_values("month"),
            x="month",
            y="price_index",
            color="food_category",
            labels={
                "month": "Month",
                "price_index": "Food Price Index",
                "food_category": "Food Category"
            }
        )

        fig.add_hline(
            y=100,
            line_dash="dash",
            annotation_text="Base index = 100"
        )

        st.plotly_chart(
            chart_theme(fig),
            use_container_width=True
        )

        # Category summary
        st.subheader("Food Category Summary")

        summary = (
            category_data.groupby("food_category")["price_index"]
            .agg(["mean", "min", "max", "std"])
            .reset_index()
        )

        summary.columns = [
            "Food Category",
            "Average Index",
            "Minimum Index",
            "Maximum Index",
            "Standard Deviation"
        ]

        st.dataframe(
            summary.round(2),
            use_container_width=True,
            hide_index=True
        )

    else:
        st.warning(
            "Food category index data is unavailable or "
            "does not contain the expected columns."
        )

    # Food basket
    st.subheader("Selected Food Basket")

    if food_basket_df is not None:
        st.dataframe(
            food_basket_df,
            use_container_width=True,
            hide_index=True
        )

    st.info(
        "Interpretation: an EFPI value above 100 indicates "
        "that the selected food basket's index is above its "
        "January 2023 base level. It does not represent the "
        "percentage change in the official national CPI."
    )


# ============================================================
# PAGE 4: EFPI FORECASTING
# ============================================================

elif page == "EFPI Forecasting":

    st.subheader("Essential Food Price Index Forecasting")

    st.markdown(
        "Explore future EFPI predictions generated by the "
        "selected forecasting model and compare them with "
        "the naive previous-month reference."
    )

    selected_model = get_selected_model()

    col1, col2 = st.columns(2)

    with col1:
        show_metric(
            "Selected Model",
            selected_model
        )

    with col2:
        show_metric(
            "Forecast Horizon",
            f"{len(forecast_df)} months"
        )

    st.divider()

    # Historical + future forecast
    st.subheader("Historical and Forecasted EFPI")

    history = efpi_df[
        ["month", "EFPI"]
    ].copy()

    history = history.rename(
        columns={"EFPI": "Actual EFPI"}
    )

    future = forecast_df.copy()

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=history["month"],
            y=history["Actual EFPI"],
            mode="lines+markers",
            name="Historical EFPI"
        )
    )

    if "forecast" in future.columns:
        fig.add_trace(
            go.Scatter(
                x=future["month"],
                y=future["forecast"],
                mode="lines+markers",
                name=f"{selected_model} Forecast",
                line=dict(dash="dash")
            )
        )

    if "Naive_forecast" in future.columns:
        fig.add_trace(
            go.Scatter(
                x=future["month"],
                y=future["Naive_forecast"],
                mode="lines+markers",
                name="Naive Reference",
                line=dict(dash="dot")
            )
        )

    fig.add_hline(
        y=100,
        line_dash="dash",
        annotation_text="Base index = 100"
    )

    fig.update_layout(
        template="plotly_white",
        xaxis_title="Month",
        yaxis_title="EFPI",
        hovermode="x unified",
        legend_title_text=""
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # Forecast table
    st.subheader("Future Forecast Values")

    display_cols = [
        col for col in [
            "month",
            "model",
            "forecast",
            "Naive_forecast",
            "change_vs_latest_pct",
            "Naive_change_vs_latest_pct"
        ]
        if col in forecast_df.columns
    ]

    st.dataframe(
        forecast_df[display_cols],
        use_container_width=True,
        hide_index=True
    )

    # Forecast range warning
    if (
        "forecast_low" in forecast_df.columns
        and "forecast_high" in forecast_df.columns
    ):
        st.caption(
            "The displayed forecast_low and forecast_high values "
            "are indicative ranges calculated using the selected "
            "model's test RMSE. They are not statistical prediction "
            "intervals or confidence intervals."
        )

    csv = forecast_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        "Download Forecast Data",
        data=csv,
        file_name="efpi_forecast.csv",
        mime="text/csv"
    )


# ============================================================
# PAGE 5: MODEL EVALUATION
# ============================================================

elif page == "Model Evaluation":

    st.subheader("Forecast Model Performance Evaluation")

    st.markdown(
        "Compare the five candidate forecasting models using "
        "MAE, RMSE and R-squared (R²)."
    )

    st.caption(
        "The naive previous-month forecast is shown separately "
        "as a reference benchmark."
    )

    if "model" not in performance_df.columns:
        st.error(
            "The model_performance.csv file does not contain "
            "the expected 'model' column."
        )
        st.stop()

    # Performance table
    st.subheader("Candidate Model Comparison")

    if "selected" in performance_df.columns:
        selected_rows = performance_df[
            performance_df["selected"].astype(str).str.lower().isin(
                ["true", "1"]
            )
        ]

        if not selected_rows.empty:
            st.success(
                f"Selected candidate model: "
                f"{selected_rows.iloc[0]['model']}"
            )

    display_cols = [
        col for col in [
            "model",
            "MAE",
            "RMSE",
            "R2",
            "RMSE_std",
            "Seeds",
            "beats_naive_rmse",
            "selected"
        ]
        if col in performance_df.columns
    ]

    st.dataframe(
        performance_df[display_cols].round(4),
        use_container_width=True,
        hide_index=True
    )

    # RMSE comparison
    st.subheader("RMSE Comparison")

    fig = px.bar(
        performance_df.sort_values("RMSE"),
        x="model",
        y="RMSE",
        color="model",
        text="RMSE",
        labels={
            "model": "Forecasting Model",
            "RMSE": "Root Mean Squared Error"
        }
    )

    st.plotly_chart(
        chart_theme(fig),
        use_container_width=True
    )

    # MAE comparison
    st.subheader("MAE Comparison")

    fig = px.bar(
        performance_df.sort_values("MAE"),
        x="model",
        y="MAE",
        color="model",
        text="MAE",
        labels={
            "model": "Forecasting Model",
            "MAE": "Mean Absolute Error"
        }
    )

    st.plotly_chart(
        chart_theme(fig),
        use_container_width=True
    )

    # R2 comparison
    st.subheader("R-squared Comparison")

    fig = px.bar(
        performance_df.sort_values("R2", ascending=False),
        x="model",
        y="R2",
        color="model",
        text="R2",
        labels={
            "model": "Forecasting Model",
            "R2": "R-squared"
        }
    )

    fig.add_hline(
        y=0,
        line_dash="dash",
        annotation_text="R² = 0"
    )

    st.plotly_chart(
        chart_theme(fig),
        use_container_width=True
    )

    # Naive reference
    st.divider()

    st.subheader("Naive Previous-Month Benchmark")

    if naive_df is not None and not naive_df.empty:

        st.dataframe(
            naive_df.round(4),
            use_container_width=True,
            hide_index=True
        )

        naive_rmse = (
            naive_df.iloc[0]["RMSE"]
            if "RMSE" in naive_df.columns
            else np.nan
        )

        naive_mae = (
            naive_df.iloc[0]["MAE"]
            if "MAE" in naive_df.columns
            else np.nan
        )

        col1, col2 = st.columns(2)

        with col1:
            show_metric(
                "Naive RMSE",
                format_number(naive_rmse, 4)
            )

        with col2:
            show_metric(
                "Naive MAE",
                format_number(naive_mae, 4)
            )

    # Actual vs predicted
    st.divider()

    st.subheader("Actual vs Predicted EFPI")

    if predictions_df is not None:

        model_columns = [
            col for col in predictions_df.columns
            if col not in ["month", "Actual"]
        ]

        selected_prediction_models = st.multiselect(
            "Select prediction models",
            model_columns,
            default=model_columns
        )

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=predictions_df["month"],
                y=predictions_df["Actual"],
                mode="lines+markers",
                name="Actual EFPI",
                line=dict(width=3)
            )
        )

        for model in selected_prediction_models:

            fig.add_trace(
                go.Scatter(
                    x=predictions_df["month"],
                    y=predictions_df[model],
                    mode="lines+markers",
                    name=model
                )
            )

        fig.update_layout(
            template="plotly_white",
            xaxis_title="Test Month",
            yaxis_title="EFPI",
            hovermode="x unified"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    st.warning(
        "Forecast performance is based on the nine-month "
        "test period from January to September 2026. "
        "Interpret differences between models cautiously "
        "because the evaluation period is relatively short."
    )


# ============================================================
# 9. FOOTER
# ============================================================

st.divider()

st.caption(
    "AI-Driven Socioeconomic Vulnerability and "
    "Essential Food Price Monitoring | "
    "Final Year Project"
)

st.caption(
    "Data analysis and model development are performed "
    "in the Jupyter Notebook. This dashboard presents "
    "the exported analytical results."
)
