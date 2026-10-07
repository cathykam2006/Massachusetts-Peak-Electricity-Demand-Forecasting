"""
app.py — Streamlit app for the Massachusetts Peak Electricity Demand project.

Loads pre-trained models and pre-computed data (produced by train_and_save.py)
rather than retraining anything live, so the app stays fast.

Run locally with:
    streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="MA Peak Electricity Demand Forecasting",
    page_icon="⚡",
    layout="wide",
)

TARGET = "Demand (MW)"


# ---------------------------------------------------------------------------
# Cached loaders — @st.cache_data / @st.cache_resource keep these from
# re-running on every user interaction (Streamlit re-runs the whole script
# top to bottom on every click otherwise).
# ---------------------------------------------------------------------------
@st.cache_data
def load_dataset():
    df = pd.read_csv("data/isone_full_dataset.csv", index_col="period")
    df.index = pd.to_datetime(df.index, utc=True).tz_convert("America/New_York")
    return df


@st.cache_data
def load_test_predictions():
    df = pd.read_csv("data/test_predictions.csv", index_col="period")
    df.index = pd.to_datetime(df.index, utc=True).tz_convert("America/New_York")
    return df


@st.cache_data
def load_comparison_table():
    return pd.read_csv("data/model_comparison.csv", index_col=0)


@st.cache_resource
def load_models():
    xgb_model = joblib.load("models/xgb_model.pkl")
    lgb_model = joblib.load("models/lgb_model.pkl")
    feature_cols = joblib.load("models/feature_cols.pkl")
    return xgb_model, lgb_model, feature_cols


@st.cache_data
def load_shap_values():
    return np.load("data/shap_values_test.npy")


df_combined = load_dataset()
test_predictions = load_test_predictions()
comparison_df = load_comparison_table()
xgb_model, lgb_model, feature_cols = load_models()
shap_values_test = load_shap_values()


# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------
st.sidebar.title("⚡ Navigation")
page = st.sidebar.radio(
    "Go to",
    ["Overview", "Forecast", "Exploratory Analysis", "Model Comparison", "Extreme Peak Explainability"],
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Data sources:** EIA (demand, renewable generation), "
    "Open-Meteo (weather).\n\n"
    "Models are pre-trained offline — this app only serves saved results."
)


# ---------------------------------------------------------------------------
# Page: Overview
# ---------------------------------------------------------------------------
if page == "Overview":
    st.title("Predicting Peak Electricity Demand in Massachusetts")
    st.markdown(
        """
        Using weather, load, renewable generation, and consumer behavior data to
        forecast hourly electricity demand for ISO New England, with a focus on
        identifying the conditions that drive **extreme peak demand** events.
        """
    )

    col1, col2, col3 = st.columns(3)
    col1.metric("Rows of hourly data", f"{len(df_combined):,}")
    col2.metric("Date range", f"{df_combined.index.min().year}–{df_combined.index.max().year}")
    best_model = comparison_df["MAPE"].idxmin()
    col3.metric("Best model (by MAPE)", best_model, f"{comparison_df.loc[best_model, 'MAPE']}% MAPE")

    st.markdown("### Key Findings")
    st.markdown(
        """
        - An **ensemble of XGBoost and LightGBM** achieved the best forecasting accuracy,
          roughly an 8x improvement over a naive seasonal-average baseline.
        - **SARIMAX**, tested separately, underperformed even the naive baseline at both
          a 90-day and a 24-hour forecast horizon.
        - **Extreme peak demand** clusters around 7 PM on weekdays in both winter and summer,
          but for opposite physical reasons: winter peaks are driven by heating load on cold
          evenings, while summer peaks are driven by cooling load on hot evenings.
        """
    )

    st.markdown("### Sample of the Combined Dataset")
    st.dataframe(df_combined.tail(10))


# ---------------------------------------------------------------------------
# Page: Forecast
# ---------------------------------------------------------------------------
elif page == "Forecast":
    st.title("Forecast Electricity Demand")
    st.markdown(
        """
        Select a historical date range and adjust conditions to see how the model's
        demand forecast responds. This uses the actual historical lag features (yesterday's
        and last week's demand) from the selected period, with your chosen overrides applied
        to weather and calendar conditions.
        """
    )

    min_date = df_combined.index.min().date()
    max_date = df_combined.index.max().date()

    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input(
            "Start date", value=max_date - pd.Timedelta(days=7),
            min_value=min_date, max_value=max_date,
        )
    with col2:
        end_date = st.date_input(
            "End date", value=max_date,
            min_value=min_date, max_value=max_date,
        )

    if start_date > end_date:
        st.error("Start date must be before end date.")
        st.stop()

    st.subheader("Scenario Adjustments (optional)")
    col3, col4, col5 = st.columns(3)
    with col3:
        temp_offset = st.slider(
            "Temperature adjustment (°C)", min_value=-15, max_value=15, value=0,
            help="Shift temperature warmer (+) or colder (-) than what actually happened.",
        )
    with col4:
        force_weekend = st.selectbox("Day type override", ["Use actual", "Force weekday", "Force weekend"])
    with col5:
        force_holiday = st.selectbox("Holiday override", ["Use actual", "Force not holiday", "Force holiday"])

    # Slice the selected period from the real dataset (keeps real lag/rolling features intact)
    mask = (df_combined.index.date >= start_date) & (df_combined.index.date <= end_date)
    scenario = df_combined.loc[mask, feature_cols].copy()

    if scenario.empty:
        st.warning("No data in the selected range.")
        st.stop()

    # Apply scenario overrides
    scenario["Temperature (°C)"] = scenario["Temperature (°C)"] + temp_offset
    scenario["cooling_degree"] = (scenario["Temperature (°C)"] - 18).clip(lower=0)
    scenario["heating_degree"] = (18 - scenario["Temperature (°C)"]).clip(lower=0)

    if force_weekend == "Force weekday":
        scenario["is_weekend"] = 0
    elif force_weekend == "Force weekend":
        scenario["is_weekend"] = 1

    if force_holiday == "Force not holiday":
        scenario["is_holiday"] = 0
    elif force_holiday == "Force holiday":
        scenario["is_holiday"] = 1

    actual = df_combined.loc[mask, TARGET]
    baseline_forecast = xgb_model.predict(df_combined.loc[mask, feature_cols])
    scenario_forecast = xgb_model.predict(scenario)

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(actual.index, actual.values, label="Actual Demand", color="black", linewidth=1.2)
    ax.plot(actual.index, baseline_forecast, label="Model Forecast (actual conditions)", color="steelblue", linewidth=1, alpha=0.8)
    if temp_offset != 0 or force_weekend != "Use actual" or force_holiday != "Use actual":
        ax.plot(actual.index, scenario_forecast, label="Scenario Forecast (adjusted)", color="darkorange", linewidth=1.2)
    ax.set_xlabel("Date")
    ax.set_ylabel("Demand (MW)")
    ax.legend()
    ax.grid(alpha=0.3)
    st.pyplot(fig)

    col6, col7, col8 = st.columns(3)
    col6.metric("Avg actual demand", f"{actual.mean():,.0f} MW")
    col7.metric("Avg model forecast", f"{baseline_forecast.mean():,.0f} MW")
    if temp_offset != 0 or force_weekend != "Use actual" or force_holiday != "Use actual":
        delta = scenario_forecast.mean() - baseline_forecast.mean()
        col8.metric("Avg scenario forecast", f"{scenario_forecast.mean():,.0f} MW", f"{delta:+,.0f} MW vs actual conditions")

    st.caption(
        "Note: this scenario tool re-uses real historical lag features (yesterday's and "
        "last week's demand) from the selected period. It is not a true future forecast — "
        "it shows how the model's prediction would shift if weather or calendar conditions "
        "during that same historical window had been different."
    )


# ---------------------------------------------------------------------------
# Page: Exploratory Analysis
# ---------------------------------------------------------------------------
elif page == "Exploratory Analysis":
    st.title("Exploratory Data Analysis")

    st.subheader("Hourly Demand Over Time (with Extreme Peaks Highlighted)")
    threshold = df_combined[TARGET].quantile(0.95)
    peaks = df_combined[df_combined[TARGET] > threshold]

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(df_combined.index, df_combined[TARGET], color="steelblue", linewidth=0.5, label="Hourly Demand")
    ax.scatter(peaks.index, peaks[TARGET], color="red", s=6, label=f"Top 5% Peak Hours (>{threshold:.0f} MW)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Demand (MW)")
    ax.legend()
    st.pyplot(fig)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Weekday vs. Weekend Demand")
        fig, ax = plt.subplots(figsize=(7, 5))
        for is_weekend, label in [(0, "Weekday"), (1, "Weekend")]:
            subset = df_combined[df_combined["is_weekend"] == is_weekend]
            hourly = subset.groupby("hour")[TARGET].mean()
            ax.plot(hourly.index, hourly.values, marker="o", label=label)
        ax.set_xlabel("Hour of Day")
        ax.set_ylabel("Average Demand (MW)")
        ax.legend()
        ax.grid(alpha=0.3)
        st.pyplot(fig)

    with col2:
        st.subheader("Demand vs. Temperature")
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.scatter(df_combined["Temperature (°C)"], df_combined[TARGET], alpha=0.08, s=4)
        ax.set_xlabel("Temperature (°C)")
        ax.set_ylabel("Demand (MW)")
        ax.grid(alpha=0.3)
        st.pyplot(fig)

    st.subheader("Monthly Demand Distribution")
    month_selection = st.multiselect(
        "Filter by month (optional)",
        options=list(range(1, 13)),
        default=list(range(1, 13)),
    )
    filtered = df_combined[df_combined["month"].isin(month_selection)]
    fig, ax = plt.subplots(figsize=(12, 5))
    filtered.boxplot(column=TARGET, by="month", ax=ax)
    ax.set_xlabel("Month")
    ax.set_ylabel("Demand (MW)")
    plt.suptitle("")
    ax.set_title("")
    st.pyplot(fig)


# ---------------------------------------------------------------------------
# Page: Model Comparison
# ---------------------------------------------------------------------------
elif page == "Model Comparison":
    st.title("Model Comparison")

    st.dataframe(
        comparison_df.style.highlight_min(subset=["MAE", "RMSE", "MAPE"], color="lightgreen")
    )

    st.subheader("MAPE by Model")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(comparison_df.index, comparison_df["MAPE"], color="steelblue")
    ax.set_ylabel("MAPE (%)")
    plt.xticks(rotation=20)
    st.pyplot(fig)

    st.subheader("Actual vs. Predicted (Test Period)")
    model_choice = st.selectbox(
        "Choose a model to plot",
        ["ensemble_pred", "xgb_pred", "lgb_pred", "baseline_pred"],
        format_func=lambda x: x.replace("_pred", "").upper(),
    )
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(test_predictions.index, test_predictions[TARGET], label="Actual", color="black", linewidth=1)
    ax.plot(test_predictions.index, test_predictions[model_choice], label="Predicted", color="orange", linewidth=1, alpha=0.8)
    ax.set_xlabel("Date")
    ax.set_ylabel("Demand (MW)")
    ax.legend()
    st.pyplot(fig)

    st.subheader("XGBoost Feature Importance")
    importances = pd.Series(xgb_model.feature_importances_, index=feature_cols).sort_values()
    fig, ax = plt.subplots(figsize=(8, 6))
    importances.plot(kind="barh", ax=ax, color="steelblue")
    ax.set_xlabel("Importance")
    st.pyplot(fig)


# ---------------------------------------------------------------------------
# Page: Extreme Peak Explainability
# ---------------------------------------------------------------------------
elif page == "Extreme Peak Explainability":
    st.title("What Drives Extreme Peak Demand?")

    st.markdown(
        """
        SHAP analysis on the top 5% highest-demand hours in the test set, showing which
        conditions the model associates most strongly with extreme demand.
        """
    )

    st.subheader("SHAP Summary — All Test Hours")
    fig, ax = plt.subplots(figsize=(10, 6))
    shap.summary_plot(shap_values_test, test_predictions[feature_cols], show=False)
    st.pyplot(plt.gcf())
    plt.clf()

    st.subheader("Winter vs. Summer: Extreme Peak Conditions")
    categories = ["Avg Temp (°C)", "Avg Heating\nDegree", "Avg Cooling\nDegree", "% Weekend"]
    winter_peak_vals = [0.63, 17.41, 0.04, 4.6]
    winter_all_vals = [7.26, 10.92, 0.18, 28.7]
    summer_peak_vals = [27.94, 0.0, 9.94, 5.5]
    summer_all_vals = [22.42, 0.16, 4.58, 29.5]

    x = np.arange(len(categories))
    width = 0.35
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].bar(x - width / 2, winter_all_vals, width, label="All Hours", color="lightblue")
    axes[0].bar(x + width / 2, winter_peak_vals, width, label="Extreme Peak Hours", color="darkblue")
    axes[0].set_title("Winter (Oct–Dec): Extreme Peaks vs. All Hours")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(categories, fontsize=9)
    axes[0].legend()
    axes[0].grid(alpha=0.3, axis="y")

    axes[1].bar(x - width / 2, summer_all_vals, width, label="All Hours", color="lightsalmon")
    axes[1].bar(x + width / 2, summer_peak_vals, width, label="Extreme Peak Hours", color="darkred")
    axes[1].set_title("Summer (Jul–Aug): Extreme Peaks vs. All Hours")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(categories, fontsize=9)
    axes[1].legend()
    axes[1].grid(alpha=0.3, axis="y")

    st.pyplot(fig)

    st.markdown(
        """
        **Winter**: extreme peaks are cold weekday evenings (~7 PM), with heating-degree
        values ~60% above the seasonal average.

        **Summer**: extreme peaks are hot weekday evenings (~7 PM), with cooling-degree
        values more than double the seasonal average.

        Both seasons show the same weekday-dominant, early-evening pattern — despite
        opposite underlying physical drivers.
        """
    )
