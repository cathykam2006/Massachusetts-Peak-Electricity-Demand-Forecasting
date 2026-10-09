"""
app.py — Streamlit app for the Massachusetts / ISO-NE day-ahead demand forecast.

Reads only the files written by train_dayahead.py and forecast_tomorrow.py
(data/dayahead/). No models are loaded or trained here, so the app stays fast.

    streamlit run app.py
"""

import json
import os

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="New England Day-Ahead Demand Forecast",
                   page_icon="⚡", layout="wide")

DATA = "data/dayahead"
TZ = "America/New_York"
TARGET = "demand_mw"

# Palette (matches .streamlit/config.toml)
INK, OURS, BAND, ISO, PEAK = "#1E2B3A", "#11698E", "#BBD9E6", "#C2185B", "#D9900F"
SEASON_COLORS = {"Winter": "#3B6EA8", "Spring": "#6AA67A", "Summer": "#D9900F", "Fall": "#9C6B4E"}

FRIENDLY = {
    "hour": "Hour of day", "dayofweek": "Day of week", "month": "Month",
    "doy_sin": "Season (sin)", "doy_cos": "Season (cos)", "is_weekend": "Weekend",
    "is_holiday": "Holiday", "is_holiday_season": "Christmas–New Year week",
    "temp_c": "Forecast temperature", "dewpoint_c": "Forecast dew point",
    "rh_pct": "Forecast humidity", "wind_kmh": "Forecast wind speed",
    "cloud_pct": "Forecast cloud cover", "ghi_wm2": "Forecast sunshine",
    "hdh": "Heating degree hours", "cdh": "Cooling degree hours",
    "temp_max_d": "Day's forecast high", "temp_min_d": "Day's forecast low",
    "temp_mean_d": "Day's forecast mean temperature", "temp_mean_3d_wtd": "3-day heat buildup",
    "cdd_d": "Cooling degree day", "hdd_d": "Heating degree day",
    "ghi_sum_d": "Day's total sunshine", "ghi_x_daylight": "Midday sunshine",
    "load_same_hour_d2": "Demand, same hour 2 days ago",
    "load_same_hour_d7": "Demand, same hour last week",
    "load_same_hour_d14": "Demand, same hour 2 weeks ago",
    "load_peak_d2": "Peak 2 days ago", "load_mean_d2": "Average 2 days ago",
    "load_peak_d7": "Peak a week ago", "load_d1_morning_mean": "This morning's demand",
    "load_d1_last_obs": "Latest observed demand",
    "load_d1_morning_vs_lastweek": "This morning vs. last week",
}
NOT_FEATURES = {TARGET, "ts_local", "period_utc", "time", "Unnamed: 0"}


# ---------------------------------------------------------------------------
# Loading and helpers
# ---------------------------------------------------------------------------
def _local_naive(series):
    """Parse timestamps and express them as New England wall-clock time."""
    return pd.to_datetime(series, utc=True).dt.tz_convert(TZ).dt.tz_localize(None)


@st.cache_data(ttl=3600)
def load(name):
    path = f"{DATA}/{name}"
    if not os.path.exists(path):
        return None
    if name.endswith(".json"):
        with open(path) as f:
            return json.load(f)
    if name.endswith(".npy"):
        return np.load(path)
    df = pd.read_csv(path)
    if "period_utc" in df:
        df["time"] = _local_naive(df["period_utc"])
    elif "ts_local" in df:
        df["time"] = _local_naive(df["ts_local"])
    if "target_date" in df:
        df["target_date"] = pd.to_datetime(df["target_date"])
    return df


def season_of(month):
    return np.select([np.isin(month, [12, 1, 2]), np.isin(month, [3, 4, 5]),
                      np.isin(month, [6, 7, 8])], ["Winter", "Spring", "Summer"], "Fall")


def mape(actual, pred):
    m = actual.notna() & pred.notna()
    return float((np.abs(pred[m] - actual[m]) / actual[m]).mean() * 100) if m.any() else np.nan


def mape_by(df, key, pred_col):
    return pd.Series({k: mape(g[TARGET], g[pred_col]) for k, g in df.groupby(key)})


def fmt_table(df):
    return df.style.format({c: "{:,.0f}" if "MW" in c else "{:.2f}" for c in df.columns})


def forecast_chart(df, show_actual=False, show_iso=True, show_ours=True, height=380):
    """Band + forecast line (+ ISO-NE, + actual): the core visual of the app.
    Layer order: band, actual, our forecast, ISO-NE on top, so the thin dashed
    ISO line is never hidden where it overlaps the actual line."""
    multi_day = df["time"].dt.date.nunique() > 1
    x = alt.X("time:T", title=None,
              axis=alt.Axis(format="%a %b %-d" if multi_day else "%-I %p", labelAngle=0))
    base = alt.Chart(df).encode(x=x)
    hour_tip = alt.Tooltip("time:T", format="%a %b %-d, %-I %p", title="Hour")
    y_scale = alt.Scale(zero=False)
    layers = []
    if show_ours:
        layers.append(base.mark_area(color=BAND, opacity=0.85).encode(
            y=alt.Y("pred_lo:Q", title="Demand (MW)", scale=y_scale), y2="pred_hi:Q",
            tooltip=[hour_tip, alt.Tooltip("pred_lo:Q", format=",.0f", title="Low end"),
                     alt.Tooltip("pred_hi:Q", format=",.0f", title="High end")]))
    if show_actual and df[TARGET].notna().any():
        layers.append(base.mark_line(color=INK, strokeWidth=1.8).encode(
            y=alt.Y(f"{TARGET}:Q", title="Demand (MW)", scale=y_scale),
            tooltip=[hour_tip, alt.Tooltip(f"{TARGET}:Q", format=",.0f", title="Actual")]))
    if show_ours:
        layers.append(base.mark_line(color=OURS, strokeWidth=2.5).encode(
            y=alt.Y("pred:Q", title="Demand (MW)", scale=y_scale),
            tooltip=[hour_tip, alt.Tooltip("pred:Q", format=",.0f", title="Our forecast")]))
    if show_iso and "iso_forecast_mw" in df and df["iso_forecast_mw"].notna().any():
        layers.append(base.mark_line(color=ISO, strokeDash=[7, 4], strokeWidth=2.2).encode(
            y=alt.Y("iso_forecast_mw:Q", title="Demand (MW)", scale=y_scale),
            tooltip=[hour_tip, alt.Tooltip("iso_forecast_mw:Q", format=",.0f", title="ISO-NE forecast")]))
    return alt.layer(*layers).properties(height=height)


def miss_chart(df, height=200):
    """Our forecast minus actual, hour by hour. Zero = perfect; above zero =
    forecast too high, below = too low. Makes small gaps visible."""
    d = df[["time", TARGET, "pred"]].dropna().copy()
    d["Miss"] = d["pred"] - d[TARGET]
    multi_day = d["time"].dt.date.nunique() > 1
    axis = (alt.Axis(format="%a %b %-d", tickCount="day", labelAngle=0) if multi_day
            else alt.Axis(format="%-I %p", labelAngle=0))
    bars = alt.Chart(d).mark_bar(width=3).encode(
        x=alt.X("time:T", title=None, axis=axis),
        y=alt.Y("Miss:Q", title="Forecast − actual (MW)"),
        color=alt.condition("datum.Miss >= 0", alt.value(OURS), alt.value(PEAK)),
        tooltip=[alt.Tooltip("time:T", format="%a %b %-d, %-I %p", title="Hour"),
                 alt.Tooltip(f"{TARGET}:Q", format=",.0f", title="Actual"),
                 alt.Tooltip("pred:Q", format=",.0f", title="Our forecast"),
                 alt.Tooltip("Miss:Q", format="+,.0f", title="Miss (MW)")])
    zero = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(color=INK, strokeWidth=1).encode(y="y:Q")
    return (bars + zero).properties(height=height)


def comparison_view(df, key=None, height=340):
    """Actual demand against our forecast, then the hour-by-hour miss."""
    legend([("Actual", INK, "line"), ("Our forecast", OURS, "line"), ("80% range", BAND, "band")])
    st.altair_chart(forecast_chart(df, show_actual=True, show_iso=False, height=height),
                    width="stretch")
    st.markdown("**How far our forecast missed, hour by hour**")
    st.altair_chart(miss_chart(df), width="stretch")
    st.caption("Zero is a perfect forecast. Blue bars: forecast too high. Amber bars: forecast too low.")


def legend(items):
    """Inline legend: layered Altair charts don't build one on their own."""
    parts = []
    for label, color, style in items:
        mark = {"line": "━━", "dash": "╍╍", "band": "███"}[style]
        parts.append(f"<span style='color:{color}'>{mark}</span>&nbsp;{label}")
    st.markdown(f"<div style='font-size:0.9rem;margin-bottom:0.25rem'>"
                f"{'&emsp;'.join(parts)}</div>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Planning buffer, calibrated on daily peaks
# ---------------------------------------------------------------------------
# A shortfall costs far more than idle capacity, so planners hold more than the
# expected peak. The buffer starts from the day's highest 90th-percentile hour,
# then adds a margin learned from recent days: how far actual peaks landed above
# that level. The margin is the quantile that would have held at the chosen risk.
RISK_LEVELS = {"1 day in 10": 0.10, "1 day in 20": 0.05, "1 day in 50": 0.02}
PLAN_WINDOW_DAYS = 90   # recent days used to size the margin
MIN_PLAN_DAYS = 30      # need at least this many before trusting the margin


@st.cache_data(ttl=3600)
def daily_peak_history():
    """Complete days from the backtest and the scored live log: actual peak,
    expected peak, and the raw plan base (highest uncalibrated P90 hour)."""
    parts = []
    for name in ("backtest.csv", "forecast_log.csv"):
        df = load(name)
        if df is None or "pred_hi_raw" not in df or TARGET not in df:
            continue
        df = df[df[TARGET].notna()]
        g = df.groupby("target_date")
        parts.append(pd.DataFrame({"actual": g[TARGET].max(), "expected": g["pred"].max(),
                                   "base": g["pred_hi_raw"].max(), "hours": g[TARGET].count()}))
    if not parts:
        return None
    d = pd.concat(parts)
    d = d[~d.index.duplicated(keep="last")].sort_index()
    d = d[d["hours"] >= 20]                    # skip partial days
    d["score"] = d["actual"] - d["base"]       # how far the peak beat the raw base
    return d


PLAN_GAMMA = 0.01        # how fast the risk level adapts after misses or hits


def plan_margin(hist, day, level):
    """Margin for `day` at quantile `level`, using only days complete by its forecast
    origin (D-2 and earlier)."""
    newest = pd.Timestamp(day) - pd.Timedelta(days=2)
    w = hist[(hist.index <= newest) & (hist.index > newest - pd.Timedelta(days=PLAN_WINDOW_DAYS))]
    if len(w) < MIN_PLAN_DAYS:
        return None
    return float(np.quantile(w["score"], min(max(level, 0.5), 0.999)))


def _apply_updates(alpha, pending, day, risk):
    """Adaptive conformal step: after each observed day, tighten the level if the peak
    went above the plan, relax it slightly if not. Days only count once they're
    complete by the forecast origin, two days before `day`."""
    for d in sorted(k for k in pending if k <= pd.Timestamp(day) - pd.Timedelta(days=2)):
        alpha += PLAN_GAMMA * (risk - pending.pop(d))
    return alpha


@st.cache_data(ttl=3600)
def plan_replay(risk):
    """Replay the planning rule day by day through history, out of sample."""
    hist = daily_peak_history()
    if hist is None:
        return None
    alpha, pending, rows = risk, {}, []
    for day, r in hist.iterrows():
        alpha = _apply_updates(alpha, pending, day, risk)
        m = plan_margin(hist, day, 1 - alpha)
        if m is None:
            continue
        plan = max(r["base"] + m, r["expected"])
        miss = float(r["actual"] > plan)
        pending[day] = miss
        rows.append({"day": day, "actual": r["actual"], "expected": r["expected"], "plan": plan})
    return {"daily": pd.DataFrame(rows).set_index("day") if rows else None,
            "alpha": alpha, "pending": pending}


@st.cache_data(ttl=3600)
def plan_backtest(risk):
    rep = plan_replay(risk)
    if rep is None or rep["daily"] is None:
        return None
    d = rep["daily"].copy()
    d["buffer"] = d["plan"] - d["expected"]
    d["exceeded"] = d["actual"] > d["plan"]
    rate = d["exceeded"].mean()
    short = (d["actual"] - d["plan"])[d["exceeded"]]
    # A single fixed buffer on the expected peak, sized for the same miss rate
    fixed = float(np.quantile(d["actual"] - d["expected"], 1 - rate)) if 0 < rate < 1 else np.nan
    return {"daily": d, "rate": rate, "avg_buffer": d["buffer"].mean(),
            "avg_buffer_pct": (d["buffer"] / d["expected"]).mean() * 100,
            "avg_shortfall": short.mean() if len(short) else 0.0,
            "max_shortfall": short.max() if len(short) else 0.0, "fixed_buffer": fixed}


def plan_for_day(lf, risk):
    """Plan-for peak for one forecast day; falls back to the top of the 80% range."""
    hist, rep = daily_peak_history(), plan_replay(risk)
    if hist is None or rep is None or "pred_hi_raw" not in lf:
        return float(lf["pred_hi"].max()), False
    day = lf["target_date"].iloc[0]
    alpha = _apply_updates(rep["alpha"], dict(rep["pending"]), day, risk)
    m = plan_margin(hist, day, 1 - alpha)
    if m is None:
        return float(lf["pred_hi"].max()), False
    return float(max(lf["pred_hi_raw"].max() + m, lf["pred"].max())), True


def iso_complete(df):
    """ISO-NE posts tomorrow's forecast during the day; treat it as missing until
    it covers most of the day, so a few overnight hours aren't read as the peak."""
    return "iso_forecast_mw" in df and df["iso_forecast_mw"].notna().sum() >= 20


FULL_LEGEND = [("Actual", INK, "line"), ("Our forecast", OURS, "line"),
               ("80% range", BAND, "band"), ("ISO-NE's forecast", ISO, "dash")]


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------
def page_tomorrow():
    lf = load("latest_forecast.csv")
    if lf is None or lf.empty:
        st.title("Tomorrow's forecast")
        st.info("No forecast has been issued yet. Run `python forecast_tomorrow.py`, "
                "or open the repo's Actions tab and run **Daily day-ahead forecast**.")
        return
    peak = lf.loc[lf["pred"].idxmax()]
    t = peak["time"]
    st.title(f"{t:%A, %B %-d}: New England demand peaks near "
             f"{peak['pred']:,.0f} MW around {t:%-I %p}")
    issued = pd.Timestamp(lf["issued_at"].iloc[0])
    risk_label = st.session_state.get("risk", "1 day in 20")
    risk = RISK_LEVELS[risk_label]
    plan, calibrated = plan_for_day(lf, risk)
    pb = plan_backtest(risk) if calibrated else None
    track = (f" Sized so the actual peak goes above it about {risk_label} (backtest: "
             f"{pb['rate'] * 100:.0f}% of days)." if pb else "")
    st.markdown(
        f"Plan for up to **{plan:,.0f} MW**, a {plan - peak['pred']:,.0f} MW buffer above the "
        f"expected peak.{track} Issued {issued:%B %-d at %-I:%M %p} ET, before any of these "
        f"hours happened.")

    show_iso = iso_complete(lf)
    items = FULL_LEGEND[1:3] + ([FULL_LEGEND[3]] if show_iso else []) + [("Plan-for level", PEAK, "dash")]
    legend(items)
    peak_pt = alt.Chart(pd.DataFrame([peak])).encode(x="time:T", y="pred:Q")
    plan_df = pd.DataFrame({"plan": [plan], "label": [f"Plan for {plan:,.0f} MW"]})
    plan_rule = alt.Chart(plan_df).mark_rule(color=PEAK, strokeDash=[6, 4], strokeWidth=1.5).encode(
        y=alt.Y("plan:Q", scale=alt.Scale(zero=False)))
    plan_text = alt.Chart(plan_df).mark_text(align="left", dx=4, dy=-8, color=PEAK, fontSize=12).encode(
        y="plan:Q", x=alt.value(0), text="label:N")
    chart = (forecast_chart(lf, show_iso=show_iso) + plan_rule + plan_text
             + peak_pt.mark_point(color=PEAK, size=120, filled=True)
             + peak_pt.mark_text(dy=20, color=OURS, fontWeight="bold", fontSize=13).encode(
                 text=alt.value(f"Expected peak {peak['pred']:,.0f} MW")))
    st.altair_chart(chart, width="stretch")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Expected peak", f"{peak['pred']:,.0f} MW")
    c2.metric("Plan-for peak", f"{plan:,.0f} MW",
              delta=f"+{plan - peak['pred']:,.0f} MW buffer", delta_color="off")
    c3.metric("Expected peak hour", f"{t:%-I %p}")
    if show_iso:
        iso_peak = lf["iso_forecast_mw"].max()
        c4.metric("ISO-NE's peak forecast", f"{iso_peak:,.0f} MW",
                  delta=f"{iso_peak - peak['pred']:+,.0f} MW vs. ours", delta_color="off")
    else:
        c4.metric("ISO-NE's peak forecast", "Pending", delta="Full day not posted yet",
                  delta_color="off", delta_arrow="off")

    with st.expander("How this forecast is made"):
        st.markdown(
            "Each morning the model forecasts all 24 hours of the next day using only what's "
            "known at that moment: the calendar, the weather forecast for eight New England "
            "cities weighted by population, demand through two days ago, and this morning's "
            "demand so far. A calibration step corrects recent bias and sizes the 80% range "
            "from how often recent forecasts missed. The plan-for level starts from the day's "
            "highest 90th-percentile hour and adds a margin learned from the last 90 days of "
            "actual peaks, sized to the risk level chosen in the sidebar. Models were trained "
            "on data through "
            f"{lf['models_trained_through'].iloc[0]}. See **How it works** for details.")


def page_track_record():
    st.title("Live track record")
    st.markdown("Every forecast is saved before the day it predicts. Once actual demand comes in, "
                "it's scored here alongside ISO-NE's own day-ahead forecast.")
    log = load("forecast_log.csv")
    scored = log[log[TARGET].notna()] if log is not None else pd.DataFrame()
    if scored.empty:
        msg = "No scored days yet. Each forecast is checked against actual demand the day after it covers."
        if log is not None and not log.empty:
            msg += f" The first forecast covers {log['target_date'].min():%B %-d}."
        st.info(msg)
        return

    inside = (scored[TARGET] >= scored["pred_lo"]) & (scored[TARGET] <= scored["pred_hi"])
    iso_m = mape(scored[TARGET], scored["iso_forecast_mw"])
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Days scored", f"{scored['target_date'].nunique()}")
    c2.metric("Our average error", f"{mape(scored[TARGET], scored['pred']):.2f}%")
    c3.metric("ISO-NE's average error", "n/a" if np.isnan(iso_m) else f"{iso_m:.2f}%")
    c4.metric("Hours inside 80% range", f"{inside.mean() * 100:.0f}%")

    daily = pd.DataFrame({"Ours": mape_by(scored, "target_date", "pred"),
                          "ISO-NE": mape_by(scored, "target_date", "iso_forecast_mw")})
    long = daily.rename_axis("day").reset_index().melt("day", var_name="Forecast",
                                                       value_name="Error").dropna()
    st.subheader("Daily error")
    st.altair_chart(alt.Chart(long).mark_line(point=True).encode(
        x=alt.X("day:T", title=None, axis=alt.Axis(format="%b %-d", tickCount="day", labelAngle=0)),
        y=alt.Y("Error:Q", title="Average error (%)"),
        color=alt.Color("Forecast:N", scale=alt.Scale(domain=["Ours", "ISO-NE"], range=[OURS, ISO]),
                        legend=alt.Legend(orient="top", title=None)),
        tooltip=[alt.Tooltip("day:T", format="%b %-d"), "Forecast:N",
                 alt.Tooltip("Error:Q", format=".2f")],
    ).properties(height=260), width="stretch")

    st.subheader("Forecast vs. what happened")
    days = sorted(scored["target_date"].dt.date.unique(), reverse=True)
    pick = st.selectbox("Day", days, format_func=lambda d: f"{d:%A, %B %-d, %Y}")
    comparison_view(scored[scored["target_date"].dt.date == pick], key="cmp_day", height=320)


def page_accuracy():
    st.title("How accurate is it?")
    metrics, bt = load("metrics.csv"), load("backtest.csv")
    summary = load("backtest_summary.json") or {}
    if metrics is None or bt is None:
        st.info("Backtest results not found. Run `python train_dayahead.py` to create them.")
        return
    metrics = metrics.set_index(metrics.columns[0]).rename_axis("Forecast")
    ours = "Model + calibration (ours)" if "Model + calibration (ours)" in metrics.index \
        else "Ensemble + calibration (ours)"
    iso, naive = ("ISO-NE day-ahead forecast",
                        "Seasonal naive (same hour last week)")
    m = metrics["MAPE (%)"]
    text = (f"Replaying a full year ({summary.get('backtest_start', '?')} to "
            f"{summary.get('backtest_end', '?')}) exactly as it would have run live (retrain, "
            f"forecast the next day, step forward), the model's average hourly error was "
            f"**{m[ours]:.2f}%**, compared with **{m[naive]:.2f}%** for a naive "
            f"\"same as last week\" forecast.")
    if iso in m:
        text += (f" ISO-NE's own day-ahead forecast, built with far more data, averaged "
                 f"**{m[iso]:.2f}%**.")
    if "interval_coverage_pct" in summary:
        text += f" The 80% range contained actual demand in **{summary['interval_coverage_pct']}%** of hours."
    st.markdown(text)

    show = metrics.rename(index={ours: "Our model", "Ensemble (raw)": "Our model, before calibration",
                                 "Model (raw)": "Our model, before calibration",
                                 iso: "ISO-NE's day-ahead forecast", naive: "Naive: same hour last week"})
    order = [r for r in ["Our model", "ISO-NE's day-ahead forecast", "Naive: same hour last week",
                         "Our model, before calibration", "LightGBM", "XGBoost"] if r in show.index]
    st.dataframe(fmt_table(show.loc[order]), width="stretch")

    perfect = load("metrics_perfect_weather.csv")
    if perfect is not None:
        p = perfect.set_index(perfect.columns[0])["MAPE (%)"].get(ours)
        if p is not None:
            st.caption(f"With perfect knowledge of the weather, the same model reaches {p:.2f}%. "
                       f"The difference is the cost of weather-forecast error.")

    if daily_peak_history() is not None:
        st.subheader("Planning buffer")
        st.markdown(
            "Running short of power costs far more than holding a little extra, so planners use a "
            "higher number than the expected peak. The plan-for level here starts from the day's "
            "highest 90th-percentile hour, then adds a margin learned from the previous 90 days: "
            "how far actual peaks landed above that level. The margin also adapts: after a day "
            "when the peak went above the plan it tightens, and after covered days it relaxes "
            "slightly, which keeps it on target through heat waves and cold snaps when misses "
            "come in streaks. Each day is sized using only days already observed, so these "
            "results are out of sample.")
        rows = {}
        for label, r in RISK_LEVELS.items():
            pb = plan_backtest(r)
            if pb:
                rows[label] = {"Target miss rate (%)": r * 100, "Actual miss rate (%)": pb["rate"] * 100,
                               "Average buffer (MW)": pb["avg_buffer"],
                               "Average buffer (%)": pb["avg_buffer_pct"],
                               "Average shortfall on missed days (MW)": pb["avg_shortfall"],
                               "Same miss rate with a fixed buffer (MW)": pb["fixed_buffer"]}
        if rows:
            tbl = pd.DataFrame(rows).T.rename_axis("Accepted risk")
            st.dataframe(tbl.style.format({c: "{:,.0f}" if "MW" in c else "{:.1f}" for c in tbl.columns}),
                         width="stretch")
            st.caption("A miss means the actual daily peak went above the plan-for level. The fixed-buffer "
                       "column shows what a single, same-size cushion every day would need to match that "
                       "miss rate; when it's larger than the average buffer, sizing by uncertainty is "
                       "the more efficient choice.")
        risk_label = st.session_state.get("risk", "1 day in 20")
        pb = plan_backtest(RISK_LEVELS[risk_label])
        if pb:
            d = pb["daily"]
            plot = d.reset_index()
            long = plot.melt("day", ["actual", "expected", "plan"], var_name="series", value_name="MW")
            names = {"actual": "Actual peak", "expected": "Expected peak", "plan": "Plan-for peak"}
            long["series"] = long["series"].map(names)
            dom = list(names.values())
            lines = alt.Chart(long).mark_line(strokeWidth=1.4).encode(
                x=alt.X("day:T", title=None),
                y=alt.Y("MW:Q", title="Daily peak (MW)", scale=alt.Scale(zero=False)),
                color=alt.Color("series:N", scale=alt.Scale(domain=dom, range=[INK, OURS, PEAK]),
                                legend=alt.Legend(orient="top", title=None)),
                strokeDash=alt.StrokeDash("series:N", scale=alt.Scale(
                    domain=dom, range=[[1, 0], [1, 0], [5, 3]]), legend=None),
                tooltip=[alt.Tooltip("day:T", format="%b %-d, %Y"), "series:N",
                         alt.Tooltip("MW:Q", format=",.0f")])
            misses = alt.Chart(plot[plot["exceeded"]]).mark_point(
                color="#B03A2E", size=50, filled=True).encode(
                x="day:T", y="actual:Q",
                tooltip=[alt.Tooltip("day:T", format="%b %-d, %Y", title="Day above plan"),
                         alt.Tooltip("actual:Q", format=",.0f", title="Actual peak"),
                         alt.Tooltip("plan:Q", format=",.0f", title="Plan-for")])
            st.altair_chart((lines + misses).properties(height=300), width="stretch")
            st.caption(f"Showing the {risk_label.lower()} setting (change it in the sidebar). Red dots "
                       f"mark days when the actual peak went above the plan-for level. The right risk "
                       f"level depends on the cost of a shortfall compared with idle capacity: if running "
                       f"short costs 19 times as much, accepting a miss about one day in 20 is the "
                       f"cost-minimizing choice.")

    st.subheader("Look at any week")
    lo_d, hi_d = bt["target_date"].min().date(), bt["target_date"].max().date()
    wk = st.date_input("Week starting", value=hi_d - pd.Timedelta(days=6),
                       min_value=lo_d, max_value=hi_d - pd.Timedelta(days=6))
    sel = bt[(bt["target_date"].dt.date >= wk) & (bt["target_date"].dt.date < wk + pd.Timedelta(days=7))]
    comparison_view(sel, key="cmp_week")

    st.subheader("Error by hour of day")
    bt["hour"] = bt["time"].dt.hour
    cols = {"Ours": "pred"}
    if "iso_forecast_mw" in bt and bt["iso_forecast_mw"].notna().any():
        cols["ISO-NE"] = "iso_forecast_mw"
    by_hour = pd.DataFrame({k: mape_by(bt, "hour", c) for k, c in cols.items()})
    long = by_hour.rename_axis("hour").reset_index().melt("hour", var_name="Forecast",
                                                          value_name="Error").dropna()
    st.altair_chart(alt.Chart(long).mark_line(point=True).encode(
        x=alt.X("hour:O", title="Hour of day", axis=alt.Axis(labelAngle=0)),
        y=alt.Y("Error:Q", title="Average error (%)"),
        color=alt.Color("Forecast:N", scale=alt.Scale(domain=["Ours", "ISO-NE"], range=[OURS, ISO]),
                        legend=alt.Legend(orient="top", title=None)),
        tooltip=["hour:O", "Forecast:N", alt.Tooltip("Error:Q", format=".2f")],
    ).properties(height=260), width="stretch")
    st.caption("Rooftop solar isn't metered by ISO-NE, so sunny middays show up as lower demand "
               "and are harder to predict. The solar forecast inputs target this.")

    seasons = load("season_metrics.csv")
    if seasons is not None:
        st.subheader("By season")
        seasons = seasons.set_index(seasons.columns[0]).rename_axis("Season").reindex(
            ["Winter", "Spring", "Summer", "Fall"]).dropna(how="all")
        st.dataframe(fmt_table(seasons), width="stretch")


def page_drivers():
    st.title("What drives demand?")
    sv, sf = load("shap_values.npy"), load("shap_features.csv")
    if sv is None or sf is None:
        st.info("Explainability files not found. Run `python train_dayahead.py` with `shap` installed.")
        return
    feat_cols = [c for c in sf.columns if c not in NOT_FEATURES][:sv.shape[1]]
    imp = pd.DataFrame({"Input": [FRIENDLY.get(c, c) for c in feat_cols],
                        "Impact": np.abs(sv).mean(axis=0)}).nlargest(12, "Impact")
    agree = None
    if os.path.exists(f"{DATA}/shap_agreement.txt"):
        with open(f"{DATA}/shap_agreement.txt") as f:
            agree = f.read().strip()
    note = (f" Explained with a companion tree model that matches the forecast model's "
            f"predictions closely (R² = {agree}), since SHAP can't be computed directly for "
            f"linear trees." if agree else "")
    st.markdown("How much each input moves the forecast on a typical hour over the past year "
                "(average absolute SHAP value)." + note)
    st.altair_chart(alt.Chart(imp).mark_bar(color=OURS).encode(
        x=alt.X("Impact:Q", title="Average effect on the forecast (MW)"),
        y=alt.Y("Input:N", sort="-x", title=None, axis=alt.Axis(labelLimit=300)),
        tooltip=["Input", alt.Tooltip("Impact:Q", format=",.0f")],
    ).properties(height=360), width="stretch")

    sf["Season"] = season_of(sf["time"].dt.month)
    season_scale = alt.Scale(domain=list(SEASON_COLORS), range=list(SEASON_COLORS.values()))
    if "temp_c" in sf:
        st.subheader("Heating and cooling")
        st.markdown("Demand rises on both sides of a mild 15–18 °C: heating on cold days, air "
                    "conditioning on hot ones. The hot side climbs much faster.")
        sample = sf.sample(min(len(sf), 4000), random_state=0)
        st.altair_chart(alt.Chart(sample).mark_circle(size=12, opacity=0.45).encode(
            x=alt.X("temp_c:Q", title="Forecast temperature (°C, population-weighted)"),
            y=alt.Y(f"{TARGET}:Q", title="Demand (MW)", scale=alt.Scale(zero=False)),
            color=alt.Color("Season:N", scale=season_scale, legend=alt.Legend(orient="top", title=None)),
        ).properties(height=340), width="stretch")

    st.subheader("When the extreme peaks happen")
    top = sf[sf[TARGET] >= sf[TARGET].quantile(0.95)].copy()
    top["Hour"] = top["time"].dt.hour
    if not top.empty:
        weekday = (top["time"].dt.dayofweek < 5).mean() * 100
        share = top["Season"].value_counts(normalize=True).mul(100).round().astype(int)
        mix = ", ".join(f"{s.lower()} {share[s]}%" for s in share.index)
        st.markdown(f"The top 5% of demand hours split {mix}. They cluster in the late afternoon "
                    f"and early evening, and {weekday:.0f}% fall on weekdays.")
        st.altair_chart(alt.Chart(top).mark_bar().encode(
            x=alt.X("Hour:O", title="Hour of day", axis=alt.Axis(labelAngle=0)),
            y=alt.Y("count():Q", title="Extreme-demand hours"),
            color=alt.Color("Season:N", scale=season_scale, legend=alt.Legend(orient="top", title=None)),
        ).properties(height=260), width="stretch")


def page_method():
    st.title("How it works")
    st.markdown("""
**The question.** How much electricity will New England use tomorrow, hour by hour, and when
will demand peak? Grid operators answer this every morning, and peaks drive capacity costs
that end up on ratepayers' bills.

**The setup.** At about 10 AM each day, the model forecasts all 24 hours of the next day. It
uses only information that exists at that moment:

- calendar facts: hour, weekday, holidays, season
- the **weather forecast** for tomorrow (temperature, dew point, humidity, wind, clouds and
  sunshine) for eight New England cities, weighted by population
- demand through two days ago, plus this morning's demand so far

**Data.** Hourly ISO-NE demand and ISO-NE's own day-ahead forecast come from the U.S. EIA's
EIA-930 API. Weather comes from Open-Meteo's archive of past weather *forecasts*, meaning what
the weather models predicted the day before. Training therefore sees the same forecast error
the live model faces.

**Models.** A single regularized LightGBM model with linear trees: each leaf fits a small linear
trend instead of a constant, so record cold snaps and heat waves aren't flattened. The settings
were fixed in advance, judged on a year not used to choose them, and confirmed on the next
year, where the model beat the previous version in most months with about half the gap between
training and test error. LightGBM quantile models provide the 80% range. A calibration step, using only days already observed, corrects recent
bias and resizes the range so it really covers 80% of hours.

**Planning buffer.** Shortfalls cost far more than spare capacity, so the app also reports a
plan-for peak. It starts from the day's highest 90th-percentile hour, then adds a margin learned
from the previous 90 days of actual peaks, sized so the peak goes above the plan at a chosen
rate (one day in 10, 20 or 50). The margin adapts as it goes, tightening after misses and
relaxing after covered days (adaptive conformal prediction), which holds the miss rate on
target through heat waves and cold snaps. The buffer is larger on uncertain days and smaller on calm ones,
and the expected forecast stays unbiased, so accuracy comparisons remain fair.

**Testing.** A rolling-origin backtest over the most recent year: retrain every 28 days,
forecast each day out of sample, step forward. The benchmarks are a naive "same hour last week"
forecast and ISO-NE's published day-ahead forecast.

**Running live.** A GitHub Action issues tomorrow's forecast every morning and saves it before
the actuals exist, and the Live track record page scores it as the days pass. The models
retrain monthly.

**What changed from version 1.** The first version reported a 1.27% error, but it used the
previous hour's actual demand, a 24-hour average that included the hour being predicted, and
observed rather than forecast weather. That made it a one-hour-ahead nowcast. Rebuilding it as
a true day-ahead forecast raised the error to about 4%, the honest number, and made a fair
comparison with the grid operator possible.
""")
    st.caption("Code: github.com/cathykam2006/Massachusetts-Peak-Electricity-Demand-Forecasting. "
               "Data: U.S. EIA; weather by Open-Meteo.com (CC BY 4.0).")


# ---------------------------------------------------------------------------
PAGES = {
    "Tomorrow's forecast": page_tomorrow,
    "Live track record": page_track_record,
    "Accuracy": page_accuracy,
    "What drives demand": page_drivers,
    "How it works": page_method,
}
st.sidebar.markdown("### ⚡ New England demand forecast")
choice = st.sidebar.radio("Page", list(PAGES), label_visibility="collapsed")
st.sidebar.radio("Planning risk: how often the actual peak may go above the plan",
                 list(RISK_LEVELS), index=1, key="risk")
st.sidebar.caption("Hourly day-ahead forecast for the ISO New England grid, updated every morning.")
PAGES[choice]()
