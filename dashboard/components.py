"""Reusable Streamlit UI pieces and Plotly charts."""

from __future__ import annotations

from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from dashboard.styles import BAND_COLORS, CSS, apply_theme


def inject_styles() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def kpi_card(label: str, value: str, hint: str = "") -> str:
    hint_html = f'<div class="kpi-hint">{hint}</div>' if hint else ""
    return (
        f'<div class="kpi-card"><div class="kpi-label">{label}</div>'
        f'<div class="kpi-value">{value}</div>{hint_html}</div>'
    )


def render_kpis(items: list[tuple[str, str, str]]) -> None:
    cards = "".join(kpi_card(label, value, hint) for label, value, hint in items)
    st.markdown(f'<div class="kpi-grid">{cards}</div>', unsafe_allow_html=True)


def callout(text: str) -> None:
    st.markdown(f'<div class="callout">{text}</div>', unsafe_allow_html=True)


def friendly_error(exc: Exception) -> None:
    message = str(exc).strip() or exc.__class__.__name__
    st.error(message)


def show_chart(fig: go.Figure) -> None:
    st.plotly_chart(apply_theme(fig), width="stretch", config={"displayModeBar": False})


def bar_rate(table: pd.DataFrame, title: str, y_label: str = "Historical fraud rate") -> go.Figure:
    fig = go.Figure(
        go.Bar(
            x=table["label"],
            y=table["fraud_rate"],
            customdata=table[["n", "frauds"]],
            hovertemplate="%{x}<br>Fraud rate=%{y:.1%}<br>Frauds=%{customdata[1]}<extra></extra>",
            marker_color="#22d3ee",
        )
    )
    fig.update_layout(title=title, xaxis_title="", yaxis_title=y_label, yaxis_tickformat=".0%")
    return fig


def count_bar(labels: list[str], values: list[int], title: str, colors: list[str] | None = None) -> go.Figure:
    fig = go.Figure(go.Bar(x=labels, y=values, marker_color=colors or "#22d3ee"))
    fig.update_layout(title=title, xaxis_title="", yaxis_title="Transactions")
    return fig


def histogram(series: pd.Series, title: str, x_title: str, color: str = "#22d3ee") -> go.Figure:
    fig = go.Figure(go.Histogram(x=series, nbinsx=40, marker_color=color))
    fig.update_layout(title=title, xaxis_title=x_title, yaxis_title="Transactions")
    return fig


def confusion_heatmap(tn: int, fp: int, fn: int, tp: int) -> go.Figure:
    fig = go.Figure(
        go.Heatmap(
            z=[[tn, fp], [fn, tp]],
            x=["Predicted legitimate", "Predicted fraud / flag"],
            y=["Actual legitimate", "Actual fraud"],
            text=[[f"TN {tn}", f"FP {fp}"], [f"FN {fn}", f"TP {tp}"]],
            texttemplate="%{text}",
            colorscale="Teal",
            showscale=False,
        )
    )
    fig.update_layout(title="Untouched test-set confusion matrix (threshold = 0.10)", height=420)
    return fig


def metric_bars(metrics: dict[str, float], title: str) -> go.Figure:
    names = list(metrics.keys())
    values = list(metrics.values())
    fig = go.Figure(go.Bar(x=names, y=values, marker_color="#a78bfa"))
    fig.update_layout(title=title, yaxis=dict(range=[0, 1], tickformat=".0%"), xaxis_title="")
    return fig


def prediction_panel(result: dict[str, Any], threshold: float) -> None:
    probability = float(result["predicted_probability"])
    score = float(result["risk_score"])
    band = str(result["risk_band"])
    flagged = int(result["predicted_label"]) == 1
    decision = "FLAG FOR REVIEW" if flagged else "LEGITIMATE / BELOW THRESHOLD"
    badge = "flag-on" if flagged else "flag-off"
    st.markdown(
        f"""
        <div class="pred-panel">
            <div class="pred-label">Fraud probability</div>
            <div class="pred-value">{probability * 100:.1f}%</div>
            <div class="pred-label">Risk score</div>
            <div class="pred-value">{score:.1f} / 100</div>
            <div class="pred-label">Risk band</div>
            <div class="pred-value">{band.upper()}</div>
            <div class="pred-label">Prediction (threshold {threshold:.2f})</div>
            <div class="{badge}">{decision}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption(
        "Risk score is a model-generated risk indicator and is not proof of fraud. "
        "Flagged transactions should be reviewed."
    )
