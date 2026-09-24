"""Streamlit fraud-detection dashboard.

Application layer only: loads the frozen Phase 7 pipeline and never retrains.
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from dashboard.components import (
    bar_rate,
    callout,
    confusion_heatmap,
    count_bar,
    friendly_error,
    histogram,
    inject_styles,
    metric_bars,
    prediction_panel,
    render_kpis,
    show_chart,
)
from dashboard.data_loader import (
    EXPECTED_RAW_DATASET_MD5,
    VALID_RAW_DATASET_MD5S,
    PHASE6_DUMMY_ACCURACY,
    PHASE6_DUMMY_RECALL,
    category_options,
    dataset_hash,
    fraud_rate_table,
    load_final_test_metrics,
    load_metadata,
    load_pipeline,
    load_raw_dataset,
    score_dataset,
    with_parsed_hour,
)
from dashboard.styles import BAND_COLORS
from src.data.loader import DATE_FORMAT, ValidationError
from src.features.feature_engineering import FEATURE_INPUT_COLUMNS
from src.models.predict import predict_transaction
from src.models.scoring import VALID_RISK_BAND_NAMES

PAGES = (
    "Overview",
    "Fraud Analytics",
    "Model Performance",
    "Transaction Prediction",
    "Risk Monitoring",
)


@st.cache_resource(show_spinner="Loading frozen production model...")
def cached_pipeline():
    return load_pipeline()


@st.cache_data(show_spinner="Loading dataset...")
def cached_raw() -> pd.DataFrame:
    return load_raw_dataset()


@st.cache_data(show_spinner="Scoring historical transactions with the frozen model...")
def cached_scored() -> pd.DataFrame:
    return score_dataset(cached_raw(), cached_pipeline())


@st.cache_data(show_spinner=False)
def cached_metadata() -> dict:
    return load_metadata()


@st.cache_data(show_spinner=False)
def cached_test_metrics() -> dict:
    return load_final_test_metrics()


def _fmt_pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def render_sidebar(metadata: dict, metrics: dict) -> str:
    st.sidebar.markdown("### Foresight")
    st.sidebar.caption("Financial Fraud Detection System using Machine Learning")
    page = st.sidebar.radio("Navigate", PAGES, index=0)
    st.sidebar.divider()
    st.sidebar.markdown("### Model information")
    st.sidebar.write(f"**Model:** {metadata.get('model_name', 'Logistic Regression')} + {metadata.get('strategy', 'Class Weight')}")
    st.sidebar.write(f"**Calibration:** {str(metadata.get('calibration_method', 'sigmoid')).title()} / 5-fold CV")
    st.sidebar.write(f"**Threshold:** {float(metadata['production_threshold']):.2f}")
    st.sidebar.write(f"**ROC-AUC:** {float(metrics['ROC_AUC']):.4f}")
    st.sidebar.write(f"**Recall:** {float(metrics['Recall']):.4f}")
    st.sidebar.caption(
        "Risk score is a model-generated risk indicator and is not proof of fraud."
    )
    st.sidebar.divider()
    st.sidebar.caption("Frozen Phase 7 artifact. The dashboard does not retrain.")
    return page


def page_overview(raw: pd.DataFrame, scored: pd.DataFrame) -> None:
    st.markdown('<div class="hero-kicker">Project Foresight</div>', unsafe_allow_html=True)
    st.title("Financial Fraud Detection System using Machine Learning")
    callout(
        "<strong>This system estimates fraud risk.</strong> It does not prove that a "
        "transaction is fraudulent. Flagged transactions should be reviewed by an analyst."
    )
    frauds = int((raw["Fraudulent"] == 1).sum())
    legit = int((raw["Fraudulent"] == 0).sum())
    flagged = int((scored["predicted_label"] == 1).sum())
    render_kpis(
        [
            ("Total transactions", f"{len(raw):,}", "Historical dataset"),
            ("Fraud transactions", f"{frauds:,}", "Historical ground truth"),
            ("Legitimate transactions", f"{legit:,}", "Historical ground truth"),
            ("Fraud rate", _fmt_pct(frauds / len(raw)), "482 / 5,000"),
            ("Transactions flagged", f"{flagged:,}", "Frozen model, full dataset"),
            ("Flagged percentage", _fmt_pct(flagged / len(scored)), f"Threshold {cached_metadata()['production_threshold']:.2f}"),
        ]
    )
    st.caption(
        "Flagged counts are model predictions on the historical file, not a new evaluation split. "
        "Official test metrics live on the Model Performance page."
    )

    labeled = with_parsed_hour(raw)
    c1, c2 = st.columns(2)
    with c1:
        counts = labeled["historical_label"].value_counts()
        fig = go.Figure(
            go.Pie(
                labels=counts.index.tolist(),
                values=counts.values.tolist(),
                hole=0.55,
                marker_colors=["#34d399", "#fb7185"],
            )
        )
        fig.update_layout(title="Historical fraud vs legitimate")
        show_chart(fig)
    with c2:
        fig = go.Figure()
        fig.add_histogram(x=labeled.loc[labeled["Fraudulent"] == 0, "Transaction_Amount"], name="Legitimate", opacity=0.7)
        fig.add_histogram(x=labeled.loc[labeled["Fraudulent"] == 1, "Transaction_Amount"], name="Fraud", opacity=0.7)
        fig.update_layout(barmode="overlay", title="Transaction amount distribution", xaxis_title="Amount")
        show_chart(fig)

    c3, c4 = st.columns(2)
    with c3:
        show_chart(bar_rate(fraud_rate_table(labeled, "Merchant_Category"), "Historical fraud rate by merchant category"))
    with c4:
        origin_table = fraud_rate_table(labeled, "origin")
        show_chart(bar_rate(origin_table, "Domestic vs international historical fraud rate"))


def page_analytics(raw: pd.DataFrame) -> None:
    st.title("Fraud analytics")
    callout(
        "Charts below use <strong>historical ground truth</strong> (`Fraudulent`). "
        "That column is never sent to the model. Sample sizes are shown on each bar."
    )
    labeled = with_parsed_hour(raw)
    f1, f2, f3 = st.columns(3)
    merchants = f1.multiselect("Merchant category", category_options(labeled, "Merchant_Category"))
    payments = f2.multiselect("Payment method", category_options(labeled, "Payment_Method"))
    devices = f3.multiselect("Device type", category_options(labeled, "Device_Type"))
    f4, f5 = st.columns(2)
    locations = f4.multiselect("Location", category_options(labeled, "Location"))
    origin = f5.multiselect("International / domestic", ["Domestic", "International"])

    filtered = labeled.copy()
    if merchants:
        filtered = filtered[filtered["Merchant_Category"].isin(merchants)]
    if payments:
        filtered = filtered[filtered["Payment_Method"].isin(payments)]
    if devices:
        filtered = filtered[filtered["Device_Type"].isin(devices)]
    if locations:
        filtered = filtered[filtered["Location"].isin(locations)]
    if origin:
        filtered = filtered[filtered["origin"].isin(origin)]

    st.caption(f"Filtered rows: {len(filtered):,} of {len(labeled):,}. Very small groups can produce noisy rates.")
    if filtered.empty:
        st.warning("No rows match the current filters.")
        return

    c1, c2 = st.columns(2)
    with c1:
        show_chart(bar_rate(fraud_rate_table(filtered, "Merchant_Category"), "Fraud rate by merchant category"))
        show_chart(bar_rate(fraud_rate_table(filtered, "Device_Type"), "Fraud rate by device type"))
        show_chart(bar_rate(fraud_rate_table(filtered, "origin"), "Domestic vs international fraud rate"))
    with c2:
        show_chart(bar_rate(fraud_rate_table(filtered, "Payment_Method"), "Fraud rate by payment method"))
        show_chart(bar_rate(fraud_rate_table(filtered, "Location"), "Fraud rate by location"))
        hour_table = fraud_rate_table(filtered.dropna(subset=["transaction_hour"]), "transaction_hour")
        hour_table = hour_table.sort_values("transaction_hour")
        hour_table["label"] = hour_table["transaction_hour"].astype(int).astype(str) + ":00 (n=" + hour_table["n"].astype(int).astype(str) + ")"
        show_chart(bar_rate(hour_table, "Fraud rate by transaction hour"))

    fig = go.Figure()
    fig.add_histogram(x=filtered.loc[filtered["Fraudulent"] == 0, "Transaction_Amount"], name="Legitimate", opacity=0.7)
    fig.add_histogram(x=filtered.loc[filtered["Fraudulent"] == 1, "Transaction_Amount"], name="Fraud", opacity=0.7)
    fig.update_layout(barmode="overlay", title="Fraud vs legitimate transaction amount", xaxis_title="Amount")
    show_chart(fig)


def page_performance(metrics: dict, threshold: float) -> None:
    st.title("Model performance")
    callout(
        "<strong>Accuracy alone is not sufficient for fraud detection</strong> because the "
        "dataset contains substantially more legitimate transactions. Figures on this page "
        "are the frozen Phase 7 one-shot test evaluation (1,000 rows, 96 frauds). "
        "They are not recalculated by the dashboard."
    )
    render_kpis(
        [
            ("Accuracy", _fmt_pct(float(metrics["Accuracy"])), "Test set"),
            ("Precision", _fmt_pct(float(metrics["Precision"])), "Test set"),
            ("Recall", _fmt_pct(float(metrics["Recall"])), "Test set"),
            ("F1", _fmt_pct(float(metrics["F1"])), "Test set"),
            ("ROC-AUC", f"{float(metrics['ROC_AUC']):.4f}", "Probabilities"),
            ("PR-AUC", f"{float(metrics['PR_AUC']):.4f}", "Imbalanced ranking"),
        ]
    )
    tn, fp, fn, tp = (int(metrics["TN"]), int(metrics["FP"]), int(metrics["FN"]), int(metrics["TP"]))
    c1, c2 = st.columns(2)
    with c1:
        show_chart(confusion_heatmap(tn, fp, fn, tp, threshold))
        st.caption("False negatives are missed fraud. False positives are legitimate transactions flagged for review.")
    with c2:
        show_chart(
            metric_bars(
                {
                    "Precision": float(metrics["Precision"]),
                    "Recall": float(metrics["Recall"]),
                    "F1": float(metrics["F1"]),
                },
                "Precision / recall / F1",
            )
        )
        show_chart(
            metric_bars(
                {"ROC-AUC": float(metrics["ROC_AUC"]), "PR-AUC": float(metrics["PR_AUC"])},
                "Ranking metrics",
            )
        )

    st.subheader("Why accuracy is misleading")
    show_chart(
        metric_bars(
            {
                "Always legitimate accuracy": PHASE6_DUMMY_ACCURACY,
                "Always legitimate recall": PHASE6_DUMMY_RECALL,
                "Production accuracy": float(metrics["Accuracy"]),
                "Production recall": float(metrics["Recall"]),
            },
            "Phase 6 dummy vs frozen production model (test set)",
        )
    )
    st.markdown(
        f"An **Always Legitimate** rule on the untouched test set is already "
        f"**{_fmt_pct(PHASE6_DUMMY_ACCURACY)} accurate** with **{_fmt_pct(PHASE6_DUMMY_RECALL)} recall**. "
        "It scores well on accuracy while catching zero fraud. The production model is less "
        "accurate but recovers 76.04% of test frauds."
    )


def page_predict(raw: pd.DataFrame, pipeline, metadata: dict) -> None:
    st.title("Transaction prediction")
    callout(
        "The form sends only production inference fields to <code>predict_transaction()</code>. "
        "Customer_ID, Transaction_ID, Suspicious_Keyword, and Fraudulent are not collected."
    )
    threshold = float(metadata["production_threshold"])
    merchants = category_options(raw, "Merchant_Category")
    payments = category_options(raw, "Payment_Method")
    devices = category_options(raw, "Device_Type")
    locations = category_options(raw, "Location")

    with st.form("score_transaction"):
        c1, c2 = st.columns(2)
        amount = c1.number_input("Transaction amount", min_value=0.0, value=120.0, step=1.0)
        average_spend = c2.number_input("Average spend", min_value=0.0, value=95.0, step=1.0)
        previous = c1.number_input("Previous transactions", min_value=0, value=12, step=1)
        account_age = c2.number_input("Account age (days)", min_value=0, value=400, step=1)
        tx_date = c1.date_input("Transaction date", value=datetime(2023, 6, 15).date())
        tx_time = c2.time_input("Transaction time", value=datetime(2023, 6, 15, 14, 30).time())
        merchant = c1.selectbox("Merchant category", merchants)
        payment = c2.selectbox("Payment method", payments)
        device = c1.selectbox("Device type", devices)
        location = c2.selectbox("Location", locations)
        international = c1.selectbox("International transaction", ["Domestic (0)", "International (1)"])
        submitted = st.form_submit_button("Score transaction", type="primary")

    if not submitted:
        st.info("Submit a transaction to score it with the frozen production pipeline.")
        return

    stamp = datetime.combine(tx_date, tx_time).strftime(DATE_FORMAT)
    record = {
        "Transaction_Date": stamp,
        "Transaction_Amount": float(amount),
        "Average_Spend": float(average_spend),
        "Previous_Transactions": int(previous),
        "Account_Age_Days": int(account_age),
        "Is_International": 1 if international.startswith("International") else 0,
        "Merchant_Category": merchant,
        "Payment_Method": payment,
        "Device_Type": device,
        "Location": location,
    }
    missing = [column for column in FEATURE_INPUT_COLUMNS if column not in record]
    if missing:
        st.error("Missing required inference fields: " + ", ".join(missing))
        return
    try:
        result = predict_transaction(record, pipeline=pipeline)
    except (ValidationError, ValueError, FileNotFoundError, TypeError) as exc:
        friendly_error(exc)
        return
    except Exception as exc:
        friendly_error(exc)
        return
    prediction_panel(result, threshold)


def page_monitor(scored: pd.DataFrame, metadata: dict) -> None:
    st.title("Risk monitoring")
    callout(
        "<strong>Model prediction</strong> comes from the frozen pipeline. "
        "<strong>Historical ground truth</strong> is the dataset label and is shown only for analytics. "
        "It is never used as a model input."
    )
    threshold = float(metadata["production_threshold"])
    band_counts = scored["risk_band"].value_counts()
    flagged = int((scored["predicted_label"] == 1).sum())
    render_kpis(
        [
            ("Total transactions", f"{len(scored):,}", "Historical file"),
            ("Low risk", f"{int(band_counts.get('Low Risk', 0)):,}", "Score 0-9"),
            ("Medium risk", f"{int(band_counts.get('Medium Risk', 0)):,}", "Score 10-19"),
            ("High risk", f"{int(band_counts.get('High Risk', 0)):,}", "Score 20-39"),
            ("Critical risk", f"{int(band_counts.get('Critical Risk', 0)):,}", "Score 40-100"),
            ("Flagged", f"{flagged:,} ({_fmt_pct(flagged / len(scored))})", f"probability ≥ {threshold:.2f}"),
        ]
    )

    c1, c2 = st.columns(2)
    with c1:
        order = list(VALID_RISK_BAND_NAMES)
        values = [int(band_counts.get(name, 0)) for name in order]
        colors = [BAND_COLORS[name] for name in order]
        show_chart(count_bar(order, values, "Risk-band distribution", colors))
        show_chart(histogram(scored["predicted_probability"], "Fraud probability distribution", "Predicted probability"))
    with c2:
        show_chart(histogram(scored["risk_score"], "Risk-score distribution", "Risk score (0-100)", "#a78bfa"))
        flag_labels = ["Flag for review", "Legitimate"]
        flag_values = [
            int((scored["predicted_label"] == 1).sum()),
            int((scored["predicted_label"] == 0).sum()),
        ]
        show_chart(count_bar(flag_labels, flag_values, "Flagged vs not flagged", ["#fb7185", "#34d399"]))

    crosstab = pd.crosstab(scored["risk_band"], scored["Historical Ground Truth"])
    fig = go.Figure()
    if 0 in crosstab.columns:
        fig.add_bar(name="Historical legitimate", x=crosstab.index.astype(str), y=crosstab[0])
    if 1 in crosstab.columns:
        fig.add_bar(name="Historical fraud", x=crosstab.index.astype(str), y=crosstab[1])
    fig.update_layout(barmode="group", title="Risk band vs actual historical outcome", yaxis_title="Transactions")
    show_chart(fig)

    st.subheader("Scored transactions")
    m1, m2, m3, m4, m5 = st.columns(5)
    band_filter = m1.multiselect("Risk band", list(VALID_RISK_BAND_NAMES))
    pred_filter = m2.multiselect("Predicted label", ["Flag for review", "Legitimate"])
    origin_filter = m3.multiselect("International / domestic", ["Domestic", "International"])
    merchant_filter = m4.multiselect("Merchant category", category_options(scored, "Merchant_Category"))
    payment_filter = m5.multiselect("Payment method", category_options(scored, "Payment_Method"))

    table = scored.copy()
    if band_filter:
        table = table[table["risk_band"].isin(band_filter)]
    if pred_filter:
        table = table[table["prediction_text"].isin(pred_filter)]
    if origin_filter:
        table = table[table["origin"].isin(origin_filter)]
    if merchant_filter:
        table = table[table["Merchant_Category"].isin(merchant_filter)]
    if payment_filter:
        table = table[table["Payment_Method"].isin(payment_filter)]

    view = table[
        [
            "Transaction_ID",
            "Transaction_Date",
            "Transaction_Amount",
            "Merchant_Category",
            "Payment_Method",
            "Device_Type",
            "Location",
            "Is_International",
            "predicted_probability",
            "risk_score",
            "risk_band",
            "predicted_label",
            "prediction_text",
            "Historical Ground Truth",
        ]
    ].copy()
    st.dataframe(view, width="stretch", hide_index=True, height=420)
    csv_bytes = view.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download scored results CSV",
        data=csv_bytes,
        file_name="scored_transactions.csv",
        mime="text/csv",
    )
    st.caption("Download does not modify data/raw/financial_fraud_detection_dataset.csv.")


def main() -> None:
    st.set_page_config(
        page_title="Foresight Fraud Detection",
        page_icon="◎",
        layout="wide",
    )
    inject_styles()
    try:
        digest = dataset_hash()
        if digest not in VALID_RAW_DATASET_MD5S and digest != EXPECTED_RAW_DATASET_MD5:
            st.error("The raw dataset hash has changed. The dashboard will not continue.")
            return
        metadata = cached_metadata()
        metrics = cached_test_metrics()
        raw = cached_raw()
        pipeline = cached_pipeline()
        scored = cached_scored()
    except Exception as exc:
        friendly_error(exc)
        st.stop()
        return

    page = render_sidebar(metadata, metrics)
    try:
        if page == "Overview":
            page_overview(raw, scored)
        elif page == "Fraud Analytics":
            page_analytics(raw)
        elif page == "Model Performance":
            page_performance(metrics, float(metadata["production_threshold"]))
        elif page == "Transaction Prediction":
            page_predict(raw, pipeline, metadata)
        else:
            page_monitor(scored, metadata)
    except (ValidationError, ValueError, FileNotFoundError, TypeError, KeyError) as exc:
        friendly_error(exc)
    except Exception as exc:
        friendly_error(exc)


if __name__ == "__main__":
    main()
