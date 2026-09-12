"""Shared visual theme for the Streamlit dashboard."""

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(15,23,42,0.35)",
    font=dict(color="#e2e8f0", family="Source Sans 3, Segoe UI, sans-serif", size=13),
    legend=dict(bgcolor="rgba(0,0,0,0)", orientation="h", yanchor="bottom", y=1.02),
    margin=dict(l=40, r=20, t=60, b=40),
    colorway=["#22d3ee", "#fb7185", "#a78bfa", "#34d399", "#fbbf24", "#60a5fa"],
)

BAND_COLORS = {
    "Low Risk": "#34d399",
    "Medium Risk": "#fbbf24",
    "High Risk": "#fb923c",
    "Critical Risk": "#fb7185",
}

CSS = """
<style>
    .stApp {
        background:
            radial-gradient(1200px 500px at 10% -10%, rgba(34, 211, 238, 0.12), transparent 50%),
            radial-gradient(900px 400px at 100% 0%, rgba(251, 113, 133, 0.10), transparent 45%),
            #0b1220;
    }
    .block-container { padding-top: 1.4rem; max-width: 1400px; }
    h1, h2, h3 { letter-spacing: -0.02em; }
    .hero-kicker {
        color: #67e8f9;
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.16em;
        text-transform: uppercase;
        margin-bottom: 0.25rem;
    }
    .kpi-grid { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 0.85rem; }
    @media (max-width: 1100px) { .kpi-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); } }
    .kpi-card {
        background: linear-gradient(180deg, rgba(30, 41, 59, 0.92), rgba(15, 23, 42, 0.92));
        border: 1px solid rgba(148, 163, 184, 0.16);
        border-radius: 16px;
        padding: 1rem 1.1rem 0.95rem;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.18);
    }
    .kpi-label {
        color: #94a3b8;
        font-size: 0.78rem;
        font-weight: 600;
        letter-spacing: 0.04em;
        text-transform: uppercase;
    }
    .kpi-value { color: #f8fafc; font-size: 1.7rem; font-weight: 700; line-height: 1.2; margin-top: 0.2rem; }
    .kpi-hint { color: #64748b; font-size: 0.78rem; margin-top: 0.15rem; }
    .callout {
        border-radius: 14px;
        padding: 0.9rem 1rem;
        border: 1px solid rgba(148, 163, 184, 0.18);
        background: rgba(15, 23, 42, 0.72);
        color: #cbd5e1;
        font-size: 0.92rem;
        line-height: 1.45;
        margin: 0.6rem 0 1rem;
    }
    .callout strong { color: #f8fafc; }
    .pred-panel {
        border-radius: 18px;
        padding: 1.15rem 1.2rem;
        border: 1px solid rgba(34, 211, 238, 0.28);
        background: linear-gradient(180deg, rgba(8, 47, 73, 0.55), rgba(15, 23, 42, 0.9));
    }
    .pred-label { color: #94a3b8; font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.05em; }
    .pred-value { color: #f8fafc; font-size: 2rem; font-weight: 700; margin: 0.15rem 0 0.8rem; }
    .flag-on {
        display: inline-block;
        background: rgba(251, 113, 133, 0.16);
        color: #fda4af;
        border: 1px solid rgba(251, 113, 133, 0.35);
        border-radius: 999px;
        padding: 0.35rem 0.85rem;
        font-weight: 700;
        letter-spacing: 0.04em;
    }
    .flag-off {
        display: inline-block;
        background: rgba(52, 211, 153, 0.12);
        color: #6ee7b7;
        border: 1px solid rgba(52, 211, 153, 0.3);
        border-radius: 999px;
        padding: 0.35rem 0.85rem;
        font-weight: 700;
        letter-spacing: 0.04em;
    }
    [data-testid="stSidebar"] {
        background: #020617;
        border-right: 1px solid rgba(148, 163, 184, 0.12);
    }
    div[data-testid="stMetricValue"] { font-size: 1.4rem; }
</style>
"""


def apply_theme(fig):
    fig.update_layout(**PLOTLY_LAYOUT)
    fig.update_xaxes(gridcolor="rgba(148,163,184,0.12)", zerolinecolor="rgba(148,163,184,0.2)")
    fig.update_yaxes(gridcolor="rgba(148,163,184,0.12)", zerolinecolor="rgba(148,163,184,0.2)")
    return fig
