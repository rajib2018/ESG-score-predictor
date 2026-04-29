"""
app.py
=======

Streamlit front-end for the ESG Risk Predictor.

The UI is intentionally simple:

  * The user enters an **organisation name** and selects a **country**
    (and an industry, used for benchmarking).
  * The app returns Environmental, Social, Governance and overall ESG
    risk scores, colour-coded as Low / Medium / High.
  * Each prediction is shown alongside its **industry peer benchmark**
    so the user can see whether the company is better or worse than
    typical peers.
  * Every prediction is **explained** using SHAP - both as a top-feature
    waterfall plot and as a plain-English narrative.

The heavy lifting (data, model, SHAP) lives in ``data_sources.py`` and
``esg_model.py`` to keep this file focused on UI/UX.
"""

from __future__ import annotations

import warnings

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from data_sources import (
    INDUSTRY_BENCHMARKS,
    classify_risk,
    list_countries,
    list_industries,
    risk_color,
)
from esg_model import ESGRiskModel, ESGPrediction, PillarPrediction

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Page configuration + small CSS polish
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="ESG Risk Predictor",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .main { padding: 0rem 1rem; }
    h1 { color: #1f4e79; font-weight: 700; }
    .risk-pill {
        display: inline-block;
        padding: 6px 14px;
        border-radius: 999px;
        color: white;
        font-weight: 600;
        font-size: 14px;
    }
    .benchmark-card {
        background-color: #f8f9fb;
        border-left: 4px solid #1f4e79;
        padding: 12px 16px;
        border-radius: 6px;
        margin-bottom: 8px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Cached model loader
# ---------------------------------------------------------------------------
# The model is fully synthetic and reproducible - we train it once per
# Streamlit session and keep it warm in memory.
@st.cache_resource(show_spinner="Training ESG risk model...")
def load_model() -> ESGRiskModel:
    return ESGRiskModel(n_samples=1500, random_state=42)


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

def gauge_chart(score: float, title: str, benchmark: float | None = None) -> go.Figure:
    """
    Build a 0-100 gauge with three coloured zones (Low / Medium / High).
    The benchmark is shown as a threshold line so the user immediately
    sees whether the prediction is above or below industry average.
    """
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=score,
            number={"font": {"size": 36}},
            domain={"x": [0, 1], "y": [0, 1]},
            title={"text": title, "font": {"size": 18}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "darkblue"},
                "bar": {"color": "#1f4e79"},
                "bgcolor": "white",
                "borderwidth": 2,
                "bordercolor": "lightgray",
                # Three colour bands matching the Low/Medium/High thresholds.
                "steps": [
                    {"range": [0, 30], "color": "#d4edda"},
                    {"range": [30, 60], "color": "#fff3cd"},
                    {"range": [60, 100], "color": "#f8d7da"},
                ],
                # Optional benchmark marker (red line).
                "threshold": (
                    {
                        "line": {"color": "red", "width": 3},
                        "thickness": 0.85,
                        "value": benchmark,
                    }
                    if benchmark is not None
                    else {}
                ),
            },
        )
    )
    fig.update_layout(height=260, margin=dict(l=20, r=20, t=50, b=20))
    return fig


def shap_waterfall(pillar: PillarPrediction, top_n: int = 8) -> go.Figure:
    """
    Horizontal bar chart of the top SHAP contributors for a pillar.
    Positive bars (red) push risk up; negative bars (green) push it
    down.  Sorted by absolute impact so the strongest drivers appear
    first.
    """
    items = sorted(
        pillar.shap_values.items(), key=lambda kv: abs(kv[1]), reverse=True
    )[:top_n]
    features = [k.replace("_", " ").title() for k, _ in items]
    values = [v for _, v in items]
    colors = ["#e74c3c" if v > 0 else "#2ecc71" for v in values]

    fig = go.Figure(
        go.Bar(
            x=values,
            y=features,
            orientation="h",
            marker=dict(color=colors),
            text=[f"{v:+.2f}" for v in values],
            textposition="outside",
        )
    )
    fig.update_layout(
        title="Top feature impacts on this prediction",
        xaxis_title="SHAP value (effect on risk score)",
        yaxis=dict(autorange="reversed"),
        height=380,
        margin=dict(l=20, r=20, t=50, b=20),
        showlegend=False,
    )
    return fig


def benchmark_bar(prediction: ESGPrediction, industry: str) -> go.Figure:
    """
    Side-by-side comparison of the predicted scores vs. the industry
    benchmark.  Lets the user see at a glance which pillar is the
    problem (or the strength).
    """
    bench = INDUSTRY_BENCHMARKS[industry]
    pillars = ["Environmental", "Social", "Governance", "Overall ESG"]
    predicted = [
        prediction.environmental.score,
        prediction.social.score,
        prediction.governance.score,
        prediction.overall_score,
    ]
    benchmarks = [
        bench["e_risk"],
        bench["s_risk"],
        bench["g_risk"],
        bench["esg_risk"],
    ]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(name=f"{industry} benchmark", x=pillars, y=benchmarks, marker_color="#95a5a6")
    )
    fig.add_trace(
        go.Bar(name="This organisation", x=pillars, y=predicted, marker_color="#1f4e79")
    )
    fig.update_layout(
        barmode="group",
        title="Predicted risk vs. industry benchmark",
        yaxis_title="Risk score (higher = more risk)",
        yaxis_range=[0, 100],
        height=360,
        margin=dict(l=20, r=20, t=50, b=20),
    )
    return fig


# ---------------------------------------------------------------------------
# Narrative generator (plain-English explanation of the SHAP output)
# ---------------------------------------------------------------------------

def explain_pillar_narrative(pillar_name: str, pillar: PillarPrediction) -> str:
    """
    Turn raw SHAP numbers into a short paragraph any reader can follow.

    We pick the three biggest absolute drivers and describe them in
    natural language, distinguishing between factors that raised risk
    (positive SHAP) and factors that lowered it (negative SHAP).
    """
    sorted_items = sorted(
        pillar.shap_values.items(), key=lambda kv: abs(kv[1]), reverse=True
    )[:3]

    raised = [f for f, v in sorted_items if v > 0]
    lowered = [f for f, v in sorted_items if v < 0]

    parts = [
        f"**{pillar_name}** risk was predicted at **{pillar.score:.1f}/100** "
        f"(**{pillar.label}** risk)."
    ]
    if raised:
        parts.append(
            "The factors that pushed risk **up** the most were "
            + ", ".join(f"`{f.replace('_', ' ')}`" for f in raised)
            + "."
        )
    if lowered:
        parts.append(
            "The factors that pulled risk **down** were "
            + ", ".join(f"`{f.replace('_', ' ')}`" for f in lowered)
            + "."
        )
    return " ".join(parts)


# ---------------------------------------------------------------------------
# UI sections
# ---------------------------------------------------------------------------

def render_sidebar(model: ESGRiskModel) -> None:
    """Static info panel describing the model and its honest limits."""
    with st.sidebar:
        st.title("🌍 ESG Risk Predictor")
        st.markdown(
            "Predicts **Environmental, Social and Governance** risk for an "
            "organisation based on its country, industry and inferred "
            "company profile."
        )
        st.markdown("---")
        st.subheader("Model")
        st.write("**Algorithm:** Random Forest (one per pillar)")
        st.write("**Explainability:** SHAP TreeExplainer")
        st.write("**Training data:** 1,500 synthetic firms")

        st.markdown("**Hold-out performance**")
        metrics_df = pd.DataFrame(model.metrics).T
        metrics_df.index.name = "Pillar"
        st.dataframe(
            metrics_df.style.format({"r2": "{:.2f}", "mae": "{:.1f}"}),
            use_container_width=True,
        )

        st.markdown("---")
        st.caption(
            "Higher score = higher risk.  Bands: 0-30 Low · 30-60 Medium · "
            "60-100 High.  This is a demo using synthetic data."
        )


def render_inputs() -> tuple[str, str, str, bool]:
    """Collect the three inputs that drive the prediction."""
    st.header("Predict ESG risk for an organisation")
    st.markdown(
        "Enter the organisation's name and the country where it is "
        "headquartered. Pick the industry so we can compare against "
        "relevant peers."
    )

    col1, col2, col3 = st.columns([2, 1.4, 1.4])
    with col1:
        org_name = st.text_input(
            "Organisation name",
            value="Acme Corporation",
            help=(
                "Free-text name. The same name always produces the same "
                "prediction (deterministic feature generation)."
            ),
        )
    with col2:
        country = st.selectbox(
            "Country (HQ)",
            options=list_countries(),
            index=list_countries().index("United States"),
            help="Country-level indicators (EPI, rule of law, etc.) feed the model.",
        )
    with col3:
        industry = st.selectbox(
            "Industry",
            options=list_industries(),
            index=list_industries().index("Technology"),
            help="Used as a model feature AND as the peer benchmark.",
        )

    submit = st.button("🎯 Predict ESG risk", type="primary", use_container_width=False)
    return org_name, country, industry, submit


def render_summary(prediction: ESGPrediction, industry: str) -> None:
    """Top-level scorecard: four big numbers + risk pills."""
    bench = INDUSTRY_BENCHMARKS[industry]
    cols = st.columns(4)

    cards = [
        ("Environmental", prediction.environmental.score, prediction.environmental.label, bench["e_risk"]),
        ("Social",        prediction.social.score,        prediction.social.label,        bench["s_risk"]),
        ("Governance",    prediction.governance.score,    prediction.governance.label,    bench["g_risk"]),
        ("Overall ESG",   prediction.overall_score,       prediction.overall_label,       bench["esg_risk"]),
    ]

    for col, (name, score, label, peer) in zip(cols, cards):
        with col:
            color = risk_color(label)
            delta = score - peer
            # Delta direction matters: lower than peer = better (negative delta good).
            delta_text = f"{delta:+.1f} vs. industry avg"
            st.metric(name, f"{score:.1f}/100", delta=delta_text, delta_color="inverse")
            st.markdown(
                f"<span class='risk-pill' style='background-color:{color}'>"
                f"{label} risk</span>",
                unsafe_allow_html=True,
            )
            st.caption(f"Industry avg: {peer:.0f}")


def render_gauges(prediction: ESGPrediction, industry: str) -> None:
    """Four gauge charts with industry benchmark threshold lines."""
    bench = INDUSTRY_BENCHMARKS[industry]
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.plotly_chart(
            gauge_chart(prediction.environmental.score, "Environmental", bench["e_risk"]),
            use_container_width=True,
        )
    with c2:
        st.plotly_chart(
            gauge_chart(prediction.social.score, "Social", bench["s_risk"]),
            use_container_width=True,
        )
    with c3:
        st.plotly_chart(
            gauge_chart(prediction.governance.score, "Governance", bench["g_risk"]),
            use_container_width=True,
        )
    with c4:
        st.plotly_chart(
            gauge_chart(prediction.overall_score, "Overall ESG", bench["esg_risk"]),
            use_container_width=True,
        )
    st.caption(
        "🔴 The red line on each gauge marks the **industry peer benchmark**. "
        "Bars left of the line indicate *better-than-peer* performance."
    )


def render_explanations(prediction: ESGPrediction) -> None:
    """SHAP-driven plots + plain-English narratives, one tab per pillar."""
    st.subheader("🔍 Why did the model predict these scores?")
    st.markdown(
        "Each prediction is decomposed using **SHAP values**.  A positive "
        "SHAP value means the feature *pushed risk up* for this "
        "organisation; a negative value means the feature *pulled risk "
        "down*.  Bars are sorted by absolute impact."
    )

    tab_e, tab_s, tab_g = st.tabs(["🌱 Environmental", "👥 Social", "⚖️ Governance"])

    for tab, name, pillar in [
        (tab_e, "Environmental", prediction.environmental),
        (tab_s, "Social",        prediction.social),
        (tab_g, "Governance",    prediction.governance),
    ]:
        with tab:
            left, right = st.columns([1.3, 1])
            with left:
                st.plotly_chart(shap_waterfall(pillar), use_container_width=True)
            with right:
                st.markdown(explain_pillar_narrative(name, pillar))
                # Show the actual feature values used so the explanation
                # is fully traceable.
                feat_df = (
                    prediction.feature_row.T.rename(columns={0: "value"})
                    .reset_index()
                    .rename(columns={"index": "feature"})
                )
                feat_df["feature"] = feat_df["feature"].str.replace("_", " ").str.title()
                st.markdown("**Feature values used**")
                st.dataframe(
                    feat_df.style.format({"value": "{:.1f}"}),
                    hide_index=True,
                    use_container_width=True,
                )


def render_benchmark_section(prediction: ESGPrediction, industry: str) -> None:
    """Side-by-side benchmark plot + Low/Medium/High call-outs."""
    st.subheader(f"📊 Industry benchmark - {industry}")
    st.plotly_chart(benchmark_bar(prediction, industry), use_container_width=True)

    bench = INDUSTRY_BENCHMARKS[industry]
    pillar_specs = [
        ("Environmental", prediction.environmental.score, bench["e_risk"]),
        ("Social",        prediction.social.score,        bench["s_risk"]),
        ("Governance",    prediction.governance.score,    bench["g_risk"]),
        ("Overall ESG",   prediction.overall_score,       bench["esg_risk"]),
    ]

    cols = st.columns(len(pillar_specs))
    for col, (name, score, peer) in zip(cols, pillar_specs):
        with col:
            label = classify_risk(score)
            color = risk_color(label)
            relative = "below" if score < peer else "above"
            st.markdown(
                f"""
                <div class='benchmark-card'>
                    <div style='font-weight:600; color:#1f4e79;'>{name}</div>
                    <div style='font-size:24px; font-weight:700;'>{score:.1f}</div>
                    <div>Industry avg: {peer:.0f}</div>
                    <div>{abs(score - peer):.1f} pts <b>{relative}</b> peers</div>
                    <div style='margin-top:8px;'>
                      <span class='risk-pill' style='background-color:{color}'>
                        {label} risk
                      </span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_methodology(model: ESGRiskModel) -> None:
    """Static methodology / glossary tab."""
    st.subheader("How this works")
    st.markdown(
        """
        **Inputs**
        * **Organisation name** - hashed deterministically into seven
          synthetic company traits (size, age, carbon intensity,
          disclosure quality, board independence, workforce diversity,
          safety record).
        * **Country** - contributes four macro indicators
          (Environmental Performance Index, Rule of Law, Social
          Progress Index, Climate Risk).
        * **Industry** - contributes three peer-average risk levels
          (E, S, G), and serves as the benchmark we compare against.

        **Model**
        Three Random Forest regressors, one per pillar, trained on a
        synthetic but transparent ground-truth function.

        **Explainability**
        SHAP TreeExplainer attributes each prediction to its features.
        The top contributors are surfaced as a waterfall chart and as
        a plain-English narrative.

        **Risk bands**
        | Score range | Label  |
        |-------------|--------|
        | 0 - 30      | 🟢 Low |
        | 30 - 60     | 🟡 Medium |
        | 60 - 100    | 🔴 High |

        **Caveats**
        * Synthetic training data; do **not** use this for investment
          decisions or compliance reporting.
        * Country and industry indicators are illustrative defaults
          inspired by public indices.
        """
    )

    st.markdown("**Global feature importance (mean |SHAP| over training set)**")
    p_tabs = st.tabs(["Environmental", "Social", "Governance"])
    for tab, pillar in zip(p_tabs, ["E", "S", "G"]):
        with tab:
            imp = model.feature_importance(pillar)
            fig = go.Figure(
                go.Bar(
                    x=imp["importance"],
                    y=imp["feature"].str.replace("_", " ").str.title(),
                    orientation="h",
                    marker_color="#1f4e79",
                )
            )
            fig.update_layout(
                height=400,
                yaxis=dict(autorange="reversed"),
                margin=dict(l=20, r=20, t=20, b=20),
                xaxis_title="Mean |SHAP|",
            )
            st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    model = load_model()
    render_sidebar(model)

    st.title("🌍 ESG Risk Predictor")
    st.markdown(
        "Estimate **Environmental, Social, Governance** and overall **ESG "
        "risk** for any organisation, with industry-aware benchmarks and "
        "transparent SHAP explanations."
    )

    org_name, country, industry, submit = render_inputs()

    # Two top-level tabs: the prediction view and the methodology view.
    tab_predict, tab_method = st.tabs(["📈 Prediction", "📚 Methodology"])

    with tab_predict:
        if submit:
            if not org_name or not org_name.strip():
                st.warning("Please enter an organisation name.")
                return

            with st.spinner("Scoring..."):
                prediction = model.predict(org_name.strip(), country, industry)

            st.success(
                f"**{org_name.strip()}** ({country}, {industry}) - overall ESG "
                f"risk **{prediction.overall_score:.1f}/100** "
                f"(**{prediction.overall_label}** risk)."
            )
            render_summary(prediction, industry)
            st.markdown("---")
            render_gauges(prediction, industry)
            st.markdown("---")
            render_benchmark_section(prediction, industry)
            st.markdown("---")
            render_explanations(prediction)
        else:
            st.info(
                "Enter an organisation name above and click **Predict ESG "
                "risk** to see scores, peer benchmarks and per-feature "
                "explanations."
            )

    with tab_method:
        render_methodology(model)


if __name__ == "__main__":
    main()
