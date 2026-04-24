"""
TherapyLens — Streamlit Dashboard
Run with: streamlit run dashboard/app.py
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json
import os

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="TherapyLens",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0f1117; }
    .block-container { padding-top: 1.5rem; }
    .metric-card {
        background: #1e2130;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        border-left: 4px solid #7c6ff7;
    }
    .flag-high   { color: #ff4b4b; font-weight: 600; }
    .flag-mod    { color: #ffa500; font-weight: 600; }
    .flag-low    { color: #21c55d; font-weight: 600; }
    .section-title {
        font-size: 1.1rem;
        font-weight: 700;
        color: #c8c8f0;
        margin-bottom: 0.3rem;
        letter-spacing: 0.04em;
        text-transform: uppercase;
    }
</style>
""", unsafe_allow_html=True)

# ── Data loading ───────────────────────────────────────────────────────────────
BASE = os.path.join(os.path.dirname(__file__), "..")


@st.cache_data
def load_data():
    df = pd.read_csv(os.path.join(BASE, "outputs", "results.csv"))
    with open(os.path.join(BASE, "outputs", "bias_audit.json")) as f:
        audit = json.load(f)
    with open(os.path.join(BASE, "outputs", "synthesis.json")) as f:
        synthesis = json.load(f)
    return df, audit, synthesis


df, audit, synthesis = load_data()

COHORT_COLORS = {
    "adult_general": "#7c6ff7",
    "teen":          "#38bdf8",
    "postpartum":    "#f472b6",
    "veteran":       "#34d399",
}

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/brain.png", width=60)
    st.title("TherapyLens")
    st.caption("NLP Pipeline for Therapy Transcript Analysis")
    st.divider()

    page = st.radio(
        "Navigate",
        ["📊 Overview", "😔 Sentiment & Emotion", "⚠️ Bias Audit",
         "📝 Session Summaries", "📈 Patient Timeline", "🗂️ Topic Distribution"],
        label_visibility="collapsed",
    )

    st.divider()
    cohorts = df["cohort"].unique().tolist()
    selected_cohorts = st.multiselect(
        "Filter cohorts", cohorts, default=cohorts)
    df_f = df[df["cohort"].isin(selected_cohorts)]

    st.caption(
        f"**{len(df_f)}** sessions · **{df_f['patient_id'].nunique()}** patients")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if page == "📊 Overview":
    st.title("📊 TherapyLens — Pipeline Overview")
    st.caption(
        "End-to-end NLP analysis of synthetic de-identified therapy transcripts")
    st.divider()

    # KPI row
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Sessions",   len(df_f))
    c2.metric("Patients",         df_f["patient_id"].nunique())
    c3.metric("Cohorts",          df_f["cohort"].nunique())
    c4.metric("High-Risk Sessions",
              int((df_f["outcome_risk_flag"] == "high").sum()))
    c5.metric("Avg Severity Score",
              f"{df_f['severity_score'].mean():.1f} / 27")

    st.divider()
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<p class="section-title">Sessions per Cohort</p>',
                    unsafe_allow_html=True)
        cohort_counts = df_f["cohort"].value_counts().reset_index()
        cohort_counts.columns = ["cohort", "count"]
        fig = px.bar(
            cohort_counts, x="cohort", y="count",
            color="cohort", color_discrete_map=COHORT_COLORS,
            text="count", template="plotly_dark",
        )
        fig.update_traces(textposition="outside")
        fig.update_layout(showlegend=False, xaxis_title="", yaxis_title="Sessions",
                          plot_bgcolor="#1e2130", paper_bgcolor="#1e2130")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown(
            '<p class="section-title">Outcome Risk Distribution</p>', unsafe_allow_html=True)
        risk_counts = df_f["outcome_risk_flag"].value_counts().reset_index()
        risk_counts.columns = ["risk", "count"]
        color_map = {"low": "#21c55d",
                     "moderate": "#ffa500", "high": "#ff4b4b"}
        fig2 = px.pie(
            risk_counts, names="risk", values="count",
            color="risk", color_discrete_map=color_map,
            hole=0.45, template="plotly_dark",
        )
        fig2.update_layout(plot_bgcolor="#1e2130", paper_bgcolor="#1e2130")
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown('<p class="section-title">Engagement Level by Cohort</p>',
                unsafe_allow_html=True)
    eng_data = df_f.groupby(["cohort", "engagement_level"]
                            ).size().reset_index(name="count")
    fig3 = px.bar(
        eng_data, x="cohort", y="count", color="engagement_level",
        barmode="group", template="plotly_dark",
        color_discrete_map={"low": "#ff4b4b",
                            "medium": "#ffa500", "high": "#21c55d"},
    )
    fig3.update_layout(plot_bgcolor="#1e2130", paper_bgcolor="#1e2130",
                       xaxis_title="", yaxis_title="Sessions")
    st.plotly_chart(fig3, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: SENTIMENT & EMOTION
# ══════════════════════════════════════════════════════════════════════════════
elif page == "😔 Sentiment & Emotion":
    st.title("😔 Sentiment & Emotion Analysis")
    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            '<p class="section-title">Mean Sentiment Score by Cohort</p>', unsafe_allow_html=True)
        sent = df_f.groupby("cohort")["sentiment_score"].mean().reset_index()
        sent.columns = ["cohort", "mean_sentiment"]
        fig = px.bar(
            sent, x="cohort", y="mean_sentiment",
            color="cohort", color_discrete_map=COHORT_COLORS,
            template="plotly_dark", text=sent["mean_sentiment"].round(3),
        )
        fig.add_hline(y=0, line_dash="dash", line_color="gray")
        fig.update_layout(showlegend=False, plot_bgcolor="#1e2130",
                          paper_bgcolor="#1e2130", yaxis_title="Mean Sentiment (-1 to +1)")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown(
            '<p class="section-title">Sentiment Score Distribution</p>', unsafe_allow_html=True)
        fig2 = px.box(
            df_f, x="cohort", y="sentiment_score",
            color="cohort", color_discrete_map=COHORT_COLORS,
            template="plotly_dark",
        )
        fig2.update_layout(showlegend=False, plot_bgcolor="#1e2130",
                           paper_bgcolor="#1e2130", xaxis_title="", yaxis_title="Sentiment Score")
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown('<p class="section-title">Dominant Emotion Breakdown by Cohort</p>',
                unsafe_allow_html=True)
    emotion_data = df_f.groupby(
        ["cohort", "dominant_emotion"]).size().reset_index(name="count")
    fig3 = px.bar(
        emotion_data, x="cohort", y="count", color="dominant_emotion",
        barmode="stack", template="plotly_dark",
        color_discrete_sequence=px.colors.qualitative.Pastel,
    )
    fig3.update_layout(plot_bgcolor="#1e2130", paper_bgcolor="#1e2130",
                       xaxis_title="", yaxis_title="Sessions")
    st.plotly_chart(fig3, use_container_width=True)

    st.divider()
    st.markdown('<p class="section-title">Emotion Radar — Cohort Averages</p>',
                unsafe_allow_html=True)
    emotions = ["anxiety", "sadness", "depression", "progress", "neutral"]
    fig4 = go.Figure()
    for cohort, info in audit["cohort_summary"].items():
        if cohort not in selected_cohorts:
            continue
        values = [info["mean_emotions"].get(e, 0) for e in emotions]
        fig4.add_trace(go.Scatterpolar(
            r=values + [values[0]],
            theta=emotions + [emotions[0]],
            fill="toself", name=cohort,
            line_color=COHORT_COLORS.get(cohort, "#aaa"),
        ))
    fig4.update_layout(
        polar=dict(bgcolor="#1e2130"),
        paper_bgcolor="#1e2130", template="plotly_dark",
    )
    st.plotly_chart(fig4, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: BIAS AUDIT
# ══════════════════════════════════════════════════════════════════════════════
elif page == "⚠️ Bias Audit":
    st.title("⚠️ Bias Audit — Cohort Disparity Report")
    st.divider()

    # Disparity flags
    flags = audit["disparity_flags"]
    if flags:
        st.error(
            f"**{len(flags)} disparity flag(s) detected** — manual clinical review recommended")
        for flag in flags:
            st.warning(f"🚩 {flag}")
    else:
        st.success("✅ No significant cohort disparities detected.")

    st.divider()
    st.markdown('<p class="section-title">Cohort Summary Table</p>',
                unsafe_allow_html=True)

    rows = []
    for cohort, info in audit["cohort_summary"].items():
        rows.append({
            "Cohort": cohort,
            "Sessions": info["n_sessions"],
            "Mean Sentiment": round(info["mean_sentiment"], 3),
            "Dominant Emotion": info["dominant_emotion"],
            "Anxiety": round(info["mean_emotions"]["anxiety"], 3),
            "Sadness": round(info["mean_emotions"]["sadness"], 3),
            "Depression": round(info["mean_emotions"]["depression"], 3),
            "Progress": round(info["mean_emotions"]["progress"], 3),
        })
    audit_df = pd.DataFrame(rows)
    st.dataframe(audit_df, use_container_width=True, hide_index=True)

    st.divider()
    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            '<p class="section-title">Mean Sentiment per Cohort</p>', unsafe_allow_html=True)
        fig = px.bar(
            audit_df, x="Cohort", y="Mean Sentiment",
            color="Cohort", color_discrete_map=COHORT_COLORS,
            template="plotly_dark", text="Mean Sentiment",
        )
        fig.add_hline(y=0, line_dash="dash", line_color="gray")
        fig.update_layout(showlegend=False,
                          plot_bgcolor="#1e2130", paper_bgcolor="#1e2130")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown(
            '<p class="section-title">Emotion Profile Heatmap</p>', unsafe_allow_html=True)
        heat_df = audit_df.set_index(
            "Cohort")[["Anxiety", "Sadness", "Depression", "Progress"]]
        fig2 = px.imshow(
            heat_df, text_auto=".3f", aspect="auto",
            color_continuous_scale="RdYlGn_r", template="plotly_dark",
        )
        fig2.update_layout(plot_bgcolor="#1e2130", paper_bgcolor="#1e2130")
        st.plotly_chart(fig2, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: SESSION SUMMARIES
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📝 Session Summaries":
    st.title("📝 LLM Session Summaries")
    st.caption("Generated by Groq LLaMA-3.3-70b based on transcript NLP signals")
    st.divider()

    syn_df = pd.DataFrame(synthesis)
    syn_df = syn_df[syn_df["cohort"].isin(selected_cohorts)]

    # Filters row
    c1, c2, c3 = st.columns(3)
    risk_filter = c1.selectbox(
        "Outcome Risk", ["All", "high", "moderate", "low"])
    eng_filter = c2.selectbox(
        "Engagement Level", ["All", "high", "medium", "low"])
    patient_filter = c3.selectbox(
        "Patient", ["All"] + sorted(syn_df["patient_id"].unique().tolist()))

    filtered = syn_df.copy()
    if risk_filter != "All":
        filtered = filtered[filtered["outcome_risk_flag"] == risk_filter]
    if eng_filter != "All":
        filtered = filtered[filtered["engagement_level"] == eng_filter]
    if patient_filter != "All":
        filtered = filtered[filtered["patient_id"] == patient_filter]

    st.caption(f"Showing {len(filtered)} sessions")
    st.divider()

    for _, row in filtered.iterrows():
        risk_color = {"high": "🔴", "moderate": "🟡", "low": "🟢"}.get(
            row["outcome_risk_flag"], "⚪")
        with st.expander(
            f"{risk_color} **{row['patient_id']}** · Session {row['session_id']} · {row['cohort']} · Risk: {row['outcome_risk_flag']}"
        ):
            st.markdown(f"**Summary:** {row['session_summary']}")
            c1, c2, c3 = st.columns(3)
            c1.metric("Engagement", row["engagement_level"].title())
            c2.metric("Outcome Risk", row["outcome_risk_flag"].title())
            c3.metric("Recommended Focus", row["recommended_focus"])
            if row.get("notable_linguistic_patterns"):
                st.markdown("**Linguistic Patterns:**")
                for p in row["notable_linguistic_patterns"]:
                    st.markdown(f"- {p}")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: PATIENT TIMELINE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📈 Patient Timeline":
    st.title("📈 Patient Session Timeline")
    st.divider()

    patients = sorted(df_f["patient_id"].unique().tolist())
    selected_patient = st.selectbox("Select Patient", patients)

    pat_df = df[df["patient_id"] ==
                selected_patient].sort_values("session_number")

    if pat_df.empty:
        st.warning("No data for this patient.")
    else:
        cohort = pat_df["cohort"].iloc[0]
        st.markdown(f"**Cohort:** `{cohort}` · **{len(pat_df)} sessions**")
        st.divider()

        col1, col2 = st.columns(2)

        with col1:
            st.markdown(
                '<p class="section-title">Sentiment Over Sessions</p>', unsafe_allow_html=True)
            fig = px.line(
                pat_df, x="session_number", y="sentiment_score",
                markers=True, template="plotly_dark",
                color_discrete_sequence=[COHORT_COLORS.get(cohort, "#7c6ff7")],
            )
            fig.add_hline(y=0, line_dash="dash", line_color="gray")
            fig.update_layout(plot_bgcolor="#1e2130", paper_bgcolor="#1e2130",
                              xaxis_title="Session #", yaxis_title="Sentiment Score")
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown(
                '<p class="section-title">Severity Score Over Sessions</p>', unsafe_allow_html=True)
            fig2 = px.line(
                pat_df, x="session_number", y="severity_score",
                markers=True, template="plotly_dark",
                color_discrete_sequence=["#f472b6"],
            )
            fig2.update_layout(plot_bgcolor="#1e2130", paper_bgcolor="#1e2130",
                               xaxis_title="Session #", yaxis_title="PHQ-9 Severity Score")
            st.plotly_chart(fig2, use_container_width=True)

        st.markdown(
            '<p class="section-title">Engagement & Lexical Diversity Over Sessions</p>', unsafe_allow_html=True)
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(
            x=pat_df["session_number"], y=pat_df["avg_patient_words_per_turn"],
            mode="lines+markers", name="Avg Words/Turn",
            line=dict(color="#38bdf8"),
        ))
        fig3.add_trace(go.Scatter(
            x=pat_df["session_number"], y=pat_df["patient_lexical_diversity"],
            mode="lines+markers", name="Lexical Diversity (TTR)",
            line=dict(color="#34d399"), yaxis="y2",
        ))
        fig3.update_layout(
            template="plotly_dark", plot_bgcolor="#1e2130", paper_bgcolor="#1e2130",
            xaxis_title="Session #",
            yaxis=dict(title="Avg Words/Turn"),
            yaxis2=dict(title="TTR", overlaying="y", side="right"),
            legend=dict(x=0.01, y=0.99),
        )
        st.plotly_chart(fig3, use_container_width=True)

        st.markdown('<p class="section-title">Session Data Table</p>',
                    unsafe_allow_html=True)
        st.dataframe(
            pat_df[["session_number", "severity_score", "sentiment_score",
                    "dominant_emotion", "primary_topic", "engagement_level",
                    "outcome_risk_flag", "recommended_focus"]].reset_index(drop=True),
            use_container_width=True, hide_index=True,
        )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: TOPIC DISTRIBUTION
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🗂️ Topic Distribution":
    st.title("🗂️ Topic Distribution")
    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            '<p class="section-title">Overall Topic Frequency</p>', unsafe_allow_html=True)
        topic_counts = df_f["primary_topic"].value_counts().reset_index()
        topic_counts.columns = ["topic", "count"]
        fig = px.bar(
            topic_counts, x="count", y="topic", orientation="h",
            color="count", color_continuous_scale="Purples",
            template="plotly_dark", text="count",
        )
        fig.update_layout(showlegend=False, plot_bgcolor="#1e2130",
                          paper_bgcolor="#1e2130", yaxis_title="",
                          xaxis_title="Sessions", yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown(
            '<p class="section-title">Topic Share by Cohort</p>', unsafe_allow_html=True)
        topic_cohort = df_f.groupby(
            ["cohort", "primary_topic"]).size().reset_index(name="count")
        fig2 = px.bar(
            topic_cohort, x="cohort", y="count", color="primary_topic",
            barmode="stack", template="plotly_dark",
            color_discrete_sequence=px.colors.qualitative.Pastel,
        )
        fig2.update_layout(plot_bgcolor="#1e2130", paper_bgcolor="#1e2130",
                           xaxis_title="", yaxis_title="Sessions")
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown('<p class="section-title">Topic × Risk Heatmap</p>',
                unsafe_allow_html=True)
    heat = df_f.groupby(["primary_topic", "outcome_risk_flag"]
                        ).size().unstack(fill_value=0)
    fig3 = px.imshow(
        heat, text_auto=True, aspect="auto",
        color_continuous_scale="YlOrRd", template="plotly_dark",
    )
    fig3.update_layout(plot_bgcolor="#1e2130", paper_bgcolor="#1e2130",
                       xaxis_title="Outcome Risk", yaxis_title="Topic")
    st.plotly_chart(fig3, use_container_width=True)

    st.markdown('<p class="section-title">Topic × Dominant Emotion</p>',
                unsafe_allow_html=True)
    emotion_topic = df_f.groupby(
        ["primary_topic", "dominant_emotion"]).size().reset_index(name="count")
    fig4 = px.bar(
        emotion_topic, x="primary_topic", y="count", color="dominant_emotion",
        barmode="stack", template="plotly_dark",
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    fig4.update_layout(plot_bgcolor="#1e2130", paper_bgcolor="#1e2130",
                       xaxis_title="", yaxis_title="Sessions",
                       xaxis_tickangle=-30)
    st.plotly_chart(fig4, use_container_width=True)
