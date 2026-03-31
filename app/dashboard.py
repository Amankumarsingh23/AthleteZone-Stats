import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from app.startup import run_pipeline

run_pipeline()

st.set_page_config(
    page_title="AthleteZone Stats",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

SIGNALS = ["heart_rate","hrv","reaction_time_ms","decision_accuracy",
           "pupil_diameter","blink_rate","saccade_velocity",
           "cortisol_proxy","focus_score","movement_efficiency"]
COLOR_MAP = {"fatigued":"#E24B4A","normal":"#EF9F27","zone":"#1D9E75"}

@st.cache_data
def load_data():
    df_raw = pd.read_csv("data/raw/athlete_sessions.csv")
    agg = {s: "mean" for s in SIGNALS}
    agg["state"] = "first"
    agg["state_label"] = "first"
    agg["sport"] = "first"
    agg["athlete_id"] = "first"
    df = df_raw.groupby("session_id").agg(agg).reset_index()
    return df, df_raw

# ── Sidebar ────────────────────────────────────────────────────────────────
st.sidebar.title("AthleteZone Stats ⚡")
st.sidebar.markdown("*Finding the ZONE in biometric data*")
st.sidebar.markdown("---")
page = st.sidebar.radio("Navigate", [
    "ZONE Explorer", "Athlete Profiler",
    "Statistical Analysis", "Cluster Visualiser",
    "Live ZONE Predictor"])
st.sidebar.markdown("---")
st.sidebar.markdown("Built for **NeuralPort** data science research")
st.sidebar.markdown("[GitHub](https://github.com/Amankumarsingh23/AthleteZone-Stats)")

df, df_raw = load_data()

# ── ZONE Explorer ──────────────────────────────────────────────────────────
if page == "ZONE Explorer":
    st.title("ZONE Explorer")
    st.markdown("Visualise how biometric signals differ across fatigue, normal, and ZONE states.")

    col1, col2, col3 = st.columns(3)
    for state, col, icon in zip(
        ["zone","normal","fatigued"],
        [col1, col2, col3],
        ["⚡","✓","⚠"]):
        sub = df[df["state_label"]==state]
        col.metric(f"{icon} {state.title()} sessions", len(sub))

    signal = st.selectbox("Select signal to explore",
                          [s.replace("_"," ").title() for s in SIGNALS])
    signal_col = signal.lower().replace(" ","_")

    fig = go.Figure()
    for state, color in COLOR_MAP.items():
        subset = df[df["state_label"]==state][signal_col]
        fig.add_trace(go.Violin(
            y=subset, name=state.title(),
            fillcolor=color, line_color=color,
            opacity=0.7, box_visible=True, meanline_visible=True))
    fig.update_layout(
        title=f"{signal} distribution by athlete state",
        yaxis_title=signal, height=420)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Signal means by state")
    means = df.groupby("state_label")[SIGNALS].mean().round(3)
    st.dataframe(means.style.background_gradient(cmap="RdYlGn", axis=0),
                 use_container_width=True)

# ── Athlete Profiler ───────────────────────────────────────────────────────
elif page == "Athlete Profiler":
    st.title("Athlete Profiler")
    athletes = sorted(df["athlete_id"].unique())
    selected = st.selectbox("Select athlete", athletes)

    athlete_df = df[df["athlete_id"]==selected]
    state_counts = athlete_df["state_label"].value_counts()

    col1, col2, col3 = st.columns(3)
    col1.metric("ZONE sessions",     state_counts.get("zone", 0))
    col2.metric("Normal sessions",   state_counts.get("normal", 0))
    col3.metric("Fatigued sessions", state_counts.get("fatigued", 0))

    fig = px.bar(
        state_counts.reset_index(),
        x="state_label", y="count",
        color="state_label",
        color_discrete_map=COLOR_MAP,
        title=f"Athlete {selected} — session state distribution")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Biometric radar — state comparison")
    norm_signals = ["reaction_time_ms","focus_score","hrv",
                    "pupil_diameter","movement_efficiency","decision_accuracy"]
    fig_radar = go.Figure()
    for state, color in COLOR_MAP.items():
        sub = athlete_df[athlete_df["state_label"]==state]
        if len(sub) == 0: continue
        vals = []
        for s in norm_signals:
            col_min = df[s].min()
            col_max = df[s].max()
            v = (sub[s].mean() - col_min) / (col_max - col_min + 1e-9)
            if s == "reaction_time_ms":
                v = 1 - v
            vals.append(round(v, 3))
        fig_radar.add_trace(go.Scatterpolar(
            r=vals + [vals[0]],
            theta=[s.replace("_"," ").title() for s in norm_signals] +
                  [norm_signals[0].replace("_"," ").title()],
            name=state.title(), fill="toself",
            line_color=color, opacity=0.6))
    fig_radar.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0,1])),
        title=f"Athlete {selected} — normalised biometric profile",
        height=450)
    st.plotly_chart(fig_radar, use_container_width=True)

# ── Statistical Analysis ───────────────────────────────────────────────────
elif page == "Statistical Analysis":
    st.title("Statistical Analysis")

    from scipy.stats import f_oneway, kruskal

    st.subheader("One-way ANOVA — which signals differ across states?")
    anova_rows = []
    for signal in SIGNALS:
        groups = [df[df["state_label"]==s][signal].values
                  for s in ["fatigued","normal","zone"]]
        f_stat, p_val = f_oneway(*groups)
        kw_stat, p_kw = kruskal(*groups)
        anova_rows.append({
            "Signal": signal.replace("_"," ").title(),
            "F-statistic": round(f_stat, 2),
            "ANOVA p-value": round(p_val, 8),
            "KW statistic": round(kw_stat, 2),
            "KW p-value": round(p_kw, 8),
            "Significant": "Yes ***" if p_val < 0.001 else "Yes *" if p_val < 0.05 else "No"
        })
    anova_df = pd.DataFrame(anova_rows).sort_values("F-statistic", ascending=False)
    st.dataframe(anova_df, use_container_width=True)

    st.subheader("Correlation heatmap")
    corr = df[SIGNALS].corr()
    fig = go.Figure(go.Heatmap(
        z=corr.values, x=[s.replace("_"," ") for s in SIGNALS],
        y=[s.replace("_"," ") for s in SIGNALS],
        colorscale="RdBu", zmid=0,
        text=np.round(corr.values, 2), texttemplate="%{text}",
        showscale=True))
    fig.update_layout(title="Feature correlation matrix", height=550)
    st.plotly_chart(fig, use_container_width=True)

# ── Cluster Visualiser ────────────────────────────────────────────────────
elif page == "Cluster Visualiser":
    st.title("Cluster Visualiser")
    st.markdown("K-means unsupervised ZONE detection — no labels used.")

    k = st.slider("Number of clusters (K)", 2, 8, 3)
    X_mat = df[SIGNALS].values
    scaler = StandardScaler()
    X_sc   = scaler.fit_transform(X_mat)

    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_sc)
    from sklearn.metrics import silhouette_score
    sil = silhouette_score(X_sc, km.labels_)

    pca = PCA(n_components=2, random_state=42)
    X_2d = pca.fit_transform(X_sc)

    col1, col2 = st.columns(2)
    col1.metric("Silhouette Score", f"{sil:.3f}")
    col2.metric("Variance explained (PC1+PC2)",
                f"{sum(pca.explained_variance_ratio_)*100:.1f}%")

    plot_df = pd.DataFrame({
        "PC1": X_2d[:,0], "PC2": X_2d[:,1],
        "Cluster": km.labels_.astype(str),
        "True State": df["state_label"].values
    })

    col_a, col_b = st.columns(2)
    with col_a:
        fig1 = px.scatter(plot_df, x="PC1", y="PC2", color="Cluster",
                          title="K-means clusters (unsupervised)",
                          opacity=0.65)
        fig1.update_traces(marker_size=4)
        st.plotly_chart(fig1, use_container_width=True)
    with col_b:
        fig2 = px.scatter(plot_df, x="PC1", y="PC2", color="True State",
                          color_discrete_map=COLOR_MAP,
                          title="True labels (for comparison)",
                          opacity=0.65)
        fig2.update_traces(marker_size=4)
        st.plotly_chart(fig2, use_container_width=True)

# ── Live ZONE Predictor ───────────────────────────────────────────────────
elif page == "Live ZONE Predictor":
    st.title("Live ZONE Predictor")
    st.markdown("Input an athlete's biometric readings and predict their state.")

    col1, col2 = st.columns(2)
    with col1:
        heart_rate   = st.slider("Heart rate (bpm)",      50, 130, 72)
        hrv          = st.slider("HRV",                   10, 100, 48)
        reaction     = st.slider("Reaction time (ms)",    100, 600, 260)
        accuracy     = st.slider("Decision accuracy (%)", 30, 100, 82)
        pupil        = st.slider("Pupil diameter",        0.2, 1.0, 0.68)
    with col2:
        blink        = st.slider("Blink rate (bpm)",      2, 40, 14)
        saccade      = st.slider("Saccade velocity",      100, 600, 340)
        cortisol     = st.slider("Cortisol proxy",        0.0, 1.0, 0.38)
        focus        = st.slider("Focus score (1-10)",    1.0, 10.0, 7.2)
        movement     = st.slider("Movement efficiency",   0.2, 1.0, 0.78)

    features = np.array([[heart_rate, hrv, reaction, accuracy,
                          pupil, blink, saccade, cortisol,
                          focus, movement]])

    X_all = df[SIGNALS].values
    scaler = StandardScaler().fit(X_all)
    X_sc   = scaler.transform(X_all)
    X_inp  = scaler.transform(features)

    km3 = KMeans(n_clusters=3, random_state=42, n_init=10)
    km3.fit(X_sc)

    cluster_map = {}
    for c in range(3):
        majority = df.iloc[km3.labels_==c]["state_label"].mode()[0]
        cluster_map[c] = majority

    pred_cluster = km3.predict(X_inp)[0]
    pred_state   = cluster_map[pred_cluster]
    icon = {"zone":"⚡","normal":"✓","fatigued":"⚠"}[pred_state]

    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    col1.metric("Predicted State", f"{icon} {pred_state.upper()}")
    col2.metric("Cluster", f"#{pred_cluster}")
    col3.metric("Focus Score", f"{focus:.1f}/10")

    color = COLOR_MAP[pred_state]
    st.markdown(
        f"<div style='background:{color}22;border-left:4px solid {color};"
        f"padding:12px;border-radius:6px;margin-top:12px'>"
        f"<b style='color:{color}'>{pred_state.upper()}</b> — "
        f"{'Athlete is in peak performance state. Optimal conditions for competition.' if pred_state=='zone' else 'Athlete showing moderate performance indicators.' if pred_state=='normal' else 'Athlete showing signs of cognitive fatigue. Consider rest period.'}"
        f"</div>", unsafe_allow_html=True)