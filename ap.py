import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import os
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="ProductIQ — Productivity Intelligence",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ─────────────────────────────────────────────
# GLOBAL CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, [data-testid="stAppViewContainer"] {
    background: #0a0a0f !important;
    color: #e8e6e1 !important;
    font-family: 'DM Sans', sans-serif !important;
}

[data-testid="stAppViewContainer"] > .main { padding: 0 !important; }
[data-testid="stHeader"]  { display: none !important; }
[data-testid="stToolbar"] { display: none !important; }
.block-container { padding: 0 !important; max-width: 100% !important; }

::-webkit-scrollbar       { width: 4px; }
::-webkit-scrollbar-track { background: #0a0a0f; }
::-webkit-scrollbar-thumb { background: #3a3a5c; border-radius: 2px; }

[data-testid="stSelectbox"] > div > div {
    background: #13131f !important;
    border: 1px solid #2a2a3d !important;
    border-radius: 10px !important;
    color: #e8e6e1 !important;
}
[data-testid="stSelectbox"] label {
    color: #9b9bb5 !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

.stButton > button {
    background: linear-gradient(135deg, #7c6fff 0%, #4ecdc4 100%) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    font-size: 15px !important;
    padding: 0.6rem 2rem !important;
    letter-spacing: 0.05em !important;
    transition: opacity 0.2s !important;
    width: 100% !important;
}
.stButton > button:hover { opacity: 0.85 !important; }

[data-testid="stMetric"] {
    background: #13131f !important;
    border: 1px solid #2a2a3d !important;
    border-radius: 14px !important;
    padding: 1.2rem !important;
}
[data-testid="stMetricLabel"] { color: #9b9bb5 !important; font-size: 12px !important; }
[data-testid="stMetricValue"] { color: #e8e6e1 !important; font-size: 24px !important; font-weight: 700 !important; }

[data-testid="stTabs"] [role="tablist"] {
    gap: 4px;
    border-bottom: 1px solid #2a2a3d !important;
    background: transparent !important;
}
[data-testid="stTabs"] [role="tab"] {
    color: #9b9bb5 !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 600 !important;
    background: transparent !important;
    border: none !important;
    border-bottom: 2px solid transparent !important;
    padding: 0.5rem 1.2rem !important;
    border-radius: 0 !important;
}
[data-testid="stTabs"] [role="tab"][aria-selected="true"] {
    color: #7c6fff !important;
    border-bottom: 2px solid #7c6fff !important;
}

[data-testid="stAlert"] { border-radius: 12px !important; border: none !important; }
hr { border-color: #2a2a3d !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────
FEATURES = [
    "sleep_hours", "sleep_quality", "screen_time", "study_hours",
    "focus_score", "exercise_minutes", "tasks_completed",
    "procrastination", "distractions", "break_time", "mood", "stress_level"
]

DARK_BG     = "#0a0a0f"
CARD_BG     = "#13131f"
BORDER      = "#2a2a3d"
MUTED       = "#9b9bb5"
PURPLE      = "#7c6fff"
TEAL        = "#4ecdc4"
RED_SOFT    = "#ff6b6b"
DIM_PURPLE  = "#3a3a5c"

MAXV = {
    "sleep_hours": 10, "sleep_quality": 3, "screen_time": 12,
    "study_hours": 10, "focus_score": 10, "exercise_minutes": 120,
    "tasks_completed": 15, "procrastination": 10, "distractions": 20,
    "break_time": 60, "mood": 10, "stress_level": 10,
}
BAD_FEATURES = {"procrastination", "distractions", "screen_time", "stress_level"}

# ─────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────
@st.cache_resource
def load_or_train_model():
    model_path = "productivity_model.pkl"
    feat_path  = "feature_order.pkl"

    if os.path.exists(model_path) and os.path.exists(feat_path):
        return joblib.load(model_path), joblib.load(feat_path)

    np.random.seed(42)
    n = 10000
    df = pd.DataFrame({
        "sleep_hours":      np.random.uniform(3, 10, n),
        "sleep_quality":    np.random.randint(1, 4, n).astype(float),
        "screen_time":      np.random.uniform(0, 12, n),
        "study_hours":      np.random.uniform(0, 10, n),
        "focus_score":      np.random.uniform(1, 10, n),
        "exercise_minutes": np.random.uniform(0, 120, n),
        "tasks_completed":  np.random.randint(0, 15, n).astype(float),
        "procrastination":  np.random.uniform(1, 10, n),
        "distractions":     np.random.randint(0, 20, n).astype(float),
        "break_time":       np.random.uniform(0, 60, n),
        "mood":             np.random.uniform(1, 10, n),
        "stress_level":     np.random.uniform(1, 10, n),
    })
    score = (
        df.sleep_hours * 3 + df.sleep_quality * 4 +
        df.study_hours * 5 + df.focus_score * 4 +
        df.exercise_minutes * 0.1 + df.tasks_completed * 3 +
        df.mood * 3 - df.screen_time * 2 -
        df.procrastination * 3 - df.stress_level * 2 -
        df.distractions * 0.5
    )
    df["productive"] = (score > score.median()).astype(int)

    X = df[FEATURES]
    y = df["productive"]
    X_tr, _, y_tr, _ = train_test_split(X, y, test_size=0.2, random_state=42)
    rf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
    rf.fit(X_tr, y_tr)
    joblib.dump(rf, model_path)
    joblib.dump(FEATURES, feat_path)
    return rf, FEATURES


# ─────────────────────────────────────────────
# SHAP  (cached — computed once)
# ─────────────────────────────────────────────
@st.cache_resource
def get_shap_explainer(_rf):
    """Returns (explainer, bg_df, sv_2d).
    sv_2d is guaranteed shape (n_bg, n_features) — works with every SHAP version."""
    np.random.seed(42)
    n_bg = 300
    bg = pd.DataFrame({
        "sleep_hours":      np.random.uniform(3, 10, n_bg),
        "sleep_quality":    np.random.randint(1, 4, n_bg).astype(float),
        "screen_time":      np.random.uniform(0, 12, n_bg),
        "study_hours":      np.random.uniform(0, 10, n_bg),
        "focus_score":      np.random.uniform(1, 10, n_bg),
        "exercise_minutes": np.random.uniform(0, 120, n_bg),
        "tasks_completed":  np.random.randint(0, 15, n_bg).astype(float),
        "procrastination":  np.random.uniform(1, 10, n_bg),
        "distractions":     np.random.randint(0, 20, n_bg).astype(float),
        "break_time":       np.random.uniform(0, 60, n_bg),
        "mood":             np.random.uniform(1, 10, n_bg),
        "stress_level":     np.random.uniform(1, 10, n_bg),
    })[FEATURES]

    explainer = shap.TreeExplainer(_rf)
    raw = explainer.shap_values(bg)

    # Normalise to exactly (n_bg, n_features) ─ handles list, 2-D, 3-D outputs
    if isinstance(raw, list):
        sv = np.array(raw[1])
    else:
        sv = np.array(raw)
    if sv.ndim == 3:          # shape (n_bg, n_features, 2)
        sv = sv[:, :, 1]
    sv = sv.reshape(n_bg, len(FEATURES))
    return explainer, bg, sv


def safe_single_shap(explainer, row_df):
    """1-D SHAP array of length n_features for one row — immune to all SHAP version quirks."""
    raw = explainer.shap_values(row_df)
    if isinstance(raw, list):
        sv = np.array(raw[1])
    else:
        sv = np.array(raw)
    if sv.ndim == 3:
        sv = sv[:, :, 1]
    return sv.flatten()[:len(FEATURES)]   # trim to exact feature count


# ─────────────────────────────────────────────
# PLOT HELPERS
# ─────────────────────────────────────────────
def _style(ax, fig):
    fig.patch.set_facecolor(DARK_BG)
    ax.set_facecolor(CARD_BG)
    for sp in ax.spines.values():
        sp.set_edgecolor(BORDER)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.xaxis.label.set_color(MUTED)
    ax.yaxis.label.set_color(MUTED)
    ax.title.set_color(MUTED)


def fig_global_bar(sv_2d, names):
    """Mean |SHAP| horizontal bar — index-safe."""
    mean_abs = np.abs(sv_2d).mean(axis=0).flatten()
    names    = list(names)
    k        = min(len(mean_abs), len(names))
    mean_abs = mean_abs[:k]
    names    = names[:k]

    idx    = np.argsort(mean_abs)
    median = float(np.median(mean_abs))
    colors = [PURPLE if float(mean_abs[int(i)]) > median else DIM_PURPLE for i in idx]

    fig, ax = plt.subplots(figsize=(7, 5))
    _style(ax, fig)
    ax.barh([names[int(i)] for i in idx], mean_abs[idx], color=colors, height=0.6)
    ax.set_xlabel("Mean |SHAP value|", color=MUTED)
    ax.set_title("Global Feature Importance", color=MUTED, fontsize=11)
    plt.tight_layout()
    return fig


def fig_individual_bar(shap_1d, names):
    """Pos / neg SHAP bar for one prediction — index-safe."""
    shap_1d = np.array(shap_1d).flatten()
    names   = list(names)
    k       = min(len(shap_1d), len(names))
    shap_1d = shap_1d[:k]
    names   = names[:k]

    idx    = np.argsort(np.abs(shap_1d))
    colors = [TEAL if float(shap_1d[int(i)]) > 0 else RED_SOFT for i in idx]

    fig, ax = plt.subplots(figsize=(8, 5))
    _style(ax, fig)
    ax.barh([names[int(i)] for i in idx], shap_1d[idx], color=colors, height=0.6)
    ax.axvline(0, color="#5a5a7a", linewidth=0.8)
    ax.set_title("Green = pushes toward productive  |  Red = pushes away",
                 color=MUTED, fontsize=10)
    plt.tight_layout()
    return fig


def fig_dependency(sv_2d, bg_df, feature):
    """SHAP dependency scatter for chosen feature."""
    col_idx = list(bg_df.columns).index(feature)
    x_vals  = bg_df[feature].values
    y_vals  = sv_2d[:, col_idx]

    fig, ax = plt.subplots(figsize=(6, 4))
    _style(ax, fig)
    sc = ax.scatter(x_vals, y_vals, c=y_vals, cmap="coolwarm",
                    alpha=0.6, s=18, edgecolors="none")
    cb = plt.colorbar(sc, ax=ax, label="SHAP value")
    cb.ax.yaxis.label.set_color(MUTED)
    cb.ax.tick_params(colors=MUTED)
    ax.axhline(0, color=BORDER, linewidth=0.8)
    ax.set_xlabel(feature, color=MUTED)
    ax.set_ylabel("SHAP value", color=MUTED)
    ax.set_title(f"SHAP Dependency: {feature}", color=MUTED, fontsize=11)
    plt.tight_layout()
    return fig


def fig_radar(user_data):
    labels = ["Sleep\nHours", "Focus\nScore", "Mood",
              "Study\nHours", "Exercise\n(scaled)", "Tasks\nDone"]
    raw = [
        user_data["sleep_hours"]      / 10,
        user_data["focus_score"]      / 10,
        user_data["mood"]             / 10,
        user_data["study_hours"]      / 10,
        user_data["exercise_minutes"] / 120,
        user_data["tasks_completed"]  / 15,
    ]
    N      = len(labels)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    vals   = raw + [raw[0]]
    angles = angles + [angles[0]]

    fig, ax = plt.subplots(figsize=(5, 5), subplot_kw=dict(polar=True))
    fig.patch.set_facecolor(DARK_BG)
    ax.set_facecolor(CARD_BG)
    ax.plot(angles, vals, color=PURPLE, linewidth=2)
    ax.fill(angles, vals, color=PURPLE, alpha=0.25)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, color=MUTED, fontsize=9)
    ax.set_yticklabels([])
    ax.spines["polar"].set_edgecolor(BORDER)
    ax.grid(color=BORDER, linewidth=0.5)
    ax.set_title("Habit Radar", color="#e8e6e1", fontsize=12, pad=15)
    plt.tight_layout()
    return fig


def fig_donut(prob):
    fig, ax = plt.subplots(figsize=(4, 4))
    fig.patch.set_facecolor(DARK_BG)
    ax.set_facecolor(DARK_BG)
    ax.pie([prob, 1 - prob], colors=[TEAL, RED_SOFT], startangle=90,
           wedgeprops=dict(width=0.45, edgecolor=DARK_BG))
    ax.text(0, 0, f"{prob*100:.1f}%\nProductive",
            ha="center", va="center", color="#e8e6e1", fontsize=13, fontweight="bold")
    ax.set_title("Confidence Breakdown", color=MUTED, fontsize=11, pad=8)
    plt.tight_layout()
    return fig


def fig_stress_mood(user_data):
    fig, axes = plt.subplots(1, 2, figsize=(7, 3))
    fig.patch.set_facecolor(DARK_BG)
    for ax, key, label, good_low in zip(
        axes,
        ["stress_level", "mood"],
        ["Stress Level", "Mood"],
        [True, False],
    ):
        val   = float(user_data[key])
        color = (RED_SOFT
                 if (good_low and val > 5) or (not good_low and val < 5)
                 else TEAL)
        ax.set_facecolor(CARD_BG)
        ax.barh([label], [val],       color=color,  height=0.4)
        ax.barh([label], [10 - val],  left=[val],   color=BORDER, height=0.4)
        ax.set_xlim(0, 10)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"{label}: {val}/10", color=MUTED, fontsize=10)
        for sp in ax.spines.values():
            sp.set_edgecolor(BORDER)
    plt.tight_layout()
    return fig


def fig_habits_bar(user_data):
    keys   = list(user_data.keys())
    vals   = [user_data[k] / MAXV[k] for k in keys]
    colors = [RED_SOFT if k in BAD_FEATURES else PURPLE for k in keys]

    fig, ax = plt.subplots(figsize=(9, 4))
    _style(ax, fig)
    bars = ax.bar(range(len(keys)), vals, color=colors, width=0.6)
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels([k.replace("_", "\n") for k in keys], fontsize=8, color=MUTED)
    ax.set_ylabel("Normalised Value (0–1)", color=MUTED)
    ax.set_ylim(0, 1.18)
    ax.set_title("Your Habit Profile  (Purple = positive driver  |  Red = negative driver)",
                 color=MUTED, fontsize=10)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.02,
                f"{v:.2f}", ha="center", va="bottom", fontsize=7, color=MUTED)
    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────
# INSIGHTS
# ─────────────────────────────────────────────
def generate_insights(data):
    tips = []
    if data["sleep_hours"] < 6:
        tips.append(("🌙", "Sleep Deficit",
                     "You're getting less than 6 hours. Aim for 7–9 hours for optimal cognitive function."))
    if data["screen_time"] > data["study_hours"]:
        tips.append(("📱", "Screen Overload",
                     "Screen time exceeds study hours. Try the Pomodoro method to reclaim focus."))
    if data["focus_score"] < 5:
        tips.append(("🎯", "Low Focus Score",
                     "Consider 90-min deep work blocks with zero interruptions."))
    if data["exercise_minutes"] < 20:
        tips.append(("🏃", "Move More",
                     "Even 20 min of walking boosts BDNF, your brain's growth factor."))
    if data["procrastination"] > 6:
        tips.append(("⏳", "Procrastination Alert",
                     "'Eat the frog' — tackle your hardest task first thing every morning."))
    if data["stress_level"] > 7:
        tips.append(("🧘", "High Stress",
                     "Box breathing (4-4-4-4) can reduce cortisol within minutes."))
    if data["mood"] < 4:
        tips.append(("😊", "Mood Boost Needed",
                     "Gratitude journaling for 5 min daily rewires your brain for positivity."))
    return tips


# ─────────────────────────────────────────────
# SECTION LABEL HELPER
# ─────────────────────────────────────────────
def section_label(text):
    st.markdown(
        f"<p style='color:{MUTED};font-size:13px;text-transform:uppercase;"
        f"letter-spacing:.05em;margin:1rem 0 .4rem;'>{text}</p>",
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────
# MAIN APP  (no login)
# ─────────────────────────────────────────────
def main():
    rf, feature_cols         = load_or_train_model()
    explainer, bg_df, sv_global = get_shap_explainer(rf)

    # ── TOP NAV ──
    st.markdown("""
    <div style="background:#0d0d1a;border-bottom:1px solid #2a2a3d;padding:1rem 2.5rem;
         display:flex;align-items:center;justify-content:space-between;
         position:sticky;top:0;z-index:999;">
        <div style="font-family:'Syne',sans-serif;font-size:1.5rem;font-weight:800;
             background:linear-gradient(135deg,#7c6fff,#4ecdc4);
             -webkit-background-clip:text;-webkit-text-fill-color:transparent;">
             ⚡ ProductIQ</div>
        <div style="display:flex;gap:2rem;font-size:13px;color:#9b9bb5;">
            <span>Dashboard</span><span>Analytics</span><span>History</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='height:2rem'></div>", unsafe_allow_html=True)

    # ── HERO ──
    st.markdown("""
    <div style="text-align:center;padding:1rem 2rem 2.5rem;">
        <h1 style="font-family:'Syne',sans-serif;font-size:2.8rem;font-weight:800;
            color:#e8e6e1;margin-bottom:0.5rem;">How Productive Will You Be Today?</h1>
        <p style="color:#6b6b8a;font-size:1.05rem;">
            Fill in your daily habits below and let AI predict your productivity.</p>
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs([
        "🎯  Predict Productivity",
        "📊  SHAP Analysis",
        "📱  Habit Tracker Apps",
    ])

    # ══════════════════════════════════════════
    # TAB 1 — PREDICT
    # ══════════════════════════════════════════
    with tab1:
        with st.form("predict_form"):
            st.markdown("""
            <p style="font-family:'Syne',sans-serif;font-size:1.1rem;font-weight:700;
               color:#e8e6e1;margin-bottom:1.2rem;padding-top:1rem;">📋 Daily Habit Inputs</p>
            """, unsafe_allow_html=True)

            c1, c2, c3 = st.columns(3)
            with c1:
                sleep_hours     = st.selectbox("Sleep Hours",           [str(i) for i in range(3, 11)], index=4)
                sleep_quality   = st.selectbox("Sleep Quality",         ["1 — Poor", "2 — Average", "3 — Excellent"], index=1)
                screen_time     = st.selectbox("Screen Time (hrs)",     [str(x) for x in range(0, 13)], index=4)
                study_hours     = st.selectbox("Study / Work Hours",    [str(x) for x in range(0, 13)], index=5)
            with c2:
                focus_score     = st.selectbox("Focus Score (1–10)",    [str(x) for x in range(1, 11)], index=6)
                exercise_mins   = st.selectbox("Exercise (minutes)",    ["0","10","20","30","45","60","90","120"], index=3)
                tasks_done      = st.selectbox("Tasks Completed",       [str(x) for x in range(0, 16)], index=5)
                procrastination = st.selectbox("Procrastination (1–10)",[str(x) for x in range(1, 11)], index=3)
            with c3:
                distractions    = st.selectbox("Distractions (count)",  [str(x) for x in range(0, 21)], index=5)
                break_time      = st.selectbox("Break Time (minutes)",  ["0","10","20","30","45","60"], index=2)
                mood            = st.selectbox("Mood (1–10)",           [str(x) for x in range(1, 11)], index=6)
                stress_level    = st.selectbox("Stress Level (1–10)",   [str(x) for x in range(1, 11)], index=4)

            st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
            submitted = st.form_submit_button("⚡ Analyze My Productivity")

        if submitted:
            user_data = {
                "sleep_hours":      float(sleep_hours),
                "sleep_quality":    float(sleep_quality[0]),
                "screen_time":      float(screen_time),
                "study_hours":      float(study_hours),
                "focus_score":      float(focus_score),
                "exercise_minutes": float(exercise_mins),
                "tasks_completed":  float(tasks_done),
                "procrastination":  float(procrastination),
                "distractions":     float(distractions),
                "break_time":       float(break_time),
                "mood":             float(mood),
                "stress_level":     float(stress_level),
            }
            st.session_state.user_data = user_data

            input_df   = pd.DataFrame([user_data])[feature_cols]
            prediction = int(rf.predict(input_df)[0])
            prob       = float(rf.predict_proba(input_df)[0][1])
            st.session_state.prediction = prediction
            st.session_state.prob       = prob

            st.markdown("<div style='height:1.5rem'></div>", unsafe_allow_html=True)

            # ── Result banner ──
            color = TEAL if prediction == 1 else RED_SOFT
            icon  = "✅" if prediction == 1 else "❌"
            label = "PRODUCTIVE" if prediction == 1 else "NOT PRODUCTIVE"
            bg_c  = "#0f2a2a" if prediction == 1 else "#2a0f0f"

            st.markdown(f"""
            <div style="background:linear-gradient(135deg,{bg_c},#13131f);
                 border:1px solid {color}40;border-radius:16px;padding:2rem;
                 text-align:center;margin-bottom:1.5rem;">
                <div style="font-size:3rem;margin-bottom:0.5rem">{icon}</div>
                <div style="font-family:'Syne',sans-serif;font-size:2rem;font-weight:800;
                     color:{color};margin-bottom:0.5rem;">{label}</div>
                <div style="font-size:0.95rem;color:#9b9bb5;">
                    Model confidence:
                    <span style="color:{color};font-weight:700;font-size:1.2rem;">
                        {prob*100:.1f}%</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # ── Metrics row ──
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Sleep",        f"{user_data['sleep_hours']}h",
                      delta="Good"     if user_data["sleep_hours"]     >= 7 else "Low")
            m2.metric("Focus Score",  f"{user_data['focus_score']}/10",
                      delta="Strong"   if user_data["focus_score"]     >= 6 else "Weak")
            m3.metric("Tasks Done",   int(user_data["tasks_completed"]),
                      delta="On track" if user_data["tasks_completed"] >= 5 else "Behind")
            m4.metric("Stress",       f"{user_data['stress_level']}/10",
                      delta="OK"       if user_data["stress_level"]    <= 5 else "High",
                      delta_color="inverse")

            st.markdown("<div style='height:1.5rem'></div>", unsafe_allow_html=True)

            # ── Charts row 1 — Donut + Radar ──
            section_label("📊 Prediction Visualisations")
            col_d, col_r = st.columns(2)
            with col_d:
                st.markdown("<p style='color:#9b9bb5;font-size:12px;text-align:center;'>🍩 Confidence Donut</p>",
                            unsafe_allow_html=True)
                f = fig_donut(prob)
                st.pyplot(f); plt.close()

            with col_r:
                st.markdown("<p style='color:#9b9bb5;font-size:12px;text-align:center;'>🕸 Habit Radar</p>",
                            unsafe_allow_html=True)
                f = fig_radar(user_data)
                st.pyplot(f); plt.close()

            # ── Chart row 2 — Full habit bar ──
            section_label("📋 Full Habit Profile")
            f = fig_habits_bar(user_data)
            st.pyplot(f); plt.close()

            # ── Chart row 3 — Stress / Mood gauge ──
            section_label("⚖ Stress vs Mood Gauge")
            f = fig_stress_mood(user_data)
            st.pyplot(f); plt.close()

            st.markdown("<div style='height:1.5rem'></div>", unsafe_allow_html=True)

            # ── Personalised insights ──
            insights = generate_insights(user_data)
            if insights:
                section_label("🧠 Personalised Insights")
                cols = st.columns(min(len(insights), 3))
                for i, (emoji, title, desc) in enumerate(insights):
                    with cols[i % 3]:
                        st.markdown(f"""
                        <div style="background:#13131f;border:1px solid #2a2a3d;
                             border-radius:14px;padding:1.2rem;margin-bottom:1rem;">
                            <div style="font-size:1.6rem;margin-bottom:.5rem">{emoji}</div>
                            <div style="font-family:'Syne',sans-serif;font-weight:700;
                                 color:#e8e6e1;font-size:.95rem;margin-bottom:.4rem">{title}</div>
                            <div style="color:#9b9bb5;font-size:.82rem;line-height:1.5">{desc}</div>
                        </div>
                        """, unsafe_allow_html=True)
            else:
                st.success("🎉 Excellent habits! You're on track for a highly productive day.")

            # ── Habit tracker CTA ──
            st.markdown("---")
            st.markdown("""
            <p style="font-family:'Syne',sans-serif;font-size:1.2rem;font-weight:700;
                 color:#e8e6e1;text-align:center;">📈 Improve Your Productivity</p>
            <p style="text-align:center;color:#9b9bb5;margin-bottom:1rem;">
                Track your habits daily and boost consistency 🚀</p>
            """, unsafe_allow_html=True)
            st.link_button("Open My Habit Tracker App 🔗",
                           "https://your-app2-link.streamlit.app")

    # ══════════════════════════════════════════
    # TAB 2 — SHAP
    # ══════════════════════════════════════════
    with tab2:
        st.markdown("""
        <p style="font-family:'Syne',sans-serif;font-size:1.2rem;font-weight:700;
           color:#e8e6e1;padding-top:1rem;margin-bottom:1rem;">📊 SHAP Feature Importance</p>
        """, unsafe_allow_html=True)

        # ① Global bar
        section_label("① Global Feature Importance (Mean |SHAP|)")
        f = fig_global_bar(sv_global, feature_cols)
        st.pyplot(f); plt.close()

        st.markdown("<div style='height:1.5rem'></div>", unsafe_allow_html=True)

        # ② Dependency scatter
        section_label("② SHAP Dependency Plot — pick a feature")
        chosen = st.selectbox("Feature to explore:", feature_cols, key="dep_feat")
        f = fig_dependency(sv_global, bg_df, chosen)
        st.pyplot(f); plt.close()

        st.markdown("<div style='height:1.5rem'></div>", unsafe_allow_html=True)

        # ③ Individual breakdown
        section_label("③ Your Prediction Breakdown")
        if "user_data" in st.session_state:
            ud_df  = pd.DataFrame([st.session_state.user_data])[feature_cols]
            ind_sv = safe_single_shap(explainer, ud_df)    # (12,) — index-safe
            f      = fig_individual_bar(ind_sv, feature_cols)
            st.pyplot(f); plt.close()
        else:
            st.info("💡 Run a prediction in the **Predict Productivity** tab first "
                    "to see your personal SHAP breakdown here.")

    # ══════════════════════════════════════════
    # TAB 3 — APP SUGGESTIONS
    # ══════════════════════════════════════════
    with tab3:
        st.markdown("""
        <div style="padding:1rem 1rem 0;">
        <p style="font-family:'Syne',sans-serif;font-size:1.2rem;font-weight:700;
           color:#e8e6e1;margin-bottom:.4rem;">📱 Recommended Habit Tracking Apps</p>
        <p style="color:#6b6b8a;font-size:.9rem;margin-bottom:2rem;">
            These apps complement your ProductIQ insights — use them daily for lasting change.
        </p>
        </div>
        """, unsafe_allow_html=True)

        apps = [
            {"emoji":"🔥","name":"Habitica","category":"Gamified Habits",
             "desc":"Turn your habits and daily tasks into an RPG adventure. Earn XP, level up, and join parties with friends.",
             "platforms":["iOS","Android","Web"],"best_for":"People who need motivation through gamification",
             "link":"https://habitica.com","color":"#7c6fff"},
            {"emoji":"🌱","name":"Streaks","category":"Habit Streaks",
             "desc":"Build up to 12 habits per day. Beautiful Apple Watch integration. Minimalist and focused.",
             "platforms":["iOS","macOS","watchOS"],"best_for":"Apple ecosystem users",
             "link":"https://streaksapp.com","color":"#4ecdc4"},
            {"emoji":"📓","name":"Notion","category":"All-in-One Planner",
             "desc":"Custom habit trackers, daily journals, task databases. Infinite flexibility with powerful templates.",
             "platforms":["iOS","Android","Web","Desktop"],"best_for":"Power users who love customization",
             "link":"https://notion.so","color":"#f7b731"},
            {"emoji":"⏱","name":"Forest","category":"Focus & Screen Time",
             "desc":"Plant virtual trees by staying off your phone. Gamifies focus sessions and plants real trees.",
             "platforms":["iOS","Android","Chrome"],"best_for":"Reducing phone screen time",
             "link":"https://www.forestapp.cc","color":"#26de81"},
            {"emoji":"😴","name":"Sleep Cycle","category":"Sleep Tracking",
             "desc":"Tracks sleep cycles using your phone's microphone. Wakes you during your lightest sleep phase.",
             "platforms":["iOS","Android"],"best_for":"Improving sleep quality scores",
             "link":"https://www.sleepcycle.com","color":"#a29bfe"},
            {"emoji":"🧘","name":"Calm","category":"Stress & Mindfulness",
             "desc":"Guided meditations, sleep stories, and breathing exercises. Top-rated mental wellness app.",
             "platforms":["iOS","Android","Web"],"best_for":"Reducing stress and anxiety",
             "link":"https://calm.com","color":"#74b9ff"},
            {"emoji":"📊","name":"Exist.io","category":"Quantified Self",
             "desc":"Correlates habits, mood, and health data from 30+ services to find what makes you productive.",
             "platforms":["iOS","Android","Web"],"best_for":"Data-driven self-improvement",
             "link":"https://exist.io","color":"#fd79a8"},
            {"emoji":"💪","name":"Todoist","category":"Task Management",
             "desc":"Powerful task manager with Karma points for productivity streaks. Integrates with everything.",
             "platforms":["iOS","Android","Web","Desktop"],"best_for":"Managing daily tasks and projects",
             "link":"https://todoist.com","color":"#e17055"},
            {"emoji":"⚡","name":"Focusmate","category":"Accountability",
             "desc":"Virtual co-working with a live partner via video. Body-doubling technique eliminates procrastination.",
             "platforms":["Web","iOS","Android"],"best_for":"Beating procrastination with accountability",
             "link":"https://focusmate.com","color":"#88bebd"},
        ]

        for row_start in range(0, len(apps), 3):
            cols = st.columns(3)
            for ci, app in enumerate(apps[row_start:row_start + 3]):
                with cols[ci]:
                    platform_html = "".join(
                        f'<span style="background:#1e1e2e;border:1px solid #2a2a3d;'
                        f'border-radius:6px;padding:2px 8px;font-size:11px;color:#9b9bb5;">'
                        f'{p}</span>'
                        for p in app["platforms"]
                    )
                    st.markdown(f"""
                    <a href="{app['link']}" target="_blank" style="text-decoration:none;">
                    <div style="background:#13131f;border:1px solid #2a2a3d;border-radius:16px;
                         padding:1.4rem;margin-bottom:1rem;border-top:3px solid {app['color']};">
                        <div style="display:flex;align-items:center;gap:10px;margin-bottom:.8rem;">
                            <span style="font-size:1.8rem;">{app['emoji']}</span>
                            <div>
                                <div style="font-family:'Syne',sans-serif;font-weight:700;
                                     color:#e8e6e1;font-size:1rem;">{app['name']}</div>
                                <div style="font-size:11px;color:{app['color']};font-weight:500;
                                     text-transform:uppercase;letter-spacing:.05em;">{app['category']}</div>
                            </div>
                        </div>
                        <p style="color:#9b9bb5;font-size:.83rem;line-height:1.5;margin-bottom:.8rem;">
                            {app['desc']}</p>
                        <div style="font-size:11px;color:#6b6b8a;margin-bottom:.5rem;">
                            <span style="color:#7c6fff;font-weight:600;">Best for:</span>
                            {app['best_for']}
                        </div>
                        <div style="display:flex;gap:6px;flex-wrap:wrap;">{platform_html}</div>
                    </div>
                    </a>
                    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────
main()
