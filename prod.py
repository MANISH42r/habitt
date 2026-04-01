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
[data-testid="stHeader"] { display: none !important; }
[data-testid="stToolbar"] { display: none !important; }
.block-container { padding: 0 !important; max-width: 100% !important; }

::-webkit-scrollbar { width: 4px; }
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
    font-family: 'DM Sans', sans-serif !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

[data-testid="stSlider"] > div { padding: 0 !important; }
[data-testid="stSlider"] label {
    color: #9b9bb5 !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}
.stSlider [data-baseweb="slider"] [data-testid="stThumbValue"] {
    color: #7c6fff !important;
    background: #1e1e2e !important;
}

[data-testid="stTextInput"] input {
    background: #13131f !important;
    border: 1px solid #2a2a3d !important;
    border-radius: 10px !important;
    color: #e8e6e1 !important;
    font-family: 'DM Sans', sans-serif !important;
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
    cursor: pointer !important;
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

[data-testid="stExpander"] {
    background: #13131f !important;
    border: 1px solid #2a2a3d !important;
    border-radius: 12px !important;
}

[data-testid="stAlert"] {
    border-radius: 12px !important;
    border: none !important;
}

hr { border-color: #2a2a3d !important; }
.stPlotlyChart, .stPlot { background: transparent !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "username" not in st.session_state:
    st.session_state.username = ""

USERS = {"admin": "password123", "demo": "demo123", "user": "prod2024"}

# ─────────────────────────────────────────────
# HELPER: GENERATE SYNTHETIC DATASET & TRAIN
# ─────────────────────────────────────────────
@st.cache_resource
def load_or_train_model():
    model_path = "productivity_model.pkl"
    feat_path  = "feature_order.pkl"

    features = [
        "sleep_hours", "sleep_quality", "screen_time", "study_hours",
        "focus_score", "exercise_minutes", "tasks_completed",
        "procrastination", "distractions", "break_time", "mood", "stress_level"
    ]

    if os.path.exists(model_path) and os.path.exists(feat_path):
        rf   = joblib.load(model_path)
        cols = joblib.load(feat_path)
        return rf, cols

    np.random.seed(42)
    n = 10000
    df = pd.DataFrame({
        "sleep_hours":      np.random.uniform(3, 10, n),
        "sleep_quality":    np.random.randint(1, 4, n),
        "screen_time":      np.random.uniform(0, 12, n),
        "study_hours":      np.random.uniform(0, 10, n),
        "focus_score":      np.random.uniform(1, 10, n),
        "exercise_minutes": np.random.uniform(0, 120, n),
        "tasks_completed":  np.random.randint(0, 15, n),
        "procrastination":  np.random.uniform(1, 10, n),
        "distractions":     np.random.randint(0, 20, n),
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

    X = df[features]
    y = df["productive"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    rf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
    rf.fit(X_tr, y_tr)
    joblib.dump(rf, model_path)
    joblib.dump(features, feat_path)
    return rf, features

# ─────────────────────────────────────────────
# INSIGHTS ENGINE
# ─────────────────────────────────────────────
def generate_insights(data):
    tips = []
    if data["sleep_hours"] < 6:
        tips.append(("🌙", "Sleep Deficit", "You're getting less than 6 hours. Aim for 7–9 hours for optimal cognitive function."))
    if data["screen_time"] > data["study_hours"]:
        tips.append(("📱", "Screen Overload", "Screen time exceeds study hours. Try the Pomodoro method to reclaim focus."))
    if data["focus_score"] < 5:
        tips.append(("🎯", "Low Focus Score", "Consider deep work sessions — 90 min blocks with zero interruptions."))
    if data["exercise_minutes"] < 20:
        tips.append(("🏃", "Move More", "Even 20 min of walking boosts BDNF, your brain's growth factor."))
    if data["procrastination"] > 6:
        tips.append(("⏳", "Procrastination Alert", "Try 'eat the frog' — tackle your hardest task first thing."))
    if data["stress_level"] > 7:
        tips.append(("🧘", "High Stress", "Box breathing (4-4-4-4) can reduce cortisol within minutes."))
    if data["mood"] < 4:
        tips.append(("😊", "Mood Boost Needed", "Gratitude journaling for 5 min daily rewires your brain for positivity."))
    return tips

# ─────────────────────────────────────────────
# LOGIN PAGE
# ─────────────────────────────────────────────
def login_page():
    st.markdown("""
    <div style="min-height:100vh; display:flex; align-items:center; justify-content:center;
         background: radial-gradient(ellipse at 30% 40%, #1a1040 0%, #0a0a0f 60%);">
        <div style="text-align:center; padding: 3rem;">
            <div style="font-family:'Syne',sans-serif; font-size:3.5rem; font-weight:800;
                 background: linear-gradient(135deg,#7c6fff,#4ecdc4); -webkit-background-clip:text;
                 -webkit-text-fill-color:transparent; margin-bottom:0.25rem;">
                ⚡ ProductIQ
            </div>
            <p style="color:#6b6b8a; font-size:1rem; margin-bottom:3rem; letter-spacing:0.1em; text-transform:uppercase;">
                Productivity Intelligence Dashboard
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 1.2, 1])
    with col2:
        st.markdown("""
        <div style="background:#13131f; border:1px solid #2a2a3d; border-radius:20px; padding:2.5rem; margin-top:-18rem;">
            <p style="font-family:'Syne',sans-serif; font-size:1.4rem; font-weight:700;
               color:#e8e6e1; text-align:center; margin-bottom:0.5rem;">Welcome Back</p>
            <p style="color:#6b6b8a; font-size:0.85rem; text-align:center; margin-bottom:2rem;">
                Sign in to access your productivity insights
            </p>
        </div>
        """, unsafe_allow_html=True)

        username = st.text_input("Username", placeholder="Enter username")
        password = st.text_input("Password", type="password", placeholder="Enter password")

        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

        if st.button("Sign In →"):
            if username in USERS and USERS[username] == password:
                st.session_state.logged_in = True
                st.session_state.username  = username
                st.rerun()
            else:
                st.error("Invalid credentials. Try: admin / password123")

        st.markdown("""
        <p style="color:#4a4a6a; font-size:12px; text-align:center; margin-top:1.5rem;">
            Demo credentials: <span style="color:#7c6fff">admin</span> /
            <span style="color:#7c6fff">password123</span>
        </p>
        """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# MAIN DASHBOARD
# ─────────────────────────────────────────────
def dashboard():
    rf, feature_cols = load_or_train_model()

    # ── TOP NAV ──
    st.markdown(f"""
    <div style="background:#0d0d1a; border-bottom:1px solid #2a2a3d; padding:1rem 2.5rem;
         display:flex; align-items:center; justify-content:space-between; position:sticky; top:0; z-index:999;">
        <div style="font-family:'Syne',sans-serif; font-size:1.5rem; font-weight:800;
             background:linear-gradient(135deg,#7c6fff,#4ecdc4); -webkit-background-clip:text;
             -webkit-text-fill-color:transparent;">⚡ ProductIQ</div>
        <div style="display:flex; gap:2rem; font-size:13px; color:#9b9bb5;">
            <span>Dashboard</span><span>Analytics</span><span>History</span>
        </div>
        <div style="display:flex; align-items:center; gap:12px;">
            <div style="width:34px; height:34px; border-radius:50%; background:linear-gradient(135deg,#7c6fff,#4ecdc4);
                 display:flex; align-items:center; justify-content:center; font-weight:700; font-size:13px; color:#fff;">
                {st.session_state.username[0].upper()}
            </div>
            <span style="color:#e8e6e1; font-size:13px; font-weight:500;">{st.session_state.username}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='height:2rem'></div>", unsafe_allow_html=True)

    # ── HERO ──
    st.markdown("""
    <div style="text-align:center; padding: 1rem 2rem 2.5rem;">
        <h1 style="font-family:'Syne',sans-serif; font-size:2.8rem; font-weight:800; color:#e8e6e1; margin-bottom:0.5rem;">
            How Productive Will You Be Today?
        </h1>
        <p style="color:#6b6b8a; font-size:1.05rem;">Fill in your daily habits below and let AI predict your productivity.</p>
    </div>
    """, unsafe_allow_html=True)

    # ── TABS ──
    tab1, tab2, tab3 = st.tabs(["🎯  Predict Productivity", "📊  SHAP Analysis", "📱  Habit Tracker Apps"])

    # ══════════════════════════════════
    # TAB 1: INPUT + PREDICTION
    # ══════════════════════════════════
    with tab1:
        st.markdown("<div style='padding: 1rem 1rem 0'>", unsafe_allow_html=True)

        with st.form("predict_form"):
            st.markdown("""
            <p style="font-family:'Syne',sans-serif; font-size:1.1rem; font-weight:700;
               color:#e8e6e1; margin-bottom:1.2rem;">📋 Daily Habit Inputs</p>
            """, unsafe_allow_html=True)

            c1, c2, c3 = st.columns(3)
            with c1:
                sleep_hours     = st.selectbox("Sleep Hours",         [str(i) for i in range(3, 11)], index=4)
                sleep_quality   = st.selectbox("Sleep Quality",       ["1 — Poor", "2 — Average", "3 — Excellent"], index=1)
                screen_time     = st.selectbox("Screen Time (hrs)",   [str(x) for x in range(0, 13)], index=4)
                study_hours     = st.selectbox("Study / Work Hours",  [str(x) for x in range(0, 13)], index=5)
            with c2:
                focus_score     = st.selectbox("Focus Score (1–10)",  [str(x) for x in range(1, 11)], index=6)
                exercise_mins   = st.selectbox("Exercise (minutes)",  ["0","10","20","30","45","60","90","120"], index=3)
                tasks_done      = st.selectbox("Tasks Completed",     [str(x) for x in range(0, 16)], index=5)
                procrastination = st.selectbox("Procrastination (1–10)", [str(x) for x in range(1, 11)], index=3)
            with c3:
                distractions    = st.selectbox("Distractions (count)",[str(x) for x in range(0, 21)], index=5)
                break_time      = st.selectbox("Break Time (minutes)",["0","10","20","30","45","60"], index=2)
                mood            = st.selectbox("Mood (1–10)",         [str(x) for x in range(1, 11)], index=6)
                stress_level    = st.selectbox("Stress Level (1–10)", [str(x) for x in range(1, 11)], index=4)

            st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
            submitted = st.form_submit_button("⚡ Analyze My Productivity")

        if submitted:
            user_data = {
                "sleep_hours":      float(sleep_hours),
                "sleep_quality":    int(sleep_quality[0]),
                "screen_time":      float(screen_time),
                "study_hours":      float(study_hours),
                "focus_score":      float(focus_score),
                "exercise_minutes": float(exercise_mins),
                "tasks_completed":  int(tasks_done),
                "procrastination":  float(procrastination),
                "distractions":     int(distractions),
                "break_time":       float(break_time),
                "mood":             float(mood),
                "stress_level":     float(stress_level),
            }
            st.session_state.user_data = user_data

            input_df   = pd.DataFrame([user_data])[feature_cols]
            prediction = rf.predict(input_df)[0]
            prob       = rf.predict_proba(input_df)[0][1]

            st.session_state.prediction = prediction
            st.session_state.prob       = prob

            st.markdown("<div style='height:1.5rem'></div>", unsafe_allow_html=True)

            # Result banner
            if prediction == 1:
                color, icon, label = "#4ecdc4", "✅", "PRODUCTIVE"
            else:
                color, icon, label = "#ff6b6b", "❌", "NOT PRODUCTIVE"

            st.markdown(f"""
            <div style="background: linear-gradient(135deg, {'#0f2a2a' if prediction==1 else '#2a0f0f'}, #13131f);
                 border: 1px solid {color}40; border-radius: 16px; padding: 2rem; text-align:center; margin-bottom:1.5rem;">
                <div style="font-size:3rem; margin-bottom:0.5rem">{icon}</div>
                <div style="font-family:'Syne',sans-serif; font-size:2rem; font-weight:800; color:{color}; margin-bottom:0.5rem;">
                    {label}
                </div>
                <div style="font-size:0.95rem; color:#9b9bb5;">
                    Model confidence: <span style="color:{color}; font-weight:700; font-size:1.2rem;">{prob*100:.1f}%</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Habit Tracker CTA
            st.markdown("---")
            st.markdown("""
            <p style="font-family:'Syne',sans-serif; font-size:1.2rem; font-weight:700;
                 color:#e8e6e1; text-align:center;">
                 📈 Improve Your Productivity
            </p>
            """, unsafe_allow_html=True)
            st.markdown("""
            <p style="text-align:center; color:#9b9bb5; margin-bottom:1rem;">
                Track your habits daily and boost consistency 🚀
            </p>
            """, unsafe_allow_html=True)
            st.link_button(
                "Open My Habit Tracker App 🔗",
                "https://your-app2-link.streamlit.app"   # Replace with your actual URL
            )

            # Metrics row
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Sleep",      f"{user_data['sleep_hours']}h",    delta="Good"     if user_data['sleep_hours']    >= 7 else "Low")
            m2.metric("Focus Score",f"{user_data['focus_score']}/10",  delta="Strong"   if user_data['focus_score']    >= 6 else "Weak")
            m3.metric("Tasks Done", user_data['tasks_completed'],       delta="On track" if user_data['tasks_completed']>= 5 else "Behind")
            m4.metric("Stress",     f"{user_data['stress_level']}/10", delta="OK"       if user_data['stress_level']   <= 5 else "High", delta_color="inverse")

            st.markdown("<div style='height:1.5rem'></div>", unsafe_allow_html=True)

            # Insights
            insights = generate_insights(user_data)
            if insights:
                st.markdown("""
                <p style="font-family:'Syne',sans-serif; font-size:1rem; font-weight:700;
                   color:#e8e6e1; margin-bottom:1rem;">🧠 Personalised Insights</p>
                """, unsafe_allow_html=True)
                ic = st.columns(min(len(insights), 3))
                for i, (emoji, title, desc) in enumerate(insights):
                    with ic[i % 3]:
                        st.markdown(f"""
                        <div style="background:#13131f; border:1px solid #2a2a3d; border-radius:14px;
                             padding:1.2rem; margin-bottom:1rem;">
                            <div style="font-size:1.6rem; margin-bottom:0.5rem">{emoji}</div>
                            <div style="font-family:'Syne',sans-serif; font-weight:700; color:#e8e6e1;
                                 font-size:0.95rem; margin-bottom:0.4rem">{title}</div>
                            <div style="color:#9b9bb5; font-size:0.82rem; line-height:1.5">{desc}</div>
                        </div>
                        """, unsafe_allow_html=True)
            else:
                st.success("🎉 Excellent habits! You're on track for a highly productive day.")

        st.markdown("</div>", unsafe_allow_html=True)

    # ══════════════════════════════════
    # TAB 2: SHAP ANALYSIS  ← FIXED
    # ══════════════════════════════════
    with tab2:                                          # FIX 1: was "with s2:"
        st.markdown("""
        <p style="font-family:'Syne',sans-serif; font-size:1.2rem; font-weight:700;
           color:#e8e6e1; margin-bottom:1rem; padding: 1rem 1rem 0;">📊 SHAP Feature Importance</p>
        """, unsafe_allow_html=True)

        with st.spinner("Computing SHAP values…"):
            # FIX 2: Build explainer and background sample (were missing entirely)
            bg = shap.sample(
                pd.DataFrame(
                    np.random.default_rng(42).uniform(0, 1, (200, len(feature_cols))),
                    columns=feature_cols
                ),
                100,
                random_state=42,
            )
            # Use a small real-looking background so values are meaningful
            np.random.seed(42)
            n_bg = 200
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
            })[feature_cols]

            explainer = shap.TreeExplainer(rf)
            sv = explainer.shap_values(bg)

            # For binary classifiers shap_values returns a list [class0, class1]
            if isinstance(sv, list):
                sv = sv[1]

        # ── Global bar chart ──
        st.markdown("""
        <p style="font-family:'Syne',sans-serif; font-weight:600;
           color:#9b9bb5; font-size:0.85rem; text-transform:uppercase; letter-spacing:.05em;
           margin-bottom:.5rem;">Feature Importance (Bar)</p>
        """, unsafe_allow_html=True)

        fig2, ax2 = plt.subplots(figsize=(7, 5))
        fig2.patch.set_facecolor("#0a0a0f")
        ax2.set_facecolor("#13131f")              # FIX 3: was at wrong indentation level

        # FIX 4: All these lines were outside the with-tab2 block / wrong indents
        mean_abs   = np.abs(sv).mean(axis=0)
        mean_abs   = np.array(mean_abs).flatten()
        feat_names = list(bg.columns)
        sorted_idx = np.argsort(mean_abs)
        median_val = float(np.median(mean_abs))

        colors = []
        for v in mean_abs[sorted_idx]:
            v = float(v)
            colors.append("#7c6fff" if v > median_val else "#3a3a5c")

        ax2.barh(
            [feat_names[i] for i in sorted_idx],
            mean_abs[sorted_idx],
            color=colors,
            height=0.6,
        )
        for spine in ax2.spines.values():
            spine.set_edgecolor("#2a2a3d")
        ax2.tick_params(colors="#9b9bb5", labelsize=10)
        ax2.xaxis.label.set_color("#9b9bb5")
        ax2.set_xlabel("Mean |SHAP value|", color="#9b9bb5")

        plt.tight_layout()                        # FIX 5: had extra leading spaces
        st.pyplot(fig2)                           # FIX 6: had extra leading spaces
        plt.close()

        # ── Individual prediction SHAP ──
        if "user_data" in st.session_state:
            st.markdown("""
            <p style="font-family:'Syne',sans-serif; font-size:1rem; font-weight:700;
               color:#e8e6e1; margin: 1.5rem 0 0.5rem;">Your Prediction Breakdown</p>
            """, unsafe_allow_html=True)

            ud_df  = pd.DataFrame([st.session_state.user_data])[feature_cols]
            ind_sv = explainer.shap_values(ud_df)
            if isinstance(ind_sv, list):
                ind_sv = ind_sv[1]
            ind_sv     = np.array(ind_sv)[0]
            feat_names = list(ud_df.columns)
            sorted_i   = np.argsort(np.abs(ind_sv))

            fig3, ax3 = plt.subplots(figsize=(8, 5))
            fig3.patch.set_facecolor("#0a0a0f")
            ax3.set_facecolor("#13131f")
            bar_colors = ["#4ecdc4" if v > 0 else "#ff6b6b" for v in ind_sv[sorted_i]]
            ax3.barh([feat_names[i] for i in sorted_i], ind_sv[sorted_i], color=bar_colors, height=0.6)
            ax3.axvline(0, color="#5a5a7a", linewidth=0.8)
            for spine in ax3.spines.values():
                spine.set_edgecolor("#2a2a3d")
            ax3.tick_params(colors="#9b9bb5", labelsize=10)
            ax3.set_title(
                "Green = pushes toward productive  |  Red = pushes away",
                color="#6b6b8a", fontsize=10
            )
            plt.tight_layout()
            st.pyplot(fig3)
            plt.close()
        else:
            st.info("Run a prediction in the first tab to see your personal SHAP breakdown.")

    # ══════════════════════════════════
    # TAB 3: APP SUGGESTIONS
    # ══════════════════════════════════
    with tab3:
        st.markdown("""
        <div style="padding: 1rem 1rem 0;">
        <p style="font-family:'Syne',sans-serif; font-size:1.2rem; font-weight:700;
           color:#e8e6e1; margin-bottom:0.4rem;">📱 Recommended Habit Tracking Apps</p>
        <p style="color:#6b6b8a; font-size:0.9rem; margin-bottom:2rem;">
            These apps complement your ProductIQ insights — use them daily for lasting change.
        </p>
        </div>
        """, unsafe_allow_html=True)

        apps = [
            {
                "emoji": "🔥", "name": "Habitica", "category": "Gamified Habits",
                "desc": "Turn your habits and daily tasks into an RPG adventure. Earn XP, level up, and join parties with friends.",
                "platforms": ["iOS", "Android", "Web"], "best_for": "People who need motivation through gamification",
                "link": "https://habitica.com", "color": "#7c6fff"
            },
            {
                "emoji": "🌱", "name": "Streaks", "category": "Habit Streaks",
                "desc": "Build up to 12 habits per day. Beautiful Apple Watch integration. Minimalist and focused.",
                "platforms": ["iOS", "macOS", "watchOS"], "best_for": "Apple ecosystem users",
                "link": "https://streaksapp.com", "color": "#4ecdc4"
            },
            {
                "emoji": "📓", "name": "Notion", "category": "All-in-One Planner",
                "desc": "Custom habit trackers, daily journals, task databases. Infinite flexibility with powerful templates.",
                "platforms": ["iOS", "Android", "Web", "Desktop"], "best_for": "Power users who love customization",
                "link": "https://notion.so", "color": "#f7b731"
            },
            {
                "emoji": "⏱", "name": "Forest", "category": "Focus & Screen Time",
                "desc": "Plant virtual trees by staying off your phone. Gamifies focus sessions and plants real trees.",
                "platforms": ["iOS", "Android", "Chrome"], "best_for": "Reducing phone screen time",
                "link": "https://www.forestapp.cc", "color": "#26de81"
            },
            {
                "emoji": "😴", "name": "Sleep Cycle", "category": "Sleep Tracking",
                "desc": "Tracks sleep cycles using your phone's microphone. Wakes you during lightest sleep phase.",
                "platforms": ["iOS", "Android"], "best_for": "Improving sleep quality scores",
                "link": "https://www.sleepcycle.com", "color": "#a29bfe"
            },
            {
                "emoji": "🧘", "name": "Calm", "category": "Stress & Mindfulness",
                "desc": "Guided meditations, sleep stories, and breathing exercises. Top-rated mental wellness app.",
                "platforms": ["iOS", "Android", "Web"], "best_for": "Reducing stress and anxiety",
                "link": "https://calm.com", "color": "#74b9ff"
            },
            {
                "emoji": "📊", "name": "Exist.io", "category": "Quantified Self",
                "desc": "Correlates habits, mood, health data from 30+ services to find what actually makes you happy & productive.",
                "platforms": ["iOS", "Android", "Web"], "best_for": "Data-driven self-improvement",
                "link": "https://exist.io", "color": "#fd79a8"
            },
            {
                "emoji": "💪", "name": "Todoist", "category": "Task Management",
                "desc": "Powerful task manager with Karma points for productivity streaks. Integrates with everything.",
                "platforms": ["iOS", "Android", "Web", "Desktop"], "best_for": "Managing daily tasks and projects",
                "link": "https://todoist.com", "color": "#e17055"
            },
            {
                "emoji": "⚡", "name": "Focusmate", "category": "Accountability",
                "desc": "Virtual co-working with a live partner via video. Body doubling technique eliminates procrastination.",
                "platforms": ["Web", "iOS", "Android"], "best_for": "Beating procrastination with accountability",
                "link": "https://focusmate.com", "color": "#88bebd"
            },
        ]

        for row in range(0, len(apps), 3):
            cols = st.columns(3)
            for ci, app in enumerate(apps[row:row+3]):
                with cols[ci]:
                    st.markdown(f"""
                    <a href="{app['link']}" target="_blank" style="text-decoration:none;">
                    <div style="background:#13131f; border:1px solid #2a2a3d; border-radius:16px;
                         padding:1.4rem; margin-bottom:1rem; cursor:pointer;
                         transition:border-color 0.2s;
                         border-top: 3px solid {app['color']};">
                        <div style="display:flex; align-items:center; gap:10px; margin-bottom:0.8rem;">
                            <span style="font-size:1.8rem;">{app['emoji']}</span>
                            <div>
                                <div style="font-family:'Syne',sans-serif; font-weight:700;
                                     color:#e8e6e1; font-size:1rem;">{app['name']}</div>
                                <div style="font-size:11px; color:{app['color']}; font-weight:500;
                                     text-transform:uppercase; letter-spacing:0.05em;">{app['category']}</div>
                            </div>
                        </div>
                        <p style="color:#9b9bb5; font-size:0.83rem; line-height:1.5; margin-bottom:0.8rem;">{app['desc']}</p>
                        <div style="font-size:11px; color:#6b6b8a; margin-bottom:0.5rem;">
                            <span style="color:#7c6fff; font-weight:600;">Best for:</span> {app['best_for']}
                        </div>
                        <div style="display:flex; gap:6px; flex-wrap:wrap;">
                            {''.join(f'<span style="background:#1e1e2e; border:1px solid #2a2a3d; border-radius:6px; padding:2px 8px; font-size:11px; color:#9b9bb5;">{p}</span>' for p in app['platforms'])}
                        </div>
                    </div>
                    </a>
                    """, unsafe_allow_html=True)

        # Logout
        st.markdown("<div style='height:2rem'></div>", unsafe_allow_html=True)
        col_l, _, _ = st.columns([1, 3, 1])
        with col_l:
            if st.button("← Sign Out"):
                st.session_state.logged_in = False
                st.session_state.username  = ""
                st.rerun()

# ─────────────────────────────────────────────
# ROUTER
# ─────────────────────────────────────────────
if st.session_state.logged_in:
    dashboard()
else:
    login_page()