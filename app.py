import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
import plotly.graph_objects as go
import plotly.express as px

# ── Page Config ───────────────────────────────────────────────
st.set_page_config(
    page_title="HabitIQ — Productivity Predictor",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── CSS ───────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=Space+Grotesk:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', sans-serif;
}
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 0 2rem 3rem 2rem !important; max-width: 1280px; }
.stApp { background: #f8f5f0 !important; }

/* ── HERO ── */
.hero {
    background: #1a1a2e;
    border-radius: 28px;
    padding: 60px 64px;
    margin-bottom: 32px;
    position: relative;
    overflow: hidden;
}
.hero-dot-grid {
    position: absolute;
    top: 0; right: 0; bottom: 0;
    width: 45%;
    background-image: radial-gradient(circle, rgba(255,255,255,0.06) 1px, transparent 1px);
    background-size: 22px 22px;
}
.hero-glow {
    position: absolute;
    top: -100px; right: -60px;
    width: 400px; height: 400px;
    background: radial-gradient(circle, rgba(251,146,60,0.15) 0%, transparent 65%);
    border-radius: 50%;
}
.hero-pill {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    background: rgba(251,146,60,0.12);
    border: 1px solid rgba(251,146,60,0.3);
    color: #fb923c;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 2.5px;
    text-transform: uppercase;
    padding: 7px 18px;
    border-radius: 100px;
    margin-bottom: 24px;
}
.hero-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 56px;
    font-weight: 700;
    color: #ffffff;
    line-height: 1.05;
    margin: 0 0 18px 0;
    letter-spacing: -1.5px;
}
.hero-title .accent { color: #fb923c; }
.hero-desc {
    font-size: 16px;
    color: #9ca3af;
    font-weight: 400;
    max-width: 500px;
    line-height: 1.75;
    margin: 0;
}
.hero-chips {
    display: flex;
    gap: 10px;
    margin-top: 36px;
    flex-wrap: wrap;
}
.hero-chip {
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.1);
    color: #d1d5db;
    border-radius: 8px;
    padding: 8px 18px;
    font-size: 13px;
    font-weight: 500;
}

/* ── SECTION LABEL ── */
.sec-label {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 11px;
    font-weight: 700;
    color: #9ca3af;
    letter-spacing: 2.5px;
    text-transform: uppercase;
    margin-bottom: 8px;
}
.sec-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 22px;
    font-weight: 700;
    color: #111827;
    margin: 0 0 24px 0;
}

/* ── INPUT CARD ── */
.input-card {
    background: #ffffff;
    border-radius: 20px;
    padding: 32px;
    border: 1px solid #e5e7eb;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    margin-bottom: 16px;
}
.input-card-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 15px;
    font-weight: 600;
    color: #374151;
    margin-bottom: 18px;
    display: flex;
    align-items: center;
    gap: 8px;
}

/* ── SLIDERS ── */
.stSlider > div > div > div {
    background: linear-gradient(90deg, #f97316, #fb923c) !important;
}
.stSlider label {
    color: #6b7280 !important;
    font-size: 13px !important;
    font-weight: 600 !important;
}

/* ── PREDICT BUTTON ── */
.stButton > button {
    background: #1a1a2e !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 14px !important;
    padding: 16px 40px !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 16px !important;
    font-weight: 700 !important;
    width: 100% !important;
    height: 58px !important;
    cursor: pointer !important;
    transition: all 0.2s !important;
    letter-spacing: 0.3px !important;
}
.stButton > button:hover {
    background: #fb923c !important;
    box-shadow: 0 8px 30px rgba(251,146,60,0.35) !important;
    transform: translateY(-1px) !important;
}

/* ── RESULT PRODUCTIVE ── */
.result-productive {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    border-radius: 24px;
    padding: 44px 36px;
    text-align: center;
    border: 2px solid #fb923c;
    position: relative;
    overflow: hidden;
}
.result-productive::after {
    content: '';
    position: absolute;
    bottom: -60px; right: -60px;
    width: 180px; height: 180px;
    background: radial-gradient(circle, rgba(251,146,60,0.2), transparent 70%);
    border-radius: 50%;
}
.result-not {
    background: linear-gradient(135deg, #1c1917 0%, #292524 100%);
    border-radius: 24px;
    padding: 44px 36px;
    text-align: center;
    border: 2px solid #6b7280;
    position: relative;
    overflow: hidden;
}
.result-emoji { font-size: 56px; margin-bottom: 12px; }
.result-verdict {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 30px;
    font-weight: 700;
    color: #fff;
    margin-bottom: 6px;
}
.result-conf {
    font-size: 14px;
    color: #9ca3af;
    margin-bottom: 24px;
}
.conf-bar-bg {
    background: rgba(255,255,255,0.08);
    border-radius: 100px;
    height: 8px;
    overflow: hidden;
    margin: 12px 0 20px 0;
}
.conf-bar-fill {
    height: 100%;
    border-radius: 100px;
    background: linear-gradient(90deg, #f97316, #fbbf24);
}

/* ── INSIGHT CARD ── */
.insight-item {
    background: #fff8f0;
    border: 1px solid #fed7aa;
    border-left: 4px solid #fb923c;
    border-radius: 10px;
    padding: 12px 16px;
    margin-bottom: 10px;
    font-size: 14px;
    color: #92400e;
    font-weight: 500;
}
.insight-good {
    background: #f0fdf4;
    border: 1px solid #bbf7d0;
    border-left: 4px solid #22c55e;
    border-radius: 10px;
    padding: 12px 16px;
    margin-bottom: 10px;
    font-size: 14px;
    color: #166534;
    font-weight: 500;
}

/* ── METRIC CARDS ── */
.metric-card {
    background: #ffffff;
    border: 1px solid #e5e7eb;
    border-radius: 16px;
    padding: 20px 24px;
    text-align: center;
}
.metric-card-val {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 28px;
    font-weight: 700;
    color: #1a1a2e;
}
.metric-card-label {
    font-size: 11px;
    color: #9ca3af;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    margin-top: 4px;
}

/* ── TABS ── */
.stTabs [data-baseweb="tab-list"] {
    background: #ffffff;
    border-radius: 14px;
    padding: 5px;
    gap: 4px;
    border: 1px solid #e5e7eb;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: #9ca3af !important;
    border-radius: 10px !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-weight: 600 !important;
    font-size: 14px !important;
}
.stTabs [aria-selected="true"] {
    background: #1a1a2e !important;
    color: #ffffff !important;
}

/* Selectbox */
div[data-baseweb="select"] > div {
    background: #f9fafb !important;
    border: 1px solid #e5e7eb !important;
    border-radius: 10px !important;
}
</style>
""", unsafe_allow_html=True)


# ── Model (cached) ────────────────────────────────────────────
@st.cache_resource
def build_model():
    np.random.seed(42)
    n = 1200
    sleep      = np.random.uniform(4, 10, n)
    study      = np.random.uniform(0, 10, n)
    screen     = np.random.uniform(1, 12, n)
    exercise   = np.random.uniform(0, 90, n)
    focus      = np.random.uniform(1, 10, n)
    meditation = np.random.uniform(0, 60, n)
    breaks     = np.random.randint(1, 8, n).astype(float)

    # Productivity score formula
    score = (
        sleep * 4 +
        study * 6 +
        (10 - screen) * 3 +
        exercise * 0.3 +
        focus * 7 +
        meditation * 0.2 +
        breaks * 1.5 +
        np.random.normal(0, 8, n)
    )
    productive = (score > np.percentile(score, 45)).astype(int)

    df = pd.DataFrame({
        'sleep_hours':        sleep.round(1),
        'study_hours':        study.round(1),
        'screen_time':        screen.round(1),
        'exercise_minutes':   exercise.round(0),
        'focus_score':        focus.round(1),
        'meditation_minutes': meditation.round(0),
        'breaks_taken':       breaks,
        'productive':         productive,
        'productivity_score': score.round(1)
    })

    X = df.drop(['productive','productivity_score'], axis=1)
    y = df['productive']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_train)
    X_te_sc = scaler.transform(X_test)

    models = {
        'Random Forest':     RandomForestClassifier(n_estimators=150, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42),
        'Logistic Regression': LogisticRegression(max_iter=1000),
    }
    results = {}
    for name, m in models.items():
        if name == 'Logistic Regression':
            m.fit(X_tr_sc, y_train)
            pred = m.predict(X_te_sc)
            prob = m.predict_proba(X_te_sc)
        else:
            m.fit(X_train, y_train)
            pred = m.predict(X_test)
            prob = m.predict_proba(X_test)
        results[name] = {'model': m, 'acc': accuracy_score(y_test, pred), 'pred': pred}

    rf = results['Random Forest']['model']
    importances = pd.Series(rf.feature_importances_, index=X.columns)

    return df, X, y, X_train, X_test, y_train, y_test, scaler, results, importances

df, X, y, X_train, X_test, y_train, y_test, scaler, results, importances = build_model()


# ── Insights ──────────────────────────────────────────────────
def generate_insights(d):
    bad, good = [], []
    if d['sleep_hours'] < 6:        bad.append("😴 Sleep is too low — aim for at least 7 hours")
    else:                            good.append("✅ Good sleep schedule!")
    if d['study_hours'] < 3:        bad.append("📚 Study hours are low — try at least 3 hrs/day")
    else:                            good.append("✅ Great study commitment!")
    if d['screen_time'] > 6:        bad.append("📵 High screen time is hurting focus — reduce it")
    else:                            good.append("✅ Screen time is well managed!")
    if d['exercise_minutes'] < 20:  bad.append("🏃 Exercise less than 20 min — move more!")
    else:                            good.append("✅ Active lifestyle — keep it up!")
    if d['focus_score'] < 5:        bad.append("🧠 Focus score is low — try Pomodoro technique")
    else:                            good.append("✅ Strong focus level!")
    if d['meditation_minutes'] < 10:bad.append("🧘 Little to no meditation — try 10 min/day")
    else:                            good.append("✅ Meditation habit is solid!")
    return bad, good


# ══════════════════════════════════════════════════════════════
#  HERO
# ══════════════════════════════════════════════════════════════
st.markdown("""
<div class="hero">
  <div class="hero-dot-grid"></div>
  <div class="hero-glow"></div>
  <div class="hero-pill">⚡ ML-Powered Analysis</div>
  <h1 class="hero-title">Habit<span class="accent">IQ</span><br>Productivity Predictor</h1>
  <p class="hero-desc">
    Enter your daily habits and instantly find out whether your routine is built
    for peak productivity — backed by machine learning.
  </p>
  <div class="hero-chips">
    <div class="hero-chip">🛏 Sleep Analysis</div>
    <div class="hero-chip">📚 Study Tracking</div>
    <div class="hero-chip">🏃 Exercise</div>
    <div class="hero-chip">🧠 Focus Score</div>
    <div class="hero-chip">🧘 Meditation</div>
  </div>
</div>
""", unsafe_allow_html=True)


# ── TABS ──────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["  ⚡  Predict  ", "  📊  Analytics  ", "  🤖  Model Info  "])


# ══════════════════════════════════════════════════════════════
#  TAB 1 — PREDICT
# ══════════════════════════════════════════════════════════════
with tab1:
    st.markdown("<br>", unsafe_allow_html=True)
    left, right = st.columns([1.05, 0.95], gap="large")

    with left:
        st.markdown('<div class="sec-label">Step 1</div><div class="sec-title">Enter Your Daily Habits</div>', unsafe_allow_html=True)

        with st.container():
            st.markdown('<div class="input-card">', unsafe_allow_html=True)
            st.markdown('<div class="input-card-title">🛏️ Rest & Recovery</div>', unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            with c1:
                sleep = st.slider("Sleep Hours", 3.0, 12.0, 7.0, 0.5,
                                  help="How many hours do you sleep per night?")
            with c2:
                meditation = st.slider("Meditation (min)", 0, 60, 10, 5,
                                       help="Minutes of meditation per day")
            st.markdown('</div>', unsafe_allow_html=True)

        with st.container():
            st.markdown('<div class="input-card">', unsafe_allow_html=True)
            st.markdown('<div class="input-card-title">📚 Work & Focus</div>', unsafe_allow_html=True)
            c3, c4 = st.columns(2)
            with c3:
                study = st.slider("Study / Work Hours", 0.0, 12.0, 4.0, 0.5,
                                  help="Hours spent studying or working")
            with c4:
                focus = st.slider("Focus Score (1–10)", 1.0, 10.0, 6.0, 0.5,
                                  help="Rate your ability to focus today")
            st.markdown('</div>', unsafe_allow_html=True)

        with st.container():
            st.markdown('<div class="input-card">', unsafe_allow_html=True)
            st.markdown('<div class="input-card-title">📱 Lifestyle</div>', unsafe_allow_html=True)
            c5, c6, c7 = st.columns(3)
            with c5:
                screen = st.slider("Screen Time (hrs)", 0.0, 14.0, 4.0, 0.5,
                                   help="Non-work screen time")
            with c6:
                exercise = st.slider("Exercise (min)", 0, 120, 30, 5,
                                     help="Minutes of physical exercise")
            with c7:
                breaks = st.slider("Breaks Taken", 1, 10, 4, 1,
                                   help="Number of short breaks during work")
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        predict_btn = st.button("⚡ Analyse My Productivity", use_container_width=True)

    # ── RIGHT: Result ─────────────────────────────────────────
    with right:
        st.markdown('<div class="sec-label">Step 2</div><div class="sec-title">Your Result</div>', unsafe_allow_html=True)

        if predict_btn:
            user_input = {
                'sleep_hours': sleep, 'study_hours': study,
                'screen_time': screen, 'exercise_minutes': float(exercise),
                'focus_score': focus, 'meditation_minutes': float(meditation),
                'breaks_taken': float(breaks)
            }
            input_df = pd.DataFrame([user_input])[X.columns]
            rf_model = results['Random Forest']['model']
            prediction = rf_model.predict(input_df)[0]
            probability = rf_model.predict_proba(input_df)[0][1]
            conf_pct = round(probability * 100, 1)

            # ── Result banner ─────────────────────────────────
            if prediction == 1:
                st.markdown(f"""
                <div class="result-productive">
                  <div class="result-emoji">🚀</div>
                  <div class="result-verdict">Productive!</div>
                  <div class="result-conf">Model confidence: {conf_pct}%</div>
                  <div class="conf-bar-bg">
                    <div class="conf-bar-fill" style="width:{conf_pct}%"></div>
                  </div>
                  <div style="color:#fb923c; font-size:13px; font-weight:600;">
                    Your habits are aligned for high output ✨
                  </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-not">
                  <div class="result-emoji">😐</div>
                  <div class="result-verdict">Not Productive</div>
                  <div class="result-conf">Model confidence: {round((1-probability)*100, 1)}%</div>
                  <div class="conf-bar-bg">
                    <div class="conf-bar-fill" style="width:{round((1-probability)*100,1)}%; background: linear-gradient(90deg,#6b7280,#9ca3af);"></div>
                  </div>
                  <div style="color:#9ca3af; font-size:13px; font-weight:600;">
                    Small habit changes can turn this around 💪
                  </div>
                </div>
                """, unsafe_allow_html=True)

            # ── Radar chart ───────────────────────────────────
            st.markdown("<br>", unsafe_allow_html=True)
            categories = ['Sleep', 'Study', 'Focus', 'Exercise', 'Meditation', 'Screen Control', 'Breaks']
            values = [
                min(sleep / 9 * 10, 10),
                min(study / 10 * 10, 10),
                focus,
                min(exercise / 90 * 10, 10),
                min(meditation / 45 * 10, 10),
                max(10 - screen, 0),
                min(breaks / 8 * 10, 10),
            ]
            fig_radar = go.Figure(go.Scatterpolar(
                r=values + [values[0]],
                theta=categories + [categories[0]],
                fill='toself',
                fillcolor='rgba(251,146,60,0.15)',
                line=dict(color='#fb923c', width=2),
                marker=dict(color='#fb923c', size=6),
            ))
            fig_radar.update_layout(
                polar=dict(
                    bgcolor='rgba(0,0,0,0)',
                    radialaxis=dict(visible=True, range=[0,10], gridcolor='#e5e7eb',
                                   tickfont=dict(size=9, color='#9ca3af'), linecolor='#e5e7eb'),
                    angularaxis=dict(gridcolor='#e5e7eb', linecolor='#e5e7eb',
                                    tickfont=dict(size=11, color='#374151', family='Space Grotesk'))
                ),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                margin=dict(l=40, r=40, t=20, b=20),
                height=280,
                showlegend=False,
            )
            st.plotly_chart(fig_radar, use_container_width=True, config={'displayModeBar': False})

            # ── Insights ──────────────────────────────────────
            bad_insights, good_insights = generate_insights(user_input)
            st.markdown("**🧠 Habit Insights**")
            for tip in bad_insights:
                st.markdown(f'<div class="insight-item">{tip}</div>', unsafe_allow_html=True)
            for tip in good_insights:
                st.markdown(f'<div class="insight-good">{tip}</div>', unsafe_allow_html=True)

        else:
            st.markdown("""
            <div style="background:#ffffff; border:2px dashed #e5e7eb; border-radius:20px;
                        padding:70px 30px; text-align:center;">
              <div style="font-size:52px; margin-bottom:16px;">⚡</div>
              <div style="font-family:'Space Grotesk',sans-serif; font-size:18px;
                          font-weight:700; color:#374151;">
                Set your habits & click Analyse
              </div>
              <div style="font-size:13px; color:#9ca3af; margin-top:8px;">
                Your productivity prediction will appear here
              </div>
            </div>
            """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
#  TAB 2 — ANALYTICS
# ══════════════════════════════════════════════════════════════
with tab2:
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="sec-label">Dataset Insights</div><div class="sec-title">Habit Analytics</div>', unsafe_allow_html=True)

    # Row 1
    c1, c2 = st.columns(2, gap="medium")
    with c1:
        fig1 = px.histogram(
            df, x='sleep_hours', color='productive',
            color_discrete_map={0:'#e5e7eb', 1:'#fb923c'},
            barmode='overlay', opacity=0.8,
            title='Sleep Hours vs Productivity',
            labels={'sleep_hours':'Sleep Hours','productive':'Productive'}
        )
        fig1.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font_color='#374151', title_font_family='Space Grotesk',
            title_font_size=15, title_font_color='#111827',
            height=300, legend=dict(bgcolor='rgba(0,0,0,0)'),
            xaxis=dict(gridcolor='#f3f4f6', linecolor='#e5e7eb'),
            yaxis=dict(gridcolor='#f3f4f6', linecolor='#e5e7eb'),
            margin=dict(l=10,r=10,t=40,b=10)
        )
        st.plotly_chart(fig1, use_container_width=True, config={'displayModeBar': False})

    with c2:
        fig2 = px.scatter(
            df.sample(300, random_state=1),
            x='study_hours', y='focus_score',
            color='productive',
            color_discrete_map={0:'#d1d5db', 1:'#fb923c'},
            opacity=0.7,
            title='Study Hours vs Focus Score',
            labels={'study_hours':'Study Hours','focus_score':'Focus Score (1–10)'}
        )
        fig2.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font_color='#374151', title_font_family='Space Grotesk',
            title_font_size=15, title_font_color='#111827',
            height=300, legend=dict(bgcolor='rgba(0,0,0,0)'),
            xaxis=dict(gridcolor='#f3f4f6', linecolor='#e5e7eb'),
            yaxis=dict(gridcolor='#f3f4f6', linecolor='#e5e7eb'),
            margin=dict(l=10,r=10,t=40,b=10)
        )
        st.plotly_chart(fig2, use_container_width=True, config={'displayModeBar': False})

    # Row 2
    c3, c4 = st.columns(2, gap="medium")
    with c3:
        imp_sorted = importances.sort_values()
        nice = {
            'sleep_hours':'Sleep Hours','study_hours':'Study Hours',
            'screen_time':'Screen Time','exercise_minutes':'Exercise',
            'focus_score':'Focus Score','meditation_minutes':'Meditation',
            'breaks_taken':'Breaks Taken'
        }
        fig3 = go.Figure(go.Bar(
            x=imp_sorted.values,
            y=[nice.get(f, f) for f in imp_sorted.index],
            orientation='h',
            marker=dict(
                color=imp_sorted.values,
                colorscale=[[0,'#f3f4f6'],[1,'#fb923c']],
                line=dict(width=0)
            ),
            text=[f"{v:.3f}" for v in imp_sorted.values],
            textposition='outside',
            textfont=dict(color='#9ca3af', size=11)
        ))
        fig3.update_layout(
            title='Feature Importance (Random Forest)',
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font_color='#374151', title_font_family='Space Grotesk',
            title_font_size=15, title_font_color='#111827',
            height=300, showlegend=False,
            xaxis=dict(gridcolor='#f3f4f6', linecolor='#e5e7eb', range=[0,0.5]),
            yaxis=dict(gridcolor='#f3f4f6', linecolor='#e5e7eb'),
            margin=dict(l=10,r=10,t=40,b=10)
        )
        st.plotly_chart(fig3, use_container_width=True, config={'displayModeBar': False})

    with c4:
        # Screen time vs productivity box
        fig4 = px.box(
            df, x='productive', y='screen_time',
            color='productive',
            color_discrete_map={0:'#d1d5db', 1:'#fb923c'},
            title='Screen Time: Productive vs Not',
            labels={'screen_time':'Screen Time (hrs)','productive':'Productive (1=Yes)'}
        )
        fig4.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font_color='#374151', title_font_family='Space Grotesk',
            title_font_size=15, title_font_color='#111827',
            height=300, showlegend=False,
            xaxis=dict(gridcolor='#f3f4f6', linecolor='#e5e7eb'),
            yaxis=dict(gridcolor='#f3f4f6', linecolor='#e5e7eb'),
            margin=dict(l=10,r=10,t=40,b=10)
        )
        st.plotly_chart(fig4, use_container_width=True, config={'displayModeBar': False})


# ══════════════════════════════════════════════════════════════
#  TAB 3 — MODEL INFO
# ══════════════════════════════════════════════════════════════
with tab3:
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="sec-label">Under the Hood</div><div class="sec-title">Model Performance</div>', unsafe_allow_html=True)

    # Model cards
    m1, m2, m3 = st.columns(3, gap="medium")
    icons_m = {'Random Forest':'🌲','Gradient Boosting':'⚡','Logistic Regression':'📐'}
    best = max(results, key=lambda k: results[k]['acc'])

    for col, (name, res) in zip([m1, m2, m3], results.items()):
        is_best = name == best
        with col:
            st.markdown(f"""
            <div style="background:#{'1a1a2e' if is_best else 'ffffff'};
                        border:2px solid #{'fb923c' if is_best else 'e5e7eb'};
                        border-radius:18px; padding:28px; text-align:center; position:relative;">
              {'<div style="position:absolute;top:-11px;left:50%;transform:translateX(-50%);background:#fb923c;color:#fff;font-size:10px;font-weight:800;padding:3px 14px;border-radius:100px;letter-spacing:1.5px;">BEST</div>' if is_best else ''}
              <div style="font-size:32px; margin-bottom:10px;">{icons_m[name]}</div>
              <div style="font-family:'Space Grotesk',sans-serif; font-size:15px; font-weight:700;
                          color:{'#fff' if is_best else '#111827'}; margin-bottom:18px;">{name}</div>
              <div style="font-family:'Space Grotesk',sans-serif; font-size:36px; font-weight:700;
                          color:#fb923c;">{res['acc']*100:.1f}%</div>
              <div style="font-size:11px; color:{'#9ca3af'}; text-transform:uppercase;
                          letter-spacing:1.5px; margin-top:4px;">Accuracy</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br><br>", unsafe_allow_html=True)

    # Productivity class breakdown
    c1, c2 = st.columns(2, gap="medium")
    with c1:
        prod_counts = df['productive'].value_counts()
        fig5 = go.Figure(go.Pie(
            labels=['Not Productive','Productive'],
            values=[prod_counts.get(0,0), prod_counts.get(1,0)],
            hole=0.6,
            marker=dict(colors=['#e5e7eb','#fb923c'], line=dict(width=0)),
            textfont=dict(family='Space Grotesk', size=13)
        ))
        fig5.add_annotation(text=f"<b>{prod_counts.get(1,0)}</b><br>Productive",
                            x=0.5, y=0.5, showarrow=False,
                            font=dict(size=16, family='Space Grotesk', color='#111827'))
        fig5.update_layout(
            title='Dataset Class Balance',
            paper_bgcolor='rgba(0,0,0,0)', height=300,
            font_color='#374151', title_font_family='Space Grotesk',
            title_font_size=15, title_font_color='#111827',
            legend=dict(bgcolor='rgba(0,0,0,0)'),
            margin=dict(l=10,r=10,t=40,b=10)
        )
        st.plotly_chart(fig5, use_container_width=True, config={'displayModeBar': False})

    with c2:
        # Avg habits comparison
        habit_cols = ['sleep_hours','study_hours','focus_score']
        prod_avgs  = df[df['productive']==1][habit_cols].mean()
        noprod_avgs= df[df['productive']==0][habit_cols].mean()
        nice_h = {'sleep_hours':'Sleep (hrs)','study_hours':'Study (hrs)','focus_score':'Focus Score'}

        fig6 = go.Figure()
        fig6.add_trace(go.Bar(
            name='Productive',
            x=[nice_h[c] for c in habit_cols],
            y=prod_avgs.values,
            marker_color='#fb923c',
            text=[f"{v:.1f}" for v in prod_avgs.values],
            textposition='outside'
        ))
        fig6.add_trace(go.Bar(
            name='Not Productive',
            x=[nice_h[c] for c in habit_cols],
            y=noprod_avgs.values,
            marker_color='#e5e7eb',
            text=[f"{v:.1f}" for v in noprod_avgs.values],
            textposition='outside'
        ))
        fig6.update_layout(
            title='Avg Habits: Productive vs Not',
            barmode='group',
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font_color='#374151', title_font_family='Space Grotesk',
            title_font_size=15, title_font_color='#111827',
            height=300,
            xaxis=dict(gridcolor='#f3f4f6', linecolor='#e5e7eb'),
            yaxis=dict(gridcolor='#f3f4f6', linecolor='#e5e7eb'),
            legend=dict(bgcolor='rgba(0,0,0,0)'),
            margin=dict(l=10,r=10,t=40,b=10)
        )
        st.plotly_chart(fig6, use_container_width=True, config={'displayModeBar': False})

    # Habit tips
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="sec-title">💡 Productivity Rules Used</div>', unsafe_allow_html=True)
    rules = [
        ("😴 Sleep < 6 hrs",    "Hurts productivity — rest is non-negotiable"),
        ("📚 Study < 3 hrs",    "Low study time correlates with unproductive days"),
        ("📱 Screen > 6 hrs",   "Excess screen time reduces deep work capacity"),
        ("🏃 Exercise < 20 min","Even light movement boosts cognitive output"),
        ("🧠 Focus < 5/10",     "Low focus score is the biggest predictor of poor output"),
        ("🧘 Meditation < 10 min","Even short sessions improve emotional regulation"),
    ]
    cols = st.columns(3)
    for i, (trigger, effect) in enumerate(rules):
        with cols[i % 3]:
            st.markdown(f"""
            <div style="background:#fff8f0; border:1px solid #fed7aa; border-radius:14px;
                        padding:18px; margin-bottom:12px;">
              <div style="font-family:'Space Grotesk',sans-serif; font-weight:700;
                          font-size:14px; color:#92400e; margin-bottom:6px;">{trigger}</div>
              <div style="font-size:13px; color:#b45309;">{effect}</div>
            </div>
            """, unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("""
<div style="text-align:center; color:#d1d5db; font-size:12px;
            padding:20px 0; border-top:1px solid #e5e7eb;">
  HabitIQ © 2025 &nbsp;·&nbsp; Built with Streamlit, Scikit-learn & Plotly
</div>
""", unsafe_allow_html=True)
