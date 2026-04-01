import streamlit as st
import pandas as pd
import os
from datetime import datetime, timedelta
from io import BytesIO

# PDF
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, Image
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet

import matplotlib.pyplot as plt

# ===============================
# FILE SETUP
# ===============================
if not os.path.exists("users.csv"):
    pd.DataFrame(columns=["username", "password"]).to_csv("users.csv", index=False)

if not os.path.exists("habit_data.csv"):
    pd.DataFrame(columns=["username","date","sleep","study","screen","exercise"]).to_csv("habit_data.csv", index=False)

# ===============================
# SESSION
# ===============================
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "user" not in st.session_state:
    st.session_state.user = ""

# ===============================
# AUTH
# ===============================
def show_auth():
    st.title("🔐 Habit Tracker Login")

    tab1, tab2 = st.tabs(["Login", "Sign Up"])

    # LOGIN
    with tab1:
        username = st.text_input("Username", key="login_user")
        password = st.text_input("Password", type="password", key="login_pass")

        if st.button("Login"):
            username = username.strip()
            password = password.strip()

            users_df = pd.read_csv("users.csv")
            users_df["username"] = users_df["username"].astype(str).str.strip()
            users_df["password"] = users_df["password"].astype(str).str.strip()

            user = users_df[
                (users_df["username"] == username) &
                (users_df["password"] == password)
            ]

            if not user.empty:
                st.session_state.logged_in = True
                st.session_state.user = username
                st.rerun()
            else:
                st.error("Invalid credentials")

    # SIGNUP
    with tab2:
        new_user = st.text_input("New Username", key="signup_user")
        new_pass = st.text_input("New Password", type="password", key="signup_pass")

        if st.button("Create Account"):
            new_user = new_user.strip()
            new_pass = new_pass.strip()

            users_df = pd.read_csv("users.csv")
            users_df["username"] = users_df["username"].astype(str).str.strip()

            if new_user in users_df["username"].values:
                st.error("User already exists")
            else:
                pd.DataFrame([[new_user, new_pass]], columns=["username","password"])\
                    .to_csv("users.csv", mode='a', header=False, index=False)

                st.success("Account created! Now login.")
                st.rerun()

# ===============================
# LOGIN FIRST
# ===============================
if not st.session_state.logged_in:
    show_auth()
    st.stop()

# ===============================
# MAIN APP
# ===============================
st.title("📊 Habit Tracker")
st.write(f"Welcome, {st.session_state.user} 👋")

# ===== INPUT =====
sleep = st.slider("Sleep Hours", 0, 12)
study = st.slider("Study Hours", 0, 12)
screen = st.slider("Screen Time", 0, 12)
exercise = st.slider("Exercise", 0, 5)

# ===== SAVE =====
if st.button("Save Today's Data"):
    data = {
        "username": st.session_state.user,
        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),  # 🔥 UNIQUE TIME
        "sleep": sleep,
        "study": study,
        "screen": screen,
        "exercise": exercise
    }

    pd.DataFrame([data]).to_csv("habit_data.csv", mode='a', header=False, index=False)

    st.success("Saved!")
    st.rerun()

# ===============================
# LOAD DATA
# ===============================
try:
    df = pd.read_csv("habit_data.csv", on_bad_lines='skip')

    required_cols = ["username","date","sleep","study","screen","exercise"]
    for col in required_cols:
        if col not in df.columns:
            df[col] = None

    df["date"] = pd.to_datetime(df["date"], errors='coerce')

    # 🔥 IMPORTANT FIX
    df = df.sort_values("date")

    user_df = df[df["username"] == st.session_state.user]

except:
    user_df = pd.DataFrame(columns=["username","date","sleep","study","screen","exercise"])

# ===============================
# GRAPH (FIXED)
# ===============================
st.subheader("📈 Your Habit Trends")

if not user_df.empty:
    chart_df = user_df.set_index("date")[["sleep","study","screen"]]

    # 🔥 REMOVE DUPLICATE DATES
    chart_df = chart_df[~chart_df.index.duplicated(keep='last')]

    st.line_chart(chart_df, height=400)
else:
    st.info("No data yet")

# ===============================
# PDF
# ===============================
def generate_pdf(df, title):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer)
    styles = getSampleStyleSheet()
    elements = []

    elements.append(Paragraph(title, styles["Title"]))
    elements.append(Spacer(1, 10))

    elements.append(Paragraph("Summary", styles["Heading2"]))
    elements.append(Paragraph(f"Avg Sleep: {df['sleep'].mean():.2f}", styles["Normal"]))
    elements.append(Paragraph(f"Avg Study: {df['study'].mean():.2f}", styles["Normal"]))
    elements.append(Paragraph(f"Avg Screen: {df['screen'].mean():.2f}", styles["Normal"]))

    fig, ax = plt.subplots()
    df.set_index("date")[["sleep","study","screen"]].plot(ax=ax)

    img = BytesIO()
    plt.savefig(img, format="png")
    plt.close()
    img.seek(0)

    elements.append(Image(img, width=400, height=200))

    doc.build(elements)
    buffer.seek(0)
    return buffer

# ===============================
# REPORTS
# ===============================
st.subheader("📄 Download Reports")

today = datetime.now()

week_df = user_df[user_df["date"] >= (today - timedelta(days=7))]
month_df = user_df[user_df["date"] >= (today - timedelta(days=30))]

st.download_button("Weekly CSV", week_df.to_csv(index=False), "weekly.csv")
st.download_button("Monthly CSV", month_df.to_csv(index=False), "monthly.csv")

if not week_df.empty:
    st.download_button("Weekly PDF", generate_pdf(week_df, "Weekly Report"), "weekly.pdf")

if not month_df.empty:
    st.download_button("Monthly PDF", generate_pdf(month_df, "Monthly Report"), "monthly.pdf")

# ===============================
# LOGOUT
# ===============================
if st.button("Logout"):
    st.session_state.logged_in = False
    st.rerun()