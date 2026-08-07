import streamlit as st
import pandas as pd
import numpy as np
import os
import hashlib
from datetime import datetime
from sklearn.linear_model import LogisticRegression
from google import genai
from dotenv import load_dotenv

# ====== CONFIG ======
load_dotenv()
try:
    client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
except:
    client = None
DEMO_MODE = True 
USER_FILE = "users.csv"
RECORD_FILE = "user_records.csv"

st.set_page_config(page_title="Athlete Injury Predictor", layout="wide")

# ====== UTILS ======
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def load_users():
    if not os.path.exists(USER_FILE):
        return pd.DataFrame(columns=["username", "password", "name", "age", "gender", "weight", "height", "sport", "past_injuries"])
    return pd.read_csv(USER_FILE)

def save_user(user_df):
    user_df.to_csv(USER_FILE, index=False)

def load_records():
    if not os.path.exists(RECORD_FILE):
        return pd.DataFrame(columns=["timestamp", "username", "sleep", "intensity", "prev_injury", "soreness", "risk_score"])
    return pd.read_csv(RECORD_FILE)

def save_record(record_df):
    record_df.to_csv(RECORD_FILE, index=False)

# ====== TRAIN FAKE MODEL ======
@st.cache_resource
def train_model():
    np.random.seed(42)
    data = {
        'Sleep': np.random.uniform(4, 10, 200),
        'Training_Intensity': np.random.uniform(3, 10, 200),
        'Previous_Injury': np.random.choice([0, 1], 200),
        'Muscle_Soreness': np.random.uniform(1, 10, 200)
    }
    df = pd.DataFrame(data)
    df['Risk'] = ((df['Sleep'] < 6) | (df['Training_Intensity'] > 8) |
                  (df['Previous_Injury'] == 1) | (df['Muscle_Soreness'] > 7)).astype(int)

    X = df[['Sleep', 'Training_Intensity', 'Previous_Injury', 'Muscle_Soreness']]
    y = df['Risk']
    model = LogisticRegression()
    model.fit(X, y)
    return model

model = train_model()

# ====== LLM FUNCTION ======
def get_llm_advice(risk_score, daily_data):
    if DEMO_MODE or client is None:
        if risk_score > 70:
            return "⚠️ **High Risk**: Aaj rest le bhai. 15 min stretching kar. Raat ko 8 ghante sona zaroori hai."
        elif risk_score > 40:
            return "🟡 **Moderate Risk**: Light workout hi kar. Pani peete reh aur warm-up 10 min zyada kar."
        else:
            return "🟢 **Low Risk**: Tu bilkul fit hai. Normal training kar sakta hai. Hydrated reh."

    prompt = f"""You are a sports injury doctor. Talk like a friend in Hinglish. Keep it to 3 lines.
    Athlete Data: Sleep={daily_data['Sleep']} hours, Training Intensity={daily_data['Training_Intensity']}/10,
    Previous Injury={'Yes' if daily_data['Previous_Injury']==1 else 'No'}, Muscle Soreness={daily_data['Muscle_Soreness']}/10.
    Calculated Risk Score: {risk_score}/100.
    Give 2-3 specific tips to reduce injury risk today."""

    try:
        response = client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
        return response.text
    except Exception as e:
        return f"Google AI Error: {e}. Set DEMO_MODE = True to use offline advice."

# ====== SESSION STATE ======
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.username = None
    st.session_state.page = "Login"

# ====== PAGE 1: LOGIN / SIGNUP ======
def login_page():
    st.title("🏃 Athlete Injury Risk Predictor")
    tab1, tab2 = st.tabs(["Login", "New User Signup"])

    users_df = load_users()

    with tab1:
        st.subheader("Login")
        username = st.text_input("Username", key="login_user")
        password = st.text_input("Password", type="password", key="login_pass")
        if st.button("Login", type="primary"):
            hashed = hash_password(password)
            user = users_df[(users_df['username'] == username) & (users_df['password'] == hashed)]
            if not user.empty:
                st.session_state.logged_in = True
                st.session_state.username = username
                st.session_state.user_data = user.iloc[0].to_dict()
                st.session_state.page = "Daily Check"
                st.rerun()
            else:
                st.error("Wrong username or password")

    with tab2:
        st.subheader("Create New Account")
        col1, col2 = st.columns(2)
        with col1:
            name = st.text_input("Full Name")
            username_new = st.text_input("Create Username")
            password_new = st.text_input("Create Password", type="password")
            age = st.number_input("Age", 12, 60, 20)
        with col2:
            gender = st.selectbox("Gender", ["Male", "Female", "Other"])
            weight = st.number_input("Weight (kg)", 30, 150, 70)
            height = st.number_input("Height (cm)", 120, 220, 175)
            sport = st.selectbox("Sport", ["Football", "Cricket", "Basketball", "Running", "Gym", "Other"])
            past_injuries = st.number_input("Past Injuries Count", 0, 10, 0)

        if st.button("Create Account"):
            if username_new in users_df['username'].values:
                st.error("Username already exists")
            else:
                new_user = pd.DataFrame([{
                    "username": username_new, "password": hash_password(password_new),
                    "name": name, "age": age, "gender": gender, "weight": weight,
                    "height": height, "sport": sport, "past_injuries": past_injuries
                }])
                users_df = pd.concat([users_df, new_user], ignore_index=True)
                save_user(users_df)
                st.success("Account created! Now Login.")

# ====== PAGE 2: DAILY RISK CHECK ======
def daily_check_page():
    st.title(f"Welcome, {st.session_state.user_data['name']} 🏃")
    st.write(f"Sport: {st.session_state.user_data['sport']} | Age: {st.session_state.user_data['age']}")

    if st.button("Logout"):
        st.session_state.logged_in = False
        st.session_state.username = None
        st.session_state.page = "Login"
        st.rerun()

    st.divider()
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Enter Today's Data")
        sleep = st.slider("Sleep Hours", 4.0, 10.0, 7.0, 0.5)
        intensity = st.slider("Training Intensity /10", 1, 10, 5)
        prev_injury_today = st.selectbox("New Injury Today?", ["No", "Yes"])
        soreness = st.slider("Muscle Soreness /10", 1, 10, 3)
        prev_injury_val = 1 if prev_injury_today == "Yes" else st.session_state.user_data['past_injuries']

    with col2:
        st.subheader("Prediction")
        if st.button("Predict My Risk", type="primary"):
            # Use profile past_injuries if no new injury today
            model_input_prev = 1 if st.session_state.user_data['past_injuries'] > 0 or prev_injury_today=="Yes" else 0

            input_data = [[sleep, intensity, model_input_prev, soreness]]
            prob = model.predict_proba(input_data)[0][1]
            risk_score = int(prob * 100)

            st.metric("Injury Risk Score", f"{risk_score}/100")
            st.progress(risk_score / 100)

            # Save record
            records_df = load_records()
            new_record = pd.DataFrame([{
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "username": st.session_state.username,
                "sleep": sleep, "intensity": intensity,
                "prev_injury": model_input_prev, "soreness": soreness,
                "risk_score": risk_score
            }])
            records_df = pd.concat([records_df, new_record], ignore_index=True)
            save_record(records_df)
            st.success("Record saved!")

            daily_data = {'Sleep': sleep, 'Training_Intensity': intensity, 'Previous_Injury': model_input_prev, 'Muscle_Soreness': soreness}
            with st.spinner("AI Doctor is thinking..."):
                advice = get_llm_advice(risk_score, daily_data)

            st.subheader("AI Doctor Advice")
            st.info(advice)

    st.divider()
    st.subheader("Your Past Records")
    records_df = load_records()
    user_records = records_df[records_df['username'] == st.session_state.username]
    st.dataframe(user_records.tail(10), use_container_width=True)

# ====== ROUTER ======
if st.session_state.page == "Login" or not st.session_state.logged_in:
    login_page()
else:
    daily_check_page()

st.caption("Note: This is for educational purpose only. Not medical advice.")