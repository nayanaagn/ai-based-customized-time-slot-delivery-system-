import streamlit as st
import pandas as pd
import random
from sklearn.tree import DecisionTreeClassifier

st.set_page_config(page_title="AI Time Slot Delivery", layout="centered")
st.title("📦 AI-Based Smart Time Slot Delivery System")

# -------------------------------
# Load or Create Historical Data
# -------------------------------
try:
    history = pd.read_csv("delivery_history.csv")
except:
    history = pd.DataFrame(columns=["Customer", "Preferred", "Slot", "Delivered"])

# -------------------------------
# Train Simple ML Model
# -------------------------------
if len(history) >= 5:
    X = history[["Preferred"]].replace(
        {"Morning": 0, "Afternoon": 1, "Evening": 2}
    )
    y = history["Delivered"]
    model = DecisionTreeClassifier()
    model.fit(X, y)
else:
    model = None

# -------------------------------
# Customer Input
# -------------------------------
st.header("Customer Interaction (AI Assisted)")

name = st.text_input("Customer Name")
address = st.text_area("Delivery Address")

preferred_time = st.selectbox(
    "When are you usually available?",
    ["Morning", "Afternoon", "Evening"]
)

# -------------------------------
# AI Recommendation Engine
# -------------------------------
def recommend_slot(pref):
    if pref == "Morning":
        return "10 AM – 12 PM"
    elif pref == "Afternoon":
        return "2 PM – 4 PM"
    else:
        return "5 PM – 7 PM"

if st.button("🤖 AI Suggest Delivery Slot"):
    slot = recommend_slot(preferred_time)

    if model:
        prob = model.predict_proba([[["Morning", "Afternoon", "Evening"].index(preferred_time)]])[0][1]
        st.info(f"AI Confidence: {round(prob*100,2)}% availability predicted")
    else:
        st.info("AI Confidence: Learning from limited data")

    st.success(f"Recommended Slot: **{slot}**")

    decision = st.radio("Confirm delivery slot?", ["Confirm", "Reschedule"])

    if decision == "Confirm":
        new_entry = {
            "Customer": name,
            "Preferred": preferred_time,
            "Slot": slot,
            "Delivered": 1
        }
        history = pd.concat([history, pd.DataFrame([new_entry])])
        history.to_csv("delivery_history.csv", index=False)
        st.success("✅ Slot Confirmed & Saved")

    if decision == "Reschedule":
        alt_slot = random.choice(
            ["9 AM – 11 AM", "12 PM – 2 PM", "6 PM – 8 PM"]
        )
        st.warning(f"Alternative AI Slot: {alt_slot}")

# -------------------------------
# Delivery Dashboard
# -------------------------------
st.header("🚚 Delivery Management Dashboard")
if len(history) > 0:
    st.dataframe(history)
else:
    st.write("No delivery data available yet.")