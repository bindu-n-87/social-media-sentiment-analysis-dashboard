import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt

from src.preprocess import clean_text

# -----------------------------
# Load model + vectorizer
# -----------------------------
model = pickle.load(open("models/model.pkl", "rb"))
vectorizer = pickle.load(open("models/vectorizer.pkl", "rb"))

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(page_title="Sentiment Dashboard", layout="wide")

st.title("Social Media Sentiment Analysis Dashboard")

# -----------------------------
# Load dataset
# -----------------------------
df = pd.read_csv("data/social_data.csv")

# -----------------------------
# Prediction Section
# -----------------------------
st.header("Sentiment Prediction")

col1, col2 = st.columns([2, 1])

with col1:
    user_input = st.text_area("Enter text for analysis")

    if st.button("Predict"):
        if user_input.strip():
            cleaned = clean_text(user_input)
            vector = vectorizer.transform([cleaned])
            prediction = model.predict(vector)[0]

            st.subheader("Prediction Result")

            if prediction == "positive":
                st.success("Positive Sentiment")
            elif prediction == "negative":
                st.error("Negative Sentiment")
            else:
                st.info("Neutral Sentiment")

        else:
            st.warning("Please enter text")

# -----------------------------
# Dataset Insights
# -----------------------------
with col2:
    st.subheader("Dataset Summary")

    st.write("Total Records:", len(df))
    st.write(df["sentiment"].value_counts())

# -----------------------------
# Visualization Section
# -----------------------------
st.header("Sentiment Analysis Insights")

col3, col4 = st.columns(2)

# Bar Chart
with col3:
    st.subheader("Sentiment Distribution (Bar)")
    fig1, ax1 = plt.subplots()
    df["sentiment"].value_counts().plot(kind="bar", ax=ax1)
    st.pyplot(fig1)

# Pie Chart (IMPORTANT UPGRADE)
with col4:
    st.subheader("Sentiment Distribution (Pie)")
    fig2, ax2 = plt.subplots()
    df["sentiment"].value_counts().plot(
        kind="pie",
        autopct="%1.1f%%",
        ax=ax2
    )
    ax2.set_ylabel("")
    st.pyplot(fig2)

# -----------------------------
# Footer Insight
# -----------------------------
st.markdown("---")
st.markdown("NLP + Machine Learning based Sentiment Analysis System")