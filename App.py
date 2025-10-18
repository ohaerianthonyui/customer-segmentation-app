
import streamlit as st
import pickle
import numpy as np

# Load the trained model
kmeans = pickle.load(open("kmeans.pkl", 'rb'))

# Cluster labels
cluster_labels = {
    0: "Low Income – Low Spending",
    1: "High Income – High Spending",
    2: "Young Low Income – High Spending"
}

# App title and header
st.set_page_config(page_title="Customer Segment Predictor", layout="centered")
st.title("🧠 Customer Segmentation with KMeans")
st.markdown("Use the sliders below to predict the **customer segment** based on gender, age, income, and spending score.")

# Sidebar for inputs
st.sidebar.header("🔍 Input Customer Details")

gender = st.sidebar.radio("Gender", ("Male", "Female"))
gender_value = 1 if gender == "Male" else 0

age = st.sidebar.slider("Age", 18, 70, 30)
income = st.sidebar.slider("Annual Income (k$)", 10, 150, 50)
score = st.sidebar.slider("Spending Score (1–100)", 1, 100, 50)

# Predict segment
if st.sidebar.button("Predict Segment"):
    input_data = np.array([[gender_value, age, income, score]])
    cluster = kmeans.predict(input_data)[0]
    segment = cluster_labels.get(cluster, "Unknown Segment")

    st.success(f"🎯 Predicted Segment: **{segment}**")
    st.info(f"🧩 Cluster ID: {cluster}")

    # Optional: Add colored segment box
    color_map = {
        0: "#FFDDC1",
        1: "#D5ECC2",
        2: "#B5EAEA"
    }
    st.markdown(
        f"<div style='background-color:{color_map[cluster]};padding:20px;border-radius:10px;'>"
        f"<h4 style='color:#333;text-align:center;'>Segment: {segment}</h4>"
        f"</div>", unsafe_allow_html=True
    )

# Footer
st.markdown("---")
st.caption("Built with ❤️ using Streamlit | KMeans Customer Segmentation")
