import streamlit as st
import requests
import tempfile
import os

st.set_page_config(page_title="Kupids Classifier", layout="wide")
st.title("Kupids Classifier: App Review Sentiment Analysis")
st.markdown("""
Analyze Google Play app reviews for sentiment, compare models, and get a natural language explanation from Gemini.
""")

# Sidebar for input
tab1, tab2 = st.tabs(["Analyze by App ID", "Upload CSV"])

with tab1:
    app_id = st.text_input("Enter Google Play App ID (e.g., com.tinder)")
    analyze_btn = st.button("Analyze App Reviews", key="analyze_app")

with tab2:
    csv_file = st.file_uploader("Upload a CSV of reviews", type=["csv"])
    analyze_csv_btn = st.button("Analyze Uploaded CSV", key="analyze_csv")

# Helper to call FastAPI backend
def call_backend(app_id=None, csv_file=None):
    url = "http://localhost:8000/analyze"
    if csv_file:
        # Save uploaded file to a temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
            tmp.write(csv_file.read())
            tmp_path = tmp.name
        data = {"app_id": app_id or "", "csv_path": tmp_path}
    else:
        data = {"app_id": app_id}
    response = requests.post(url, json=data)
    if csv_file:
        os.remove(tmp_path)
    return response

# Main logic
if (analyze_btn and app_id) or (analyze_csv_btn and csv_file):
    with st.spinner("Running analysis..."):
        if analyze_btn:
            response = call_backend(app_id=app_id)
        else:
            response = call_backend(app_id="", csv_file=csv_file)
        if response.ok:
            result = response.json()
            st.subheader("Exploratory Data Analysis (EDA)")
            col1, col2 = st.columns(2)
            with col1:
                st.image(result["eda"]["sentiment_distribution_plot"], caption="Sentiment Distribution", use_column_width=True)
            with col2:
                st.image(result["eda"]["wordcloud_plot"], caption="Word Cloud", use_column_width=True)
            st.markdown("#### Summary Statistics")
            st.json(result["eda"]["summary_stats"])
            st.subheader("Model Comparison")
            st.dataframe(result["model_comparison"], use_container_width=True)
            st.success(f"Best Model: {result['best_model']}")
            st.subheader("Gemini Explanation")
            st.write(result["gemini_explanation"])
        else:
            st.error(f"Analysis failed: {response.text}") 