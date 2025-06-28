import streamlit as st
from detect import run_detection
from dashboard import run_dashboard

st.set_page_config(page_title="Retail Store Analytics", layout="wide")
st.title("🛒 Retail Analytics using Object Detection - YOLOv8")

# Use st.tabs to create tabs
tab1, tab2 = st.tabs(["📸 Object Detection", "📊 Analytics Dashboard"])

with tab1:
    run_detection()

with tab2:
    run_dashboard()
