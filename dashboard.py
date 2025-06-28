import streamlit as st
import pandas as pd
import os

def run_dashboard():
    # Reset button
    if st.button("🔄 Reset Analytics Data"):
        if os.path.exists("analytics/footfall.csv"):
            os.remove("analytics/footfall.csv")
            st.success("Footfall data reset! Run detection again to generate new analytics.")
            st.stop()

    st.subheader("📊 Retail Store Footfall Analytics")

    path = "analytics/footfall.csv"

    # Check if data exists
    if not os.path.exists(path):
        st.info("No analytics available yet. Run a video detection to generate data.")
        return

    # Load and parse timestamp column
    df = pd.read_csv(path, parse_dates=["timestamp"])

    # Calculate total visitors
    total_visitors = df["person_count"].sum()

    # Peak timestamp (most crowded moment)
    peak_row = df[df["person_count"] == df["person_count"].max()].iloc[0]
    peak_time = peak_row["timestamp"]
    peak_count = peak_row["person_count"]

    # Extract hour from timestamp for hourly stats
    df["hour"] = df["timestamp"].dt.hour
    hourly_avg = df.groupby("hour")["person_count"].mean()

    # --- Display KPIs ---
    col1, col2 = st.columns(2)
    with col1:
        st.metric("👥 Total Visitors", total_visitors)
    with col2:
        st.metric("⏰ Peak Time", f"{peak_time.strftime('%Y-%m-%d %H:%M:%S')} ({peak_count} visitors)")

    # --- Line Chart: Footfall Over Time ---
    st.subheader("📈 Footfall Trend Over Time")
    st.line_chart(df.set_index("timestamp")["person_count"])

    # --- Bar Chart: Hourly Average ---
    st.subheader("⏱️ Average Footfall Per Hour")
    st.bar_chart(hourly_avg)

    # --- Raw Data Table ---
    st.subheader("📋 Raw Detection Data")
    st.dataframe(df)

    # --- Download CSV ---
    st.download_button(
        label="📥 Download Footfall CSV",
        data=df.to_csv(index=False),
        file_name="footfall.csv",
        mime="text/csv"
    )
