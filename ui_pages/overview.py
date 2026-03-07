import streamlit as st
import plotly.express as px
import pandas as pd
from utils.metrics_loader import load_model_metrics

def show_overview():
    st.title("🛒 Executive Overview Dashboard")
    st.markdown("### AI-Powered Retail Analytics Summary")

    metrics = load_model_metrics()
    
    # KPI Metric Cards
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Model Accuracy", f"{metrics['accuracy']*100:.1f}%", "+0.5%")
    with col2:
        st.metric("Product Categories", "81", "Active")
    with col3:
        st.metric("Total AI Scans", "12,450", "+120 today")
    with col4:
        st.metric("Avg Latency", "145ms", "-10ms")

    st.markdown("---")

    # Interactive Plotly Charts
    chart_col1, chart_col2 = st.columns(2)
    
    # Simulated Daily Scan Volume (Time Series)
    days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    scans = [1200, 1500, 1100, 1800, 2200, 2500, 2100]
    df_scans = pd.DataFrame({"Day": days, "Scans": scans})
    
    with chart_col1:
        st.markdown("#### Simulated Daily Scan Volume")
        fig_line = px.line(df_scans, x="Day", y="Scans", markers=True, 
                           color_discrete_sequence=["#007bff"])
        st.plotly_chart(fig_line, use_container_width=True)

    # Category Detection Frequency (Bar Chart)
    categories = ["Fruit", "Vegetable", "Packages", "Refrigerated"]
    counts = [450, 320, 210, 150]
    df_cat = pd.DataFrame({"Category": categories, "Count": counts})
    
    with chart_col2:
        st.markdown("#### Category Detection Frequency")
        fig_bar = px.bar(df_cat, x="Category", y="Count", 
                         color="Category", color_discrete_sequence=px.colors.qualitative.Safe)
        st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown("---")
    
    # System Summary
    st.markdown("""
    #### Platform Role in Operations
    **RetailVision AI** serves as a decision-support tool for store managers and operations teams. 
    By leveraging computer vision, the platform provides:
    - **Real-time Product Recognition**: Identifying items on shelves via image analysis.
    - **Category Demand Tracking**: Monitoring which product groups are most scanned.
    - **Operational Insights**: Generating data-driven recommendations for inventory management.
    """)
