import streamlit as st
import plotly.express as px
import pandas as pd
from utils.metrics_loader import get_category_stats

def show_analytics():
    st.title("📊 Operations Analytics Dashboard")
    st.markdown("### Category Recognition & Demand Analysis")

    # Metrics Section (Simulated Statistics)
    df_stats = get_category_stats()
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Top Category", df_stats.loc[0, "Category"])
    with col2:
        st.metric("Total Scans Monitored", df_stats["Predictions Count"].sum())
    with col3:
        st.metric("Avg Recognition Confidence", f"{df_stats['Confidence Average'].mean()*100:.1f}%")

    st.markdown("---")

    # Data Visualization
    col_chart1, col_chart2 = st.columns(2)

    with col_chart1:
        st.markdown("#### Recognition Distribution by Category")
        fig_pie = px.pie(df_stats, values="Predictions Count", names="Category", 
                         hole=0.4, color_discrete_sequence=px.colors.sequential.RdBu)
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_chart2:
        # Simulated Demand Patterns (Heatmap or Bar)
        st.markdown("#### Predicted Product Frequency")
        fig_bar = px.bar(df_stats, x="Category", y="Predictions Count", 
                         color="Confidence Average", 
                         labels={"Confidence Average": "Avg Confidence"},
                         color_continuous_scale="Viridis")
        st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown("---")

    # Tabular Data
    st.markdown("#### Product Category Statistics (Last 30 Days)")
    st.dataframe(df_stats.set_index("Category"), use_container_width=True)

    # Simulated Demand Pattern (Time series for specific category)
    st.markdown("---")
    st.markdown("#### Simulated Hourly Demand Patterns")
    hours = [f"{h:02d}:00" for h in range(8, 23)]
    demand = [15, 20, 45, 60, 55, 40, 35, 30, 70, 95, 110, 85, 45, 30, 20]
    df_hourly = pd.DataFrame({"Hour": hours, "Simulated Scans": demand})
    
    fig_area = px.area(df_hourly, x="Hour", y="Simulated Scans", 
                       title="Aggregated Store Traffic Pattern (AI Scans)",
                       color_discrete_sequence=["#ff7f0e"])
    st.plotly_chart(fig_area, use_container_width=True)
