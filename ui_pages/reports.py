import streamlit as st
import pandas as pd
from datetime import datetime
from utils.metrics_loader import get_category_stats, load_model_metrics

def show_reports():
    st.title("📄 Operational Reporting & Exports")
    st.markdown("### Retail Operations Data Center")

    # Generate Dummy Data for Export
    df_stats = get_category_stats()
    metrics = load_model_metrics()
    
    # Export Section
    st.markdown("#### Available Reports")
    
    # Prediction Logs
    st.markdown("---")
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("##### 🛒 Category Recognition Logs")
        st.markdown("Comprehensive log of all AI product recognitions for the current period.")
    
    with col2:
        csv_stats = df_stats.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Prediction Logs (CSV)",
            data=csv_stats,
            file_name=f'prediction_logs_{datetime.now().strftime("%Y%m%d")}.csv',
            mime='text/csv',
            key="download_prediction_logs"
        )

    # Model Performance Summary
    st.markdown("---")
    col3, col4 = st.columns([2, 1])
    
    with col3:
        st.markdown("##### 📈 Model Reliability Summary")
        st.markdown("Technical report summarizing model performance, accuracy, and latency metrics.")
    
    with col4:
        # Create a dataframe from metrics
        df_perf = pd.DataFrame([{
            "Metric": "Accuracy",
            "Value": metrics["accuracy"]
        }, {
            "Metric": "Precision",
            "Value": metrics["precision"]
        }, {
            "Metric": "Recall",
            "Value": metrics["recall"]
        }, {
            "Metric": "F1 Score",
            "Value": metrics["f1_score"]
        }])
        
        csv_perf = df_perf.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Model Performance (CSV)",
            data=csv_perf,
            file_name=f'model_performance_{datetime.now().strftime("%Y%m%d")}.csv',
            mime='text/csv',
            key="download_model_perf"
        )

    # Data Quality Indicators
    st.markdown("---")
    st.markdown("#### Data Export Configuration")
    
    with st.expander("Show Preview of Export Data"):
        st.markdown("##### Category Prediction Logs")
        st.dataframe(df_stats, use_container_width=True)
        
        st.markdown("##### Performance Summary")
        st.dataframe(df_perf, use_container_width=True)
    
    st.markdown("""
    **Note**: Exported reports follow the standard ISO-8601 date naming convention. 
    Data is generated based on the current operational session.
    """)
