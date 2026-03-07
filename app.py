import streamlit as st
import os

# Set page configuration
st.set_page_config(
    page_title="RetailVision AI | Retail Analytics Platform",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sidebar Navigation
st.sidebar.title("RetailVision AI")
st.sidebar.image("assets/logo.png", use_container_width=True) if os.path.exists("assets/logo.png") else None
st.sidebar.markdown("---")

navigation = st.sidebar.radio(
    "Navigation",
    ["Overview", "Product Scanner", "Operations Analytics", "Model Intelligence", "Inventory Insights", "Reports", "System Info"]
)

# Display content based on navigation
if navigation == "Overview":
    from ui_pages.overview import show_overview
    show_overview()
elif navigation == "Product Scanner":
    from ui_pages.scanner import show_scanner
    show_scanner()
elif navigation == "Operations Analytics":
    from ui_pages.analytics import show_analytics
    show_analytics()
elif navigation == "Model Intelligence":
    from ui_pages.model_intelligence import show_model_intelligence
    show_model_intelligence()
elif navigation == "Inventory Insights":
    from ui_pages.insights import show_insights
    show_insights()
elif navigation == "Reports":
    from ui_pages.reports import show_reports
    show_reports()
elif navigation == "System Info":
    st.title("System Info")
    st.markdown("""
    ### Project Overview
    **RetailVision AI** is an advanced computer vision platform designed for supermarket operations. 
    It leverages Convolutional Neural Networks (CNN) to automate product recognition and provide 
    actionable inventory insights.

    ### AI Model Architecture
    - **Backbone**: CNN Classifier
    - **Training Framework**: TensorFlow/Keras
    - **Target Input Size**: 224x224 pixels
    - **Normalization**: Standard [0, 1] range

    ### Dataset Information
    The model is trained on the **Grocery Store Dataset**, featuring natural images of various 
    grocery products including fruits, vegetables, and packaged goods.

    ### Technology Stack
    - **UI Framework**: Streamlit
    - **Inference**: TensorFlow
    - **Image Processing**: OpenCV, Pillow
    - **Data Analytics**: Pandas, Plotly
    - **Model Serialization**: Joblib

    ### Business Value
    By automating product identification, RetailVision AI helps reduce human error in shelf 
    monitoring, optimizes restocking schedules, and provides data-driven category demand patterns.
    """)
