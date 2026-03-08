import streamlit as st
from PIL import Image
import time
from utils.image_processing import preprocess_image
from utils.prediction import load_model_and_encoder, get_predictions
import plotly.express as px
import pandas as pd

def show_scanner():
    st.title("🛡️ Product Scanner Console")
    st.markdown("### AI-Powered Product Recognition System")

    # Load Model and Encoder
    model, label_encoder = load_model_and_encoder()
    
    if model is None or label_encoder is None:
        st.warning("⚠️ Model files (`grocery_model.onnx`, `label_encoder.pkl`) not found in the `model/` directory.")
        st.info("Please ensure your trained model and label encoder are uploaded to continue.")
        return

    # Upload Section
    st.markdown("---")
    uploaded_file = st.file_uploader("📷 Upload Product Image", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Layout: Image Preview and Prediction Summary
        col1, col2 = st.columns([1, 1.2])
        
        with col1:
            st.markdown("#### Image Preview")
            image = Image.open(uploaded_file)
            st.image(image, use_container_width=True)

        with col2:
            st.markdown("#### Recognition Analysis")
            
            # Start timer for latency measurement
            start_time = time.time()
            
            # Preprocess and Predict
            preprocessed_img = preprocess_image(image)
            predictions = get_predictions(model, label_encoder, preprocessed_img)
            
            # Latency calculation
            end_time = time.time()
            latency = (end_time - start_time) * 1000 # convert to ms
            
            # Main Prediction Result
            top_category, top_confidence = predictions[0]
            
            # Display Key Metrics
            m_col1, m_col2 = st.columns(2)
            with m_col1:
                st.metric("Predicted Category", top_category)
            with m_col2:
                st.metric("Inference Latency", f"{latency:.0f}ms")
            
            st.progress(top_confidence, text=f"Confidence Score: {top_confidence*100:.1f}%")
            
            st.markdown("---")
            
            # Top 5 Probability Distribution
            st.markdown("#### Top 5 Classification Probabilities")
            df_preds = pd.DataFrame(predictions, columns=["Category", "Probability"])
            fig = px.bar(df_preds, x="Probability", y="Category", orientation='h',
                         color='Probability', color_continuous_scale='Blues',
                         text_auto='.1%')
            fig.update_layout(yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig, use_container_width=True)

    else:
        # Placeholder view
        st.markdown("""
        #### Operational Workflow:
        1. **Capture**: Take a clear photo of the grocery item.
        2. **Upload**: Use the file uploader above to submit the image.
        3. **Process**: The AI engine will analyze the image for product features.
        4. **Analyze**: Review the classification results and confidence metrics.
        """)
        st.image("https://images.unsplash.com/photo-1542838132-92c53300491e?auto=format&fit=crop&q=80&w=1074", use_container_width=True, caption="Sample Supermarket Shelf")
