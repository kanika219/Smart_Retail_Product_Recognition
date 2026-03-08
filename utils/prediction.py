import os
import joblib
import streamlit as st
import numpy as np
import onnxruntime as ort

@st.cache_resource
def load_model_and_encoder():
    """
    Load ONNX model and label encoder.
    Uses st.cache_resource to prevent repeated loading.
    """
    model_path = "model/grocery_model.onnx"
    encoder_path = "model/label_encoder.pkl"

    if not os.path.exists(model_path) or not os.path.exists(encoder_path):
        return None, None

    # Load ONNX InferenceSession
    session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
    
    # Load Label Encoder
    label_encoder = joblib.load(encoder_path)

    return session, label_encoder


def get_predictions(model_session, label_encoder, preprocessed_img):
    """
    Run inference using ONNX session.
    Returns Top-5 categories with probabilities.
    """
    # Get input/output names from the session
    input_name = model_session.get_inputs()[0].name
    output_name = model_session.get_outputs()[0].name

    # Run Inference
    preds = model_session.run([output_name], {input_name: preprocessed_img.astype('float32')})[0][0]

    # Get Top-5 predictions
    top_5_indices = preds.argsort()[-5:][::-1]
    top_5_categories = [label_encoder.inverse_transform([i])[0] for i in top_5_indices]
    top_5_probabilities = [float(preds[i]) for i in top_5_indices]

    return list(zip(top_5_categories, top_5_probabilities))
