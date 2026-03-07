import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
import joblib
import streamlit as st

@st.cache_resource
def load_model_and_encoder():
    """
    Load the saved model and label encoder from the model directory.
    Uses @st.cache_resource to load them only once.
    """
    model_path = "model/grocery_model.h5"
    encoder_path = "model/label_encoder.pkl"
    
    if not os.path.exists(model_path) or not os.path.exists(encoder_path):
        # We handle the case where the user has not uploaded the model yet
        # Returning None to let the UI display an error
        return None, None
    
    model = tf.keras.models.load_model(model_path)
    label_encoder = joblib.load(encoder_path)
    return model, label_encoder

def get_predictions(model, label_encoder, preprocessed_img):
    """
    Perform CNN inference and return top-5 categories.
    """
    preds = model.predict(preprocessed_img)[0]
    top_5_indices = preds.argsort()[-5:][::-1]
    
    top_5_categories = [label_encoder.inverse_transform([i])[0] for i in top_5_indices]
    top_5_probabilities = [float(preds[i]) for i in top_5_indices]
    
    return list(zip(top_5_categories, top_5_probabilities))
