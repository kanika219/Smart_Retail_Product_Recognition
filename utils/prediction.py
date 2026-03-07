import os
import joblib
import streamlit as st
import numpy as np
try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    import tensorflow.lite as tflite

@st.cache_resource
def load_model_and_encoder():
    model_path = "model/grocery_model.tflite"
    encoder_path = "model/label_encoder.pkl"

    if not os.path.exists(model_path) or not os.path.exists(encoder_path):
        return None, None

    interpreter = tflite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    label_encoder = joblib.load(encoder_path)

    return (interpreter, input_details, output_details), label_encoder


def get_predictions(model, label_encoder, preprocessed_img):
    interpreter, input_details, output_details = model

    interpreter.set_tensor(input_details[0]['index'], preprocessed_img.astype('float32'))
    interpreter.invoke()

    preds = interpreter.get_tensor(output_details[0]['index'])[0]

    top_5_indices = preds.argsort()[-5:][::-1]

    top_5_categories = [label_encoder.inverse_transform([i])[0] for i in top_5_indices]
    top_5_probabilities = [float(preds[i]) for i in top_5_indices]

    return list(zip(top_5_categories, top_5_probabilities))