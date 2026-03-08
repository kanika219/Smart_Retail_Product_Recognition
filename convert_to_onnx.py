import tensorflow as tf
import tf2onnx
import onnx
import os

def convert_h5_to_onnx(h5_path, onnx_path):
    """
    Utility script to convert a Keras .h5 model to ONNX format.
    Run this locally where TensorFlow is installed.
    """
    if not os.path.exists(h5_path):
        print(f"Error: {h5_path} not found.")
        return

    print(f"Loading Keras model from {h5_path}...")
    model = tf.keras.models.load_model(h5_path)

    print("Converting to ONNX...")
    spec = (tf.TensorSpec((None, 224, 224, 3), tf.float32, name="input"),)
    
    model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=spec, opset=13)
    
    with open(onnx_path, "wb") as f:
        f.write(model_proto.SerializeToString())
    
    print(f"✓ Model successfully converted and saved to {onnx_path}")

if __name__ == "__main__":
    # Ensure the model directory exists
    os.makedirs("model", exist_ok=True)
    
    # Try to convert .h5 if it exists
    h5_file = "model/grocery_model.h5"
    onnx_file = "model/grocery_model.onnx"
    
    convert_h5_to_onnx(h5_file, onnx_file)
