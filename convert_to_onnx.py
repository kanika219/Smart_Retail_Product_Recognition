import os
import subprocess

def convert_to_onnx_cli(model_path, onnx_path):
    """
    Convert model using tf2onnx CLI.
    """
    print(f"Converting {model_path} to {onnx_path} using CLI...")
    
    # Simpler command construction
    if os.path.exists(model_path):
        cmd = f"python -m tf2onnx.convert --keras \"{model_path}\" --output \"{onnx_path}\" --opset 13"
        print(f"Running: {cmd}")
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"Successfully converted to {onnx_path}")
        else:
            print("Conversion failed!")
            print(result.stderr)
            print(result.stdout)
    else:
        print(f"Source {model_path} not found.")

if __name__ == "__main__":
    # We'll use the .keras file created by create_model.py
    keras_file = "model/grocery_model.keras"
    onnx_file = "model/grocery_model.onnx"
    
    convert_to_onnx_cli(keras_file, onnx_file)
