import cv2
import numpy as np
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

def preprocess_image(image: Image.Image, target_size=(224, 224)):
    """
    Preprocess image for MobileNetV2 inference.
    - Resize to target_size (224x224)
    - Apply MobileNetV2 specific preprocessing (scaling to [-1, 1])
    - Expand dimensions for batch
    """
    # Convert PIL to RGB and then to numpy array
    img_array = np.array(image.convert("RGB"))
    
    # Resize using OpenCV
    img_resized = cv2.resize(img_array, target_size)
    
    # MobileNetV2 preprocessing: Scales pixels to [-1, 1]
    # This matches the training script: applications.mobilenet_v2.preprocess_input
    img_preprocessed = preprocess_input(img_resized.astype("float32"))
    
    # Expand dimensions to (1, 224, 224, 3)
    img_batch = np.expand_dims(img_preprocessed, axis=0)
    
    return img_batch
