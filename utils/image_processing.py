import cv2
import numpy as np
from PIL import Image

def preprocess_image(image: Image.Image, target_size=(224, 224)):
    """
    Preprocess image for MobileNetV2 inference without TensorFlow.
    - Resize to target_size (224x224)
    - Apply MobileNetV2 specific preprocessing (scaling to [-1, 1])
    - Expand dimensions for batch
    """
    # Convert PIL to RGB and then to numpy array
    img_array = np.array(image.convert("RGB"))
    
    # Resize using OpenCV
    img_resized = cv2.resize(img_array, target_size)
    
    # MobileNetV2 preprocessing (TF style): Scales pixels from [0, 255] to [-1, 1]
    # Formula: (x / 127.5) - 1.0
    img_preprocessed = (img_resized.astype("float32") / 127.5) - 1.0
    
    # Expand dimensions to (1, 224, 224, 3)
    img_batch = np.expand_dims(img_preprocessed, axis=0)
    
    return img_batch
