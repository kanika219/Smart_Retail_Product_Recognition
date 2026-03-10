import cv2
import numpy as np
from PIL import Image, ImageOps

def preprocess_image(image: Image.Image, target_size=(224, 224)):
    """
    Preprocess image for CNN inference without TensorFlow.
    - Handle EXIF orientation
    - Resize to target_size (224x224)
    - Apply standard [0, 1] normalization
    - Expand dimensions for batch
    """
    # Fix image orientation from EXIF
    image = ImageOps.exif_transpose(image)
    
    # Convert PIL to RGB and then to numpy array
    img_array = np.array(image.convert("RGB"))
    
    # Resize using OpenCV
    img_resized = cv2.resize(img_array, target_size)
    
    # Standard normalization: Scales pixels from [0, 255] to [0, 1]
    # As specified in app.py System Info
    img_preprocessed = img_resized.astype("float32") / 255.0
    
    # Expand dimensions to (1, 224, 224, 3)
    img_batch = np.expand_dims(img_preprocessed, axis=0)
    
    return img_batch
