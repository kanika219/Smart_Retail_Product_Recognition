import cv2
import numpy as np
from PIL import Image

def preprocess_image(image: Image.Image, target_size=(224, 224)):
    """
    Preprocess image for CNN inference.
    - Resize to target_size (default 224x224)
    - Normalize pixel values to [0, 1]
    - Expand dimensions for batch
    """
    img_array = np.array(image.convert("RGB"))
    img_resized = cv2.resize(img_array, target_size)
    img_normalized = img_resized.astype("float32") / 255.0
    img_batch = np.expand_dims(img_normalized, axis=0)
    return img_batch
