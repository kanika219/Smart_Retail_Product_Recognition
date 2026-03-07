import json
import os
import pandas as pd

def load_model_metrics(file_path="model/model_metrics.json"):
    """
    Load model performance metrics from a JSON file.
    Returns a dictionary of metrics.
    """
    if not os.path.exists(file_path):
        # Default placeholder metrics for demonstration if file doesn't exist
        return {
            "accuracy": 0.85,
            "precision": 0.84,
            "recall": 0.83,
            "f1_score": 0.84,
            "training_history": {
                "epochs": [1, 2, 3, 4, 5],
                "train_acc": [0.6, 0.7, 0.8, 0.83, 0.85],
                "val_acc": [0.55, 0.65, 0.75, 0.8, 0.82]
            }
        }
    
    with open(file_path, "r") as f:
        return json.load(f)

def get_category_stats():
    """
    Generate simulated category statistics for the Analytics page.
    """
    data = {
        "Category": ["Fruit", "Vegetable", "Packages", "Refrigerated"],
        "Predictions Count": [450, 320, 210, 150],
        "Confidence Average": [0.92, 0.88, 0.85, 0.89]
    }
    return pd.DataFrame(data)
