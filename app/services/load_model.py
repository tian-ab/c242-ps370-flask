import os
import tensorflow as tf
import requests

def load_model(model_url_or_path):
    # Check if the model_url_or_path is a URL (Google Cloud Storage)
    if model_url_or_path.startswith("https://"):
        # Download the model from GCS if it's a URL
        response = requests.get(model_url_or_path)
        if response.status_code == 200:
            model_content = response.content
        else:
            raise Exception(f"Failed to download model from {model_url_or_path}. Status code: {response.status_code}")
        
        # Save the downloaded model temporarily
        model_path = "/tmp/temp_model.tflite"  # Temporary path to save the model
        with open(model_path, "wb") as f:
            f.write(model_content)
    else:
        # If it's a local path, use the model from the local directory
        model_path = model_url_or_path
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")
    
    # Load the model into the TensorFlow Lite interpreter
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter
