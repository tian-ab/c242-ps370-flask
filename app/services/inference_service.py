import numpy as np
import tensorflow as tf
import cv2
from PIL import Image
import io
from .exceptions import InputError

def preprocess_image(image_file, target_size=(105, 105)):
    """
    Preprocess image for face recognition model
    Args:
        image_file: File-like object from Flask
        target_size: Target image size for model input
    Returns:
        Preprocessed image array
    """
    # Read image
    img = Image.open(image_file)
    img = img.convert('RGB')
    
    # Resize and convert to numpy array
    img = img.resize(target_size)
    img_array = np.array(img, dtype=np.float32)
    
    # Normalize
    img_array = (img_array - 127.5) / 128.0
    
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

def predict_face_recognition(model, test_image, reference_image, threshold=0.7):
    """
    Perform face recognition
    Args:
        model: Keras h5 model
        test_image: Image to be recognized
        reference_image: Reference image for comparison
        threshold: Similarity threshold for recognition
    Returns:
        List of recognition results
    """
    try:
        # Preprocess test image
        test_img_tensor = preprocess_image(test_image)
        
        # Preprocess reference image
        ref_img_tensor = preprocess_image(reference_image)
        
        # Set input tensors
        input_details = model.get_input_details()
        output_details = model.get_output_details()
        
        model.set_tensor(input_details[0]['index'], test_img_tensor)
        model.set_tensor(input_details[1]['index'], ref_img_tensor)
        model.invoke()
        
        # Get embeddings
        test_embedding = model.get_tensor(output_details[0]['index']).flatten()
        ref_embedding = model.get_tensor(output_details[1]['index']).flatten()
        
        # Calculate cosine similarity
        similarity = np.dot(test_embedding, ref_embedding) / (
            np.linalg.norm(test_embedding) * np.linalg.norm(ref_embedding)
        )
        
        # Prepare results
        results = {
            "similarity": float(similarity),
            "matched": similarity > threshold
        }
        
        return results
    
    except Exception as e:
        raise InputError(f"Face recognition error: {str(e)}")