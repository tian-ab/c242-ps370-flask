import uuid
import logging 
import base64
import numpy as np
from datetime import datetime
from flask import request, jsonify, current_app
from services.inference_service import predict_face_recognition
from services.store_data import store_data
from services.get_all_data import get_all_data

logging.basicConfig(level=logging.DEBUG)

def post_predict_handler():
    try:
        # Handle test and reference image inputs
        if 'test_image' not in request.files or 'reference_image' not in request.files:
            return jsonify({"status": "error", "message": "Both test and reference images are required"}), 400
        
        test_image = request.files['test_image']
        reference_image = request.files['reference_image']
        
        # Optional: Additional parameters can be sent as form data
        threshold = float(request.form.get('threshold', 0.7))
        
        # Load model from app configuration
        model = current_app.config["model"]
        logging.debug("Model loaded: %s", model)
        
        # Process and predict
        results = predict_face_recognition(
            model, 
            test_image, 
            reference_image, 
            threshold
        )
        
        # Generate unique record
        record_id = str(uuid.uuid4())
        created_at = datetime.utcnow().isoformat()
        
        # Prepare data for storage
        data = {
            "id": record_id,
            "results": results,
            "createdAt": created_at
        }
        
        # Store results
        store_data(record_id, data)
        logging.debug("Face recognition results stored.")
        
        # Prepare response
        response = {
            "status": "success",
            "message": "Face recognition completed",
            "data": data
        }
        
        return jsonify(response), 201
    
    except Exception as e:
        logging.error("Error in face recognition: %s", e)
        return jsonify({"status": "error", "message": str(e)}), 500

def get_predict_histories_handler():
    # Similar to previous implementation
    all_data = get_all_data()
    formatted_data = [
        {
            "id": doc.id,
            "history": {
                "results": doc.to_dict().get("results"),
                "createdAt": doc.to_dict().get("createdAt"),
                "id": doc.id
            }
        }
        for doc in all_data
    ]
    response = {"status": "success", "data": formatted_data}
    return jsonify(response), 200