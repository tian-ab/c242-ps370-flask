import uuid
from datetime import datetime
from flask import request, jsonify, current_app
from services.inference_service import predict_food_recommendations
from services.store_data import store_data
from services.get_all_data import get_all_data

def post_predict_handler():
    payload = request.get_json()
    model = current_app.config["model"]

    allergens = payload.get("allergens")
    ingredients = payload.get("ingredients")
    last_order = payload.get("lastOrder")
    category = payload.get("category")

    recommendations = predict_food_recommendations(model, {
        "user_allergens": allergens,
        "user_last_order": last_order,
        "food_category": category,
        "food_ingredients": ingredients
    })

    record_id = str(uuid.uuid4())
    created_at = datetime.utcnow().isoformat()

    data = {
        "id": record_id,
        "recommendations": recommendations,
        "createdAt": created_at
    }

    store_data(record_id, data)

    response = {
        "status": "success",
        "message": "Menu recommendations generated successfully",
        "data": data
    }
    return jsonify(response), 201

def get_predict_histories_handler():
    all_data = get_all_data()

    formatted_data = [
        {
            "id": doc.id,
            "history": {
                "recommendations": doc.to_dict().get("recommendations"),
                "createdAt": doc.to_dict().get("createdAt"),
                "id": doc.id
            }
        }
        for doc in all_data
    ]

    response = {"status": "success", "data": formatted_data}
    return jsonify(response), 200
