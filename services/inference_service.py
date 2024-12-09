import tensorflow as tf
from app.exceptions import InputError
from services.food_list import food_list

def predict_food_recommendations(model, input_json):
    try:
        user_allergens = input_json["user_allergens"]
        user_last_order = input_json["user_last_order"]
        food_category = input_json["food_category"]
        food_ingredients = input_json["food_ingredients"]

        input_data = [user_allergens, user_last_order, food_category, food_ingredients]
        tensor_input = tf.convert_to_tensor([input_data])

        predictions = model(tensor_input).numpy()
        confidence_scores = predictions.flatten()

        recommended_foods = [
            {"food": food["name"], "confidence": score * 100}
            for food, score in zip(food_list, confidence_scores)
        ]
        recommended_foods.sort(key=lambda x: x["confidence"], reverse=True)

        return [food["food"] for food in recommended_foods]
    except Exception:
        raise InputError("An error occurred during prediction.")
