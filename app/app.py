import os
from flask import Flask, jsonify
from werkzeug.exceptions import HTTPException
from services.load_model import load_model
from exceptions import InputError
#from services.food_list import food_list # testing food list

# Initialize Flask app
app = Flask(__name__)

# Load model (local or cloud)
MODEL_URL_OR_PATH = os.getenv("https://storage.googleapis.com/capstone-c242-ps370.appspot.com/ml-models/face-recognition/siamesemodel.h5", "./siamesemodel.h5")  # Default to local if not set
model = load_model(MODEL_URL_OR_PATH)
app.config["model"] = model

# Move this import inside the main execution block
def initialize_routes():
    from routes import register_routes
    register_routes(app)

# Error-handling middleware
@app.errorhandler(Exception)
def handle_exceptions(e):
    if isinstance(e, InputError):
        response = {"status": "fail", "message": str(e)}
        return jsonify(response), 400
    elif isinstance(e, HTTPException):
        response = {"status": "fail", "message": e.description}
        return jsonify(response), e.code
    else:
        response = {"status": "error", "message": "Internal Server Error"}
        return jsonify(response), 500

if __name__ == "__main__":
    initialize_routes()  # Import and register routes here
    HOST = "0.0.0.0"
    PORT = int(os.getenv("PORT", 3000))
    app.run(host=HOST, port=PORT, debug=True)
