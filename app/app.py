import os
from flask import Flask, jsonify
from werkzeug.exceptions import HTTPException
from services.load_model import load_model
from routes import register_routes
from exceptions import InputError

# Initialize Flask app
app = Flask(__name__)

# Load model
MODEL_URL = os.getenv("MODEL_URL")
model = load_model(MODEL_URL)
app.config["model"] = model

# Register routes
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
    HOST = "0.0.0.0"
    PORT = int(os.getenv("PORT", 3000))
    app.run(host=HOST, port=PORT, debug=True)
