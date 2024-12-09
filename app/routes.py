from flask import Blueprint
from handler import post_predict_handler, get_predict_histories_handler

def register_routes(app):
    # Blueprint for API routes
    api = Blueprint("api", __name__)

    # Define routes
    api.add_url_rule("/predict", methods=["POST"], view_func=post_predict_handler)
    api.add_url_rule("/predict/histories", methods=["GET"], view_func=get_predict_histories_handler)

    # Register blueprint
    app.register_blueprint(api)
