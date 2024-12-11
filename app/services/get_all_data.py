from google.cloud import firestore

def get_all_data():
    db = firestore.Client()
    predict_collection = db.collection("predictions")
    return predict_collection.stream()
