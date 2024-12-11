from google.cloud import firestore

def store_data(doc_id, data):
    db = firestore.Client()
    predict_collection = db.collection("predictions")
    predict_collection.document(doc_id).set(data)
