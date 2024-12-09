import tensorflow as tf

def load_model(model_url):
    return tf.keras.models.load_model(model_url)
