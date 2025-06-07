import tensorflow as tf
import numpy as np
from PIL import Image
import streamlit as st

def load_model():
    @st.cache_resource
    def _load():
        return tf.keras.models.load_model('best_model.h5')
    return _load()

def preprocess_image(image):
    img = image.resize((150, 150))
    img = img.convert('L')
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = np.expand_dims(img_array, axis=-1)
    img_array = img_array / 255.0
    return img_array

def analyze_single_image(image, model):
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)[0][0]
    result = "Норма" if prediction > 0.5 else "Пневмония"
    confidence = prediction if prediction > 0.5 else 1 - prediction
    return {
        "Результат": result,
        "Уверенность": f"{confidence:.2%}",
        "Вероятность нормы": f"{prediction:.2%}",
        "Вероятность пневмонии": f"{(1-prediction):.2%}"
    } 