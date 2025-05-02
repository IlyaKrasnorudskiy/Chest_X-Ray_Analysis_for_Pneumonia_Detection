import streamlit as st
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model

img_size = 150
model = load_model('first_data/pneumonia_cnn_model.h5')

st.title("Pneumonia Detection from Chest X-Ray")
st.write("Upload a chest X-ray image (JPEG/PNG, grayscale or color).")

uploaded_file = st.file_uploader("Choose an X-ray image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('L')
    st.image(image, caption='Uploaded X-ray.', use_column_width=True)
    image = image.resize((img_size, img_size))
    img_array = np.array(image) / 255.0
    img_array = img_array.reshape(-1, img_size, img_size, 1)
    prediction = model.predict(img_array)[0][0]
    if prediction > 0.5:
        st.error(f"Prediction: PNEUMONIA ({prediction:.2f})")
    else:
        st.success(f"Prediction: NORMAL ({1-prediction:.2f})")
