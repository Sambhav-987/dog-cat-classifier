import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Load model
model = tf.keras.models.load_model("dog_cat_model.h5")

# Title
st.title("🐶🐱 Dog vs Cat Classifier")

# Upload image
uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    # Open image
    img = Image.open(uploaded_file)

    # Display image
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Resize image
    img = img.resize((128,128))

    # Convert to numpy array
    img_array = np.array(img)

    # Normalize
    img_array = img_array / 255.0

    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    prediction = model.predict(img_array)

    confidence = prediction[0][0]

    # Show result
    if confidence > 0.5:
        st.success(f"Dog 🐶 ({confidence*100:.2f}% confidence)")
    else:
        st.success(f"Cat 🐱 ({(1-confidence)*100:.2f}% confidence)")
