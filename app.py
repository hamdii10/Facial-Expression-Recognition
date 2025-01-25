import streamlit as st
import tensorflow as tf
import os
import numpy as np
from PIL import Image, ImageOps

#the two lines below for streamlit online (anyone can access)
model_path = os.path.join('models', 'model.keras')
model = tf.keras.models.load_model(open(model_path, 'rb'))


# Load model
#Uncomment the following to lines to run the streamlit local
''' model = tf.keras.models.load_model("X:/FER/model.keras")
'''

class_names = {0: "Angry", 1: "Disgust", 2: "Fear", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprise"}

# Title and description
st.title("Facial Expression Recognition Application")
st.write("""
This application predicts the facial expression in an uploaded image. 
Please upload a suitable image and click 'Predict'.
""")

# Define a reset function
def reset_inputs():
    st.session_state.image = None

# Initialize session state for image upload
if "image" not in st.session_state:
    reset_inputs()

# File uploader
image = st.file_uploader("Upload an image (JPG/JPEG/PNG)...", type=["jpg", "jpeg", "png"])

# Display uploaded image
if image is not None:
    st.image(image, caption="Uploaded Image", use_column_width=True)
    image = Image.open(image)
    if len(np.array(image).shape) != 2:  # If not grayscale
        image = ImageOps.grayscale(image)
        st.warning("Invalid Image Format. Automatically converted to Grayscale.")

# Buttons for predict and clear
col1, col2 = st.columns(2)
with col1:
    state = st.button('Predict')
with col2:
    clear = st.button('Clear', on_click=reset_inputs)

# Prediction logic
if state and image is not None:
    with st.spinner("Processing Image..."):
        # Resize image
        image = image.resize([96, 96])
        # Normalize image
        image = np.array(image) / 255.0
        # Predict
        pred = model.predict(np.expand_dims(image, axis=0))
        pred_label = np.argmax(pred)
        pred_class = class_names[pred_label]
        

    # Display result
    st.success("Prediction Completed!")
    st.markdown(f"### Predicted Label: **{pred_class}**")
else:
    if state:
        st.error("Please upload an image first.")

# Sidebar information
st.sidebar.title("About")
st.sidebar.info("""
This facial expression recognition model uses a convolutional neural network built with TensorFlow.
""")

st.sidebar.header("How It Works")
st.sidebar.write("""
1. Upload an image in JPEG or PNG format.
2. Click the 'Predict' button to get the result.
3. Use the 'Clear' button to reset inputs.
""")

st.sidebar.header("Developer Notes")
st.sidebar.write("""
- This application is built with Python and Streamlit.
- The machine learning model used was trained on a custom dataset for facial expression recognition.
""")

st.sidebar.header("Contact")
st.sidebar.write("""
For questions or suggestions, reach out to:
- **Email**: ahmed.hamdii.kamal@gmail.com
- **GitHub**: [hamdii10](https://github.com/hamdii10)
""")
