import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load the model once, outside any functions
model = tf.keras.models.load_model("trained_model.keras", compile=False)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# TensorFlow Model Prediction
def model_prediction(test_image):
    image = Image.open(test_image)
    image = image.resize((128, 128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.expand_dims(input_arr, axis=0)  # Convert single image to batch
    input_arr = input_arr / 255.0  # Normalize the image
    predictions = model.predict(input_arr)
    st.write(f"Raw predictions: {predictions}")
    return np.argmax(predictions)  # Return index of max element

# Sidebar
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page", ["Home", "About", "Disease Recognition"])

# Main Page
if app_mode == "Home":
    st.header("PLANT LEAF DISEASE DETECTION SYSTEM")
    image_path = "home_page.jpeg"
    st.image(image_path, use_column_width=True)
    st.markdown("""
    Welcome to the Plant Leaf Disease Detection System! 🌿🔍
    
    Our mission is to help in identifying plant diseases efficiently. Upload an image of a plant, and our system will analyze it to detect any signs of diseases. Together, let's protect our crops and ensure a healthier harvest!

    ### How It Works
    1. **Upload Image:** Go to the **Disease Recognition** page and upload an image of a plant with suspected diseases.
    2. **Analysis:** Our system will process the image using advanced algorithms to identify potential diseases.
    3. **Results:** View the results and recommendations for further action.

    ### Why Choose Us?
    - **Accuracy:** Our system utilizes state-of-the-art machine learning techniques for accurate disease detection.
    - **User-Friendly:** Simple and intuitive interface for seamless user experience.
    - **Fast and Efficient:** Receive results in seconds, allowing for quick decision-making.

    ### Get Started
    Click on the **Disease Recognition** page in the sidebar to upload an image and experience the power of our Plant Disease Recognition System!

    ### About Us
    Learn more about the project, our team, and our goals on the **About** page.
    """)

# About Project
elif app_mode == "About":
    st.header("About")
    st.markdown("""
    #### About Dataset
    This dataset is recreated using offline augmentation from the original dataset. The original dataset can be found on this GitHub repo.
    This dataset consists of about 87K RGB images of healthy and diseased crop leaves which are categorized into 38 different classes. The total dataset is divided into an 80/20 ratio of training and validation sets preserving the directory structure.
    A new directory containing 33 test images is created later for prediction purposes.

    #### Content
    1. train (70295 images)
    2. test (33 images)
    3. validation (17572 images)
    """)

# Prediction Page
elif app_mode == "Disease Recognition":
    st.header("Disease Recognition")
    test_image = st.file_uploader("Choose an Image:")
    
    if test_image:
        image = Image.open(test_image)
        if st.button("Show Image"):
            st.image(image, use_column_width=True)
        
        # Predict button
        if st.button("Predict"):
        
            st.write("Our Prediction")
            result_index = model_prediction(test_image)
            # Reading Labels
            class_names = [
                'Apple___Apple_scab',
                  'Apple___Cedar_apple_rust', 
                  'Apple___healthy',
                    'Cherry_(including_sour)___Powdery_mildew',
                      'Cherry_(including_sour)___healthy',
                  'Corn_(maize)___Common_rust_',
                    'Corn_(maize)___healthy',
                      'Tomato___Tomato_mosaic_virus', 
                      'Tomato___healthy'
            ]
            st.success(f"Model is predicting it's a {class_names[result_index]}")