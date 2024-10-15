import streamlit as st
import tensorflow as tf
import numpy as np
import os

def load_model():
    """Try loading the model from both .h5 and .keras formats."""
    model_path_h5 = 'trained_model.h5'
    model_path_keras = 'trained_plant_disease_model.keras'

    if os.path.isfile(model_path_h5):
        try:
            return tf.keras.models.load_model(model_path_h5)
        except Exception as e:
            print(f"Error loading .h5 model: {e}")

    if os.path.isfile(model_path_keras):
        try:
            return tf.keras.models.load_model(model_path_keras)
        except Exception as e:
            print(f"Error loading .keras model: {e}")

    raise FileNotFoundError("Model file not found. Ensure you have the correct model file.")

def model_prediction(model, test_image):
    """Predict the class of the uploaded image using the pre-trained model."""
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128, 128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])  # Convert single image to batch
    predictions = model.predict(input_arr)
    return np.argmax(predictions)  # Return index of max element

# Sidebar
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page", ["Home", "About", "Disease Recognition"])

# Main Page
if app_mode == "Home":
    st.header("PLANT DISEASE RECOGNITION SYSTEM")
    image_path = "homepage.jpg"
    st.image(image_path, use_column_width=True)
    st.markdown("""
    Welcome to the Plant Disease Recognition System! 🌿🔍
    
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
    This dataset consists of about 87K RGB images of healthy and diseased crop leaves categorized into 38 different classes. The total dataset is divided into an 80/20 ratio of training and validation sets preserving the directory structure.
    A new directory containing 33 test images is created later for prediction purposes.
    #### Content
    1. train (70295 images)
    2. test (33 images)
    3. validation (17572 images)
    """)

# Prediction Page
elif app_mode == "Disease Recognition":
    st.header("Disease Recognition")
    test_image = st.file_uploader("Choose an Image:", type=['jpg', 'jpeg', 'png'])
    
    if test_image is not None:
        if st.button("Show Image"):
            st.image(test_image, width=400, use_column_width=True)
        
        # Predict button
        if st.button("Predict"):
            st.snow()  # Add a spinner while predicting
            st.write("Our Prediction")

            # Load model (cached)
            model = load_model()
            
            # Perform the prediction
            result_index = model_prediction(model, test_image)
            
            # Define class names
            class_names = [
                'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
                'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 
                'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot_Gray_leaf_spot', 
                'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 
                'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 
                'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot',
                'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 
                'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 
                'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew', 
                'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 
                'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 
                'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites_Two-spotted_spider_mite', 
                'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
                'Tomato___healthy'
            ]
            
            # Define treatments
            treatments = {
                'Apple___Apple_scab': "Apply fungicides such as copper-based products or other scab-specific fungicides.",
                'Apple___Black_rot': "Remove and destroy infected fruit. Apply fungicides like Mancozeb or Chlorothalonil.",
                'Apple___Cedar_apple_rust': "Use fungicides like sulfur or potassium bicarbonate. Remove cedar trees if possible.",
                'Apple___healthy': "No treatment needed.",
                'Blueberry___healthy': "No treatment needed.",
                'Cherry_(including_sour)___Powdery_mildew': "Use fungicides like sulfur or potassium bicarbonate.",
                'Cherry_(including_sour)___healthy': "No treatment needed.",
                'Corn_(maize)___Cercospora_leaf_spot_Gray_leaf_spot': "Apply fungicides like Azoxystrobin or Pyraclostrobin.",
                'Corn_(maize)___Common_rust_': "Use resistant varieties. Apply fungicides such as Triazole.",
                'Corn_(maize)___Northern_Leaf_Blight': "Apply fungicides and practice crop rotation.",
                'Corn_(maize)___healthy': "No treatment needed.",
                'Grape___Black_rot': "Apply fungicides like Myclobutanil or copper-based products.",
                'Grape___Esca_(Black_Measles)': "Prune affected vines and use fungicides for control.",
                'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)': "Apply fungicides such as Chlorothalonil.",
                'Grape___healthy': "No treatment needed.",
                'Orange___Haunglongbing_(Citrus_greening)': "Use systemic insecticides and remove infected trees.",
                'Peach___Bacterial_spot': "Apply copper-based bactericides and remove infected plant material.",
                'Peach___healthy': "No treatment needed.",
                'Pepper,_bell___Bacterial_spot': "Use copper-based bactericides and avoid overhead irrigation.",
                'Pepper,_bell___healthy': "No treatment needed.",
                'Potato___Early_blight': "Apply fungicides such as Chlorothalonil or Mancozeb.",
                'Potato___Late_blight': "Use fungicides like Ridomil Gold or Copper-based products.",
                'Potato___healthy': "No treatment needed.",
                'Raspberry___healthy': "No treatment needed.",
                'Soybean___healthy': "No treatment needed.",
                'Squash___Powdery_mildew': "Apply fungicides like Sulfur or Potassium Bicarbonate.",
                'Strawberry___Leaf_scorch': "Apply fungicides and improve air circulation around plants.",
                'Strawberry___healthy': "No treatment needed.",
                'Tomato___Bacterial_spot': "Use copper-based bactericides and remove infected plant material.",
                'Tomato___Early_blight': "Apply fungicides such as Chlorothalonil or Mancozeb.",
                'Tomato___Late_blight': "Use fungicides like Ridomil Gold or Copper-based products.",
                'Tomato___Leaf_Mold': "Improve ventilation and apply fungicides like Chlorothalonil.",
                'Tomato___Septoria_leaf_spot': "Apply fungicides and remove affected leaves.",
                'Tomato___Spider_mites_Two-spotted_spider_mite': "Use miticides or increase humidity around plants.",
                'Tomato___Target_Spot': "Apply fungicides such as Chlorothalonil or Mancozeb.",
                'Tomato___Tomato_Yellow_Leaf_Curl_Virus': "Remove infected plants and control whiteflies.",
                'Tomato___Tomato_mosaic_virus': "Remove infected plants and practice crop rotation.",
                'Tomato___healthy': "No treatment needed."
            }
            
            # Display the result
            predicted_class = class_names[result_index]
            treatment = treatments.get(predicted_class, "Treatment information not available.")
            st.success(f"Model predicts it's a {predicted_class}.")
            st.write(f"**Recommended Treatment:** {treatment}")
