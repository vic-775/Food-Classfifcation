import streamlit as st
import numpy as np
import tensorflow as tf


# Model Prediction
def model_prediction(test_image):
    # load the model
    cnn_model = tf.keras.models.load_model(r'C:\Users\PC\Desktop\fuits_model\cnn_model_fruits.keras')
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128, 128))

    # converting image to array form
    input_arr = tf.keras.preprocessing.image.img_to_array(image)

    # converting input array into numpy / batch
    input_arr = np.array([input_arr])

    # predicting
    predictions = cnn_model.predict(input_arr)
    
    # return index of max element
    return np.argmax(predictions)

# The App
st.sidebar.title('Dashboard')
app = st.sidebar.selectbox("Select Page", ["Home", "About Project", "Predictions"])

# Home Page
if (app == "Home"):
    st.header("FOOD PREDICTION")
    st.write("""
        Welcome to the Food Prediction System! 🍎🥦

        This tool uses a deep learning model to predict the type of fruit or vegetable in an image you upload.
        It's designed to be fast, accurate, and easy to use — just exploring computer vision technology.

        To get started, navigate to the **Predict** tab and upload an image of a fruit or vegetable.
    """)
    
# About Page 
if (app == "About Project"):
    st.header("About Project")
    st.write("""
        This Fruit and Vegetable Classification System uses a Convolutional Neural Network (CNN) model to identify various fruits and vegetables from images.
        The goal is to support quick and accurate food recognition, which can be useful in applications such as:
        
        - Automated checkout systems in grocery stores
        - Dietary tracking and food logging apps
        - Educational tools for kids to learn about healthy foods
        - Agricultural produce classification

        The model has been trained on a diverse dataset of common fruits and vegetables to ensure high prediction accuracy.
        Upload an image, and the system will classify it in real-time with its corresponding label.
    """)

# Predictions Page   
if (app == "Predictions"):
    st.header("Predictions")
    test_image = st.file_uploader("Chose an image")
    if (st.button("Show image")):
        st.image(test_image)
        
    # Predictions
    if (st.button("Predict")):
        st.write("The Prediction")
        result_index = model_prediction(test_image)
        # Labels
        with open(r"C:\Users\PC\Desktop\fuits_model\labels.txt") as f:
            content = f.readlines()
            label = []
            for i in content:
                label.append(i[:-1])
            st.success('Its {}'.format(label[result_index]))
