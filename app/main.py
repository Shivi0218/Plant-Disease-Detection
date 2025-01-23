import os
import json
from PIL import Image

import numpy as np
import tensorflow as tf
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Set Streamlit app-wide configurations
st.set_page_config(
    page_title="Plant Disease Classifier",
    page_icon="🌿",
    layout="wide",
)


st.markdown(
   """
    <style>
    body{
        background-image: url('https://www.pexels.com/photo/people-harvesting-2131784/');
        background-size: cover;
    }
   </style>
    """,
    unsafe_allow_html=True
)

# Load the pre-trained model and class indices
working_dir = os.path.dirname(os.path.abspath(__file__))
model_path = f"{working_dir}/trained_model/plant_disease_model.h5"
model = tf.keras.models.load_model(model_path)
class_indices = json.load(open(f"{working_dir}/class_indices.json"))

# Define color palette for class labels
colors = list(mcolors.TABLEAU_COLORS.values())

# Function to Load and Preprocess the Image using Pillow
def load_and_preprocess_image(image, target_size=(224, 224)):
    img = image.convert('RGB')  # Convert grayscale image to RGB
    img = img.resize(target_size)
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array.astype('float32') / 255.
    return img_array

# Function to Predict the Class of an Image
def predict_image_class(model, image, class_indices):
    preprocessed_img = load_and_preprocess_image(image)
    predictions = model.predict(preprocessed_img)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_class_name = class_indices[str(predicted_class_index)]
    return predicted_class_name

# Function to plot histogram
def plot_histogram(image_array, figsize=(8, 6)):
    plt.figure(figsize=figsize)
    plt.hist(image_array.ravel(), bins=256, color='orange', histtype='step')
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Frequency')
    plt.title('Histogram')
    plt.grid(True)
    return plt

# Streamlit App
st.title('🌿 Plant Disease Classifier')

# Upload image
uploaded_image = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

# Classify image and perform analysis
if uploaded_image is not None:
    # Display uploaded image
    st.subheader("Uploaded Image")
    image = Image.open(uploaded_image)
    st.image(image, caption='Uploaded Image', width=300)

    # Classify on button click
    if st.button('Classify'):
        # Perform image classification
        prediction = predict_image_class(model, image, class_indices)
        st.success(f'Prediction: {str(prediction)}')

        # Visualize class distribution
        class_names = list(class_indices.values())
        class_counts = [0] * len(class_names)
        class_index = class_names.index(prediction)
        class_counts[class_index] = 1

        # Create bar chart for color image prediction
        fig_color, ax_color = plt.subplots(figsize=(10, 8))
        ax_color.barh(class_names, class_counts, color=colors)
        plt.xlabel('Count')
        plt.ylabel('Disease Class')
        plt.title('Distribution of Predicted Classes (Color Image)')
        plt.grid(axis='x')
        plt.xticks(np.arange(0, 2, step=1))  # Set x-axis ticks to integers
        plt.yticks(np.arange(len(class_names)), class_names)  # Add space in y-axis
        plt.tight_layout()  # Adjust layout to prevent clipping of labels
        plt.subplots_adjust(left=0.3)  # Add space on the left for the y-axis labels
        st.pyplot(fig_color)

    # Display original and grayscale images with histograms
    st.subheader("Image Analysis")
    col1, col2 = st.columns(2)

    with col1:
        st.write("**Original Image**")
        st.image(image, caption='Original Image', width=400)
        st.write("Histogram of Original Image")
        plt_original = plot_histogram(np.array(image), figsize=(6, 4))
        st.pyplot(plt_original)

    with col2:
        # Convert image to grayscale
        grayscale_image = image.convert('L')
        st.write("**Grayscale Image**")
        st.image(grayscale_image, caption='Grayscale Image', width=400)
        st.write("Histogram of Grayscale Image")
        plt_grayscale = plot_histogram(np.array(grayscale_image), figsize=(6, 4))
        st.pyplot(plt_grayscale)

    # Classify on button click for grayscale image
    if st.button('Classify (Grayscale)'):
        # Perform image classification
        prediction_grayscale = predict_image_class(model, grayscale_image, class_indices)
        st.success(f'Prediction (Grayscale): {str(prediction_grayscale)}')

        # Visualize class distribution
        class_names = list(class_indices.values())
        class_counts = [0] * len(class_names)
        class_index = class_names.index(prediction_grayscale)
        class_counts[class_index] = 1

        # Create bar chart for grayscale image prediction
        fig_grayscale, ax_grayscale = plt.subplots(figsize=(10, 8))
        ax_grayscale.barh(class_names, class_counts, color=colors)
        plt.xlabel('Count')
        plt.ylabel('Disease Class')
        plt.title('Distribution of Predicted Classes (Grayscale Image)')
        plt.grid(axis='x')
        plt.xticks(np.arange(0, 2, step=1))  # Set x-axis ticks to integers
        plt.yticks(np.arange(len(class_names)), class_names)  # Add space in y-axis
        plt.tight_layout()  # Adjust layout to prevent clipping of labels
        plt.subplots_adjust(left=0.3)  # Add space on the left for the y-axis labels
        st.pyplot(fig_grayscale)

        # Explanation for grayscale prediction
        st.info("The prediction for the grayscale image may not be accurate because the model was trained on color images. Grayscale images lack color information, which may lead to different predictions.")