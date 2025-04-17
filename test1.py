import cv2
import numpy as np
import joblib
import tensorflow as tf
from tensorflow.keras.applications import VGG16
import matplotlib.pyplot as plt

# Load VGG16 model for feature extraction
vgg16 = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
feature_extractor = tf.keras.Model(inputs=vgg16.input, outputs=vgg16.get_layer("block5_pool").output)

# Load the trained models
diagnosis_model = joblib.load("model/diagnosis.pkl")  # Load LightGBM model
nmf = joblib.load("model/nmf_model.pkl")  # Load the pre-trained NMF model

# Load the image
image_path = r"C:\Users\nelso\Desktop\sadhuman.jpg"  # Replace with your image path
image = cv2.imread(image_path)

if image is None:
    print("Error: Could not read the image.")
else:
    # Preprocess the image for VGG16
    image_resized = cv2.resize(image, (224, 224)) / 255.0  # Normalize
    image_expanded = np.expand_dims(image_resized, axis=0)  # Add batch dimension
    
    # Extract deep features using VGG16
    features = feature_extractor.predict(image_expanded)
    features = features.flatten().reshape(1, -1)  # Flatten to 1D vector
    
    # Apply the **pre-trained** NMF for dimensionality reduction
    features_reduced = nmf.transform(features)  # Use transform, NOT fit_transform

    # Predict diagnosis probability (0 = Normal, 1 = Down Syndrome)
    prob = diagnosis_model.predict(features_reduced)[0]  # Get probability score
    THRESHOLD = 0.4  # Adjust threshold if needed
    prediction = 1 if prob > THRESHOLD else 0
    
    # Print the result
    print("Prediction Probability:", prob)
    print("Final Adjusted Diagnosis:", "Down Syndrome" if prediction == 1 else "Normal")
    
