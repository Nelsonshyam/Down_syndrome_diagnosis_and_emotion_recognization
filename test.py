import cv2
import numpy as np
import joblib
import tensorflow as tf
import time
from tensorflow.keras.models import load_model
from tensorflow.keras.applications import VGG16

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Load VGG16 model for feature extraction
vgg16 = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
feature_extractor = tf.keras.Model(inputs=vgg16.input, outputs=vgg16.get_layer("block5_pool").output)

# Load trained models
diagnosis_model = joblib.load("model/diagnosis.pkl")  # LightGBM model for Down Syndrome detection
nmf = joblib.load("model/nmf_model.pkl")  # Pre-trained NMF model

# Load emotion recognition models
model_normal = load_model("model/fer.h5")  # Normal children emotion model
model_syndrome = load_model("model/syndrome.keras")  # Down syndrome children emotion model

# Define emotion labels
emotion_labels_normal = ["Happy", "Angry", "Fear", "Surprise", "Disgust", "Sad", "Neutral"]
emotion_labels_syndrome = ["Happy", "Angry", "Fear", "Surprise", "Disgust", "Sad", "Neutral", "Contempt"]

# Global variables
diagnosis_counts = {0: 0, 1: 0}  # Track counts over 5 seconds
start_time = time.time()
has_syndrome = None  # Default to None to ensure diagnosis runs first
emotion = "Neutral"  # Default emotion
diagnosis_done = False  # Flag to indicate if diagnosis is completed

# 📌 **Function to Process Diagnosis**
def process_diagnosis(face):
    global has_syndrome, diagnosis_counts, start_time, diagnosis_done

    if diagnosis_done:
        return  # Stop diagnosis after 5 seconds

    # Extract deep features using VGG16
    features = feature_extractor.predict(face, verbose=0)
    features = features.flatten().reshape(1, -1)

    # Apply NMF for dimensionality reduction
    features_reduced = nmf.transform(features)

    # Predict Down Syndrome diagnosis
    prob = diagnosis_model.predict(features_reduced)[0]
    current_diagnosis = 1 if prob > 0.5 else 0
    diagnosis_counts[current_diagnosis] += 1

    # Check if 5 seconds have passed
    if time.time() - start_time >= 5:
        has_syndrome = max(diagnosis_counts, key=diagnosis_counts.get)  # Choose the most frequent label
        diagnosis_done = True  # Stop diagnosis process
        diagnosis_counts = {0: 0, 1: 0}  # Reset counts

# 📌 **Function to Process Emotion Prediction**
def process_emotion(face, frame, x, y, w, h):
    global emotion
    
    if not diagnosis_done:
        return  # Wait until diagnosis is done

    # Choose emotion model
    model = model_syndrome if has_syndrome else model_normal
    emotion_labels = emotion_labels_syndrome if has_syndrome else emotion_labels_normal

    # Ensure correct input shape for the emotion model
    input_size = model.input_shape[1:3]
    face_cropped = frame[y:y+h, x:x+w]  # Crop face properly

    # Convert to grayscale if model expects (48,48,1)
    if model.input_shape[-1] == 1:  
        face_cropped = cv2.cvtColor(face_cropped, cv2.COLOR_BGR2GRAY)

    # Resize & normalize
    face_resized = cv2.resize(face_cropped, input_size).astype("float32") / 255.0  

    # Expand dimensions to match model input
    if model.input_shape[-1] == 1:  
        face_resized = np.expand_dims(face_resized, axis=-1)  # Add channel for grayscale
    input_data = np.expand_dims(face_resized, axis=0)

    # Predict emotion
    emotion_preds = model.predict(input_data, verbose=0)
    emotion_probs = tf.nn.softmax(emotion_preds).numpy()
    emotion_index = np.argmax(emotion_probs)
    emotion = emotion_labels[emotion_index]

# 📌 **Real-time Emotion Recognition using Webcam**
def real_time_emotion_recognition():
    global has_syndrome, emotion

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FPS, 30)  # Set camera FPS

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture image")
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = face_cascade.detectMultiScale(rgb, scaleFactor=1.2, minNeighbors=4, minSize=(50, 50))

        for (x, y, w, h) in faces:
            face_rgb = cv2.resize(frame[y:y+h, x:x+w], (224, 224)) / 255.0
            input_data = np.expand_dims(face_rgb, axis=0)

            # Process diagnosis in the first 5 seconds
            if not diagnosis_done:
                process_diagnosis(input_data)  # Call function directly
            else:
                # Process emotion prediction immediately after diagnosis is done
                process_emotion(input_data, frame, x, y, w, h)  # Call function directly

            # Draw rectangle & display results
            color = (0, 0, 255) if has_syndrome else (0, 255, 0)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            label_text = f"Diagnosis: {'Down Syndrome' if has_syndrome else 'Normal'}"
            emotion_text = f"Emotion: {emotion}"

            cv2.putText(frame, label_text, (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.putText(frame, emotion_text, (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        cv2.imshow("Emotion Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# 📌 **Run Real-time Emotion Recognition**
if __name__ == "__main__":
    real_time_emotion_recognition()
