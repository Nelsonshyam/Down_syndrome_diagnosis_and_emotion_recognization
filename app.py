from flask import Flask, render_template, request, Response, jsonify, url_for
import cv2
import numpy as np
import joblib
import tensorflow as tf
import time
import os
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow.keras.applications import VGG16

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

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
emotion_labels_normal = ["Sad", "Angry", "Fear","Happy" , "Neutral","Surprise" ,"Disgust" ]
emotion_labels_syndrome = ["Surprise", "Angry", "Fear", "Happy", "Neutral", "Sad","Disgust" , "Contempt"]

# Global variables for video stream
diagnosis_counts = {0: 0, 1: 0}  # Track counts over 5 seconds
start_time = None
has_syndrome = None  # Default to None to ensure diagnosis runs first
emotion = "Neutral"  # Default emotion
diagnosis_done = False  # Flag to indicate if diagnosis is completed
video_capture = None

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Process the image
            diagnosis_result, emotion_result, face_detected = process_image(filepath)
            
            if not face_detected:
                return jsonify({
                    'error': 'No face detected in the image',
                    'image_path': url_for('static', filename=f'uploads/{filename}')
                }), 400
            
            return jsonify({
                'diagnosis': diagnosis_result,
                'emotion': emotion_result,
                'image_path': url_for('static', filename=f'uploads/{filename}')
            })
        
        return jsonify({'error': 'File type not allowed'}), 400
    except Exception as e:
        print(f"Error in upload_file: {str(e)}")
        return jsonify({'error': f'Server error: {str(e)}'}), 500

def process_image(image_path):
    try:
        image = cv2.imread(image_path)
        if image is None:
            return "Error", "Error", False
        
        # Detect faces (for visualization only)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=4, minSize=(50, 50))
        
        if len(faces) == 0:
            return "No face detected", "N/A", False
        
        # For visualization - get face coordinates
        x, y, w, h = faces[0]
        
        # Process the ENTIRE image instead of just the face
        # Preprocess for VGG16
        image_resized = cv2.resize(image, (224, 224)) / 255.0
        image_expanded = np.expand_dims(image_resized, axis=0)
        
        # Extract features from the entire image
        features = feature_extractor.predict(image_expanded, verbose=0)
        features = features.flatten().reshape(1, -1)
        
        # Apply NMF for dimensionality reduction
        features_reduced = nmf.transform(features)
        
        # Predict Down Syndrome diagnosis
        prob = diagnosis_model.predict(features_reduced)[0]
        THRESHOLD = 0.55  # Adjust threshold if needed
        has_syndrome = 1 if prob > THRESHOLD else 0
        diagnosis_result = "Down Syndrome" if has_syndrome else "Normal"
        
        # Predict emotion based on diagnosis
        if has_syndrome:
            emotion_model = model_syndrome
            emotion_labels = emotion_labels_syndrome
        else:
            emotion_model = model_normal
            emotion_labels = emotion_labels_normal
        
        # Preprocess entire image for emotion model
        input_size = emotion_model.input_shape[1:3]
        
        # Convert to grayscale if model expects (48,48,1)
        if emotion_model.input_shape[-1] == 1:
            image_emotion = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            image_emotion = image
        
        # Resize & normalize
        image_emotion = cv2.resize(image_emotion, input_size).astype("float32") / 255.0
        
        # Expand dimensions
        if emotion_model.input_shape[-1] == 1:
            image_emotion = np.expand_dims(image_emotion, axis=-1)
        image_emotion = np.expand_dims(image_emotion, axis=0)
        
        # Predict emotion
        emotion_preds = emotion_model.predict(image_emotion, verbose=0)
        emotion_probs = tf.nn.softmax(emotion_preds).numpy()
        emotion_index = np.argmax(emotion_probs)
        emotion_result = emotion_labels[emotion_index]
        
        # Draw rectangle on image to show face detection (for visualization only)
        cv2.rectangle(image, (x, y), (x + w, y + h), 
                    (0, 0, 255) if has_syndrome else (0, 255, 0), 2)
                    
        # Add text with diagnosis and emotion
        cv2.putText(image, f"Diagnosis: {diagnosis_result}", (x, y - 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
                    (0, 0, 255) if has_syndrome else (0, 255, 0), 2)
        cv2.putText(image, f"Emotion: {emotion_result}", (x, y + h + 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
                    (0, 0, 255) if has_syndrome else (0, 255, 0), 2)
        
        # Save the annotated image
        annotated_path = os.path.join(os.path.dirname(image_path), "annotated_" + os.path.basename(image_path))
        cv2.imwrite(annotated_path, image)
        
        return diagnosis_result, emotion_result, True
    except Exception as e:
        print(f"Error in process_image: {str(e)}")
        return "Error", "Error", False
    
def process_diagnosis(face):
    global has_syndrome, diagnosis_counts, start_time, diagnosis_done
    
    if diagnosis_done:
        return
    
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
        has_syndrome = max(diagnosis_counts, key=diagnosis_counts.get) == 1
        diagnosis_done = True
        diagnosis_counts = {0: 0, 1: 0}  # Reset counts

def process_emotion(face, frame, x, y, w, h):
    global emotion
    
    if not diagnosis_done:
        return
    
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

def generate_frames():
    global video_capture, start_time, diagnosis_done, emotion, has_syndrome
    
    # Reset state variables
    start_time = time.time()
    diagnosis_done = False
    emotion = "Neutral"
    has_syndrome = None
    diagnosis_counts = {0: 0, 1: 0}
    
    # Initialize video capture
    if video_capture is None:
        video_capture = cv2.VideoCapture(0)
    
    while True:
        try:
            success, frame = video_capture.read()
            if not success:
                break
            
            # Process frame
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            faces = face_cascade.detectMultiScale(rgb, scaleFactor=1.2, minNeighbors=4, minSize=(50, 50))
            
            for (x, y, w, h) in faces:
                face_rgb = cv2.resize(frame[y:y+h, x:x+w], (224, 224)) / 255.0
                input_data = np.expand_dims(face_rgb, axis=0)
                
                # Process diagnosis in the first 5 seconds
                if not diagnosis_done:
                    process_diagnosis(input_data)
                else:
                    # Process emotion prediction after diagnosis is done
                    process_emotion(input_data, frame, x, y, w, h)
                
                # Draw rectangle & display results
                color = (0, 0, 255) if has_syndrome else (0, 255, 0)
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                
                # Display diagnosis status
                if not diagnosis_done:
                    elapsed = time.time() - start_time
                    remaining = max(0, 5 - elapsed)
                    cv2.putText(frame, f"Diagnosing... {remaining:.1f}s", (x, y - 20), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                else:
                    label_text = f"Diagnosis: {'Down Syndrome' if has_syndrome else 'Normal'}"
                    emotion_text = f"Emotion: {emotion}"
                    cv2.putText(frame, label_text, (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                    cv2.putText(frame, emotion_text, (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # Convert to JPEG
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            
            # Yield frame
            yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        except Exception as e:
            print(f"Error in generate_frames: {str(e)}")
            time.sleep(0.1)  # Add a small delay to prevent CPU overload in case of repeated errors

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start_video', methods=['POST'])
def start_video():
    global video_capture, start_time, diagnosis_done, emotion, has_syndrome
    
    # Reset state variables
    start_time = time.time()
    diagnosis_done = False
    emotion = "Neutral"
    has_syndrome = None
    diagnosis_counts = {0: 0, 1: 0}
    
    return jsonify({'status': 'started'})

@app.route('/stop_video', methods=['POST'])
def stop_video():
    global video_capture
    if video_capture is not None:
        video_capture.release()
        video_capture = None
    return jsonify({'status': 'stopped'})

if __name__ == '__main__':
    app.run(debug=True)