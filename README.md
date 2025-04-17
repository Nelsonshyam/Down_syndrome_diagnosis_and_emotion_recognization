
# Down Syndrome Diagnosis and Emotion Recognition

This project presents a multi-stage deep learning system that first detects Down Syndrome using facial image analysis, and then accurately recognizes the emotional state of the subject. It is specially tailored to assist in health monitoring and emotional understanding of individuals with Down Syndrome using intelligent computer vision.


## 📌 Features

📷 Upload or live video input for real-time analysis

🧠 Detects Down Syndrome using VGG16 + NMF + LightGBM

😊 Performs emotion recognition with specialized models:

`fer.h5`: Emotion detection for normal children

`syndrome.keras`: Emotion detection for children with Down Syndrome

🎯 Dual-mode pipeline with automatic switching of models based on diagnosis

📈 Lightweight and accurate — designed for real-world performance


## 🧠 Model Architecture
1. Down Syndrome Diagnosis
Input: Facial image (RGB)

Feature Extractor: Pre-trained VGG16 (`include_top=False`)

Dimensionality Reduction: Pre-trained Non-negative Matrix Factorization (NMF)

Classifier: Trained LightGBM model

Threshold: Diagnosis classified as “Down Syndrome” if probability > 0.55

2. Emotion Recognition
Diagnosis-dependent model switching:

If diagnosed as Normal → `fer.h5` is used

If diagnosed as Syndrome → `syndrome.keras` is used

Preprocessing: Resized to model-specific dimensions, normalized, optionally grayscale

Output: Emotion label (e.g., Happy, Sad, Surprise, etc.)

Postprocessing: Uses `softmax` to interpret class probabilities

🏗️ Application Flow
Upload Image OR Start Camera Feed

Face is detected using Haar Cascade

Diagnosis model runs for 5 seconds (real-time mode)

Emotion prediction is performed after diagnosis

Annotated results shown on screen


## 🏗️ Application Flow
1. Upload Image OR Start Camera Feed

2. Face is detected using Haar Cascade

3. Diagnosis model runs for 5 seconds (real-time mode)

4. Emotion prediction is performed after diagnosis

5. Annotated results shown on screen
## 📁 File Structure

```
├── app.py                  # Flask backend with model loading and inference
├── index.html              # Frontend UI for upload and real-time video
├── model/
│   ├── diagnosis.pkl       # LightGBM model for Down Syndrome
│   ├── nmf_model.pkl       # NMF for dimensionality reduction
│   ├── fer.h5              # Emotion model for normal children
│   └── syndrome.keras      # Emotion model for children with Down Syndrome
├── static/uploads/         # Folder to store uploaded/annotated images
└── templates/index.html    # HTML frontend interface
```
## 🔧 Installation
```
pip install -r requirements.txt
```


## 🎯 Use Case
This application is designed to help:

- Caregivers and therapists of children with Down Syndrome

- Emotion monitoring in educational or health settings

- Integration into assistive technology devices

## Some files couldn't upload because of the space in github
use this - https://drive.google.com/drive/folders/1qU8iTPvzLcTSuQZfy_dO7Lb5S7LF9EN1?usp=sharing

