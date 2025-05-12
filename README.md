Plant Disease Detection Using Multi-Model Deep Learning Fusion
This project presents a deep learning-based system for detecting plant leaf diseases and recommending sustainable organic treatments. It uses a fusion of three models—Simple CNN, Deep CNN, and EfficientNetB0—to enhance prediction accuracy by capturing both shallow and deep image features.

Features
Multi-model fusion for robust plant disease classification

Real-time prediction via a user-friendly web interface

Organic pesticide recommendations based on detected disease

Trained using PlantVillage, Plant Leaves, and PlantDoc datasets

Developed with TensorFlow and Keras

How It Works
Users upload an image of a plant leaf via the web app.

The image is processed through three parallel CNN branches.

Extracted features are combined and passed through dense layers.

The system identifies the disease and suggests an organic treatment.

Tech Stack
Python, TensorFlow, Keras

HTML/CSS/JavaScript (for web app)

JSON (for pesticide mapping)

Datasets Used
PlantVillage (clean lab images)

Plant Leaves (varied lighting conditions)

PlantDoc (real-world noisy images)

Goal
To create an accurate, accessible, and eco-friendly tool for supporting sustainable agriculture and smart farming.

