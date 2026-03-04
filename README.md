# Tell Me Why – MRI Tumor Classification with CNN & ResNet50
This repository contains the implementation of deep learning models for the classification of brain MRI images into four diagnostics categories: 
- No tumor
- Glioma
- Meningioma
- Pituitary tumor
The main objective is to compare two different deep learning architectures (CNN and ResNet50) for medical image classification and analyze their interpretability using Grad-CAM based explainability techniques.

## Project Overview
The workflow of the project consists of four main stages:
1. Exploratory Data Analysis (EDA)
2. Model training and comparison
3. Explainability analysis using GradCAM
4. Quantitative evaluation of explanation methods

Two architectures were implemented and compared:
- A custom Convolutional Neural Network (CNN)
- A pretrained ResNet50 model
Both models were trained with and without early stopping, in order to evaluate the effect of training regularization on performance and computational efficiency. The final comparison considers both classification accuracy and explainability metrics, identifying the most reliable model for tumor detection.
