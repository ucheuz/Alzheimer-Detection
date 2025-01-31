# Alzheimer-Detection
A machine learning algorithm designed to diagnose Alzheimer's disease by analysing handwriting samples, distinguishing between healthy individuals and Alzheimer's patients, to provide a more accurate and simplified detection method for clinical practice. The project includes: A Python Script (main.py) that runs the model A Juypter Notebook (Alzheimer Detection.ipynb) Task 2 with exploratory data analyis, model training and conclusions about the model at each stage.

# Description
In this project, I aimed to develop a machine learning model for Alzheimer's disease detection using handwriting analysis. The goal was to distinguish between healthy individuals and Alzheimer's patients by extracting features from handwriting samples. To reduce dimensionality, I applied PCA and Laplacian Eigenmaps, allowing for better visualisation of the data. The model was trained using Random Forest and cross-validated for accuracy. To improve model simplicity and accuracy, I employed Recursive Feature Elimination (RFE) with logistic regression to select the most important features. The final approach included training a neural network in PyTorch with a simple architecture, evaluating its performance, and visualising the decision boundary, ensuring the model provided an efficient and accurate method for Alzheimer's detection.

# How to run
Option 1: Running the Python Script: Can be done directly in terminal. This will train the model and output predictions and accuracy scores.

Option 2: Running the Juypter Notebook: Download and open Alzheimer Detection.ipynb and run the cells in Task 1 step by step.

# Dataset
File: dataset/DARWIN_dataset.csv
Description: The DARWIN dataset includes handwriting data from 174 participants.

# Results
Test Accuracy: 0.83
Conclusion: The Alzheimer's detection model successfully classified patients using handwriting samples with reduced dimensionality, achieving a solid classification performance.

# Contributors
Kosiasochukwu Uchemudi Uzoka - Author


