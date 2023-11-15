# Health-Insurance-Claims-Prediction-using-Artificial-Neural-Networks-ANN-
Overview
This project revolves around predicting health insurance claims utilizing the power of Artificial Neural Networks (ANN). The objective is to develop a robust model that accurately forecasts whether a policyholder is likely to file a health insurance claim. The dataset used for training and evaluation encompasses structured information on policy details, medical history, and demographics.

Problem Statement
The primary challenge addressed in this project is the need for an efficient health insurance claims prediction system. By leveraging Artificial Neural Networks, the goal is to construct a model that aids in assessing the likelihood of health insurance claims, providing valuable insights for risk management and resource allocation in the insurance industry.

Dataset
The dataset employed in this project comprises a structured set of features related to policyholders, including but not limited to age, gender, medical history, and policy specifics. The target variable indicates no.of claims for each policyholder based on patient ID.

Model Architecture
The predictive model is constructed using an Artificial Neural Network (ANN), a machine learning technique inspired by the human brain. The ANN architecture is tailored to the specific features of health insurance claims prediction, incorporating layers,dense layers, neurons,multi layer perceptron(MLP) and activation functions for optimal performance.

Feature Importance

To understand the impact of different features on the predicted health insurance claims amount, feature importance analysis is performed using the Random Forest Regressor. This analysis highlights which features contribute most significantly to the overall prediction, providing valuable insights into the factors influencing the monetary value of insurance claims.

Data Preparation
Prior to training the ANN model, data preprocessing tasks are undertaken, including handling missing values, encoding categorical variables, and standardizing numerical features. These steps ensure the dataset is well-structured and suitable for training the predictive model.

Training and Evaluation
The dataset is divided into training and testing sets to evaluate the model's generalization ability. The ANN is trained using the training set, and its performance is assessed using metrics such as accuracy, precision, recall, and F1 score. This rigorous evaluation ensures the model's effectiveness in predicting health insurance claims.

Usage
Environment Setup:

Ensure the required dependencies are installed using requirements.txt.
Python version: 3.x
Data Preparation:

Obtain the dataset and place it in the designated data directory.
Model Training:

Run the training script, e.g., train_model.py, to train the ANN.
Model Evaluation:

Evaluate the trained ANN using the evaluation script, e.g., evaluate_model.py.
Inference:

Implement the trained ANN for health insurance claims prediction in a production environment using the inference script, e.g., predict.py.
Results
The project aims to achieve a well-performing ANN model with high accuracy and precision in predicting health insurance claims. The detailed results, along with comparative analyses, will be documented in the project report.

Future Work
Future iterations of the project may involve:

Fine-tuning hyperparameters for optimizing the ANN model.
Exploring additional data sources to enhance predictive capabilities.
Consideration of model deployment for real-time predictions.
Acknowledgments
This project is developed as part of [Msc Data Science Course] at NMIMS University
