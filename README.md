# Credit Card Fraud Detection System
End-to-End Machine Learning Project | Data Science Group Project

## Project Overview

The Credit Card Fraud Detection System was developed to address the growing challenge of identifying fraudulent financial transactions in real time. Fraudulent activities in digital payment systems continue to increase drastically, leading to financial losses, customer distrust, and greater risks for financial institutions.

This project combines Exploratory Data Analysis (EDA), machine learning model training, backend API development, and frontend deployment into a complete fraud detection workflow capable of analyzing transaction patterns and predicting suspicious activities.

The system was designed as a collaborative end-to-end machine learning solution using the following tools:

- Python
- Pandas
- Numpy
- Matplotlit
- Seaborn
- FastAPI
- Streamlit
- Scikit-learn
- XGBoost 

## Problem Statement

Financial institutions process millions of transactions daily. Detecting fraudulent transactions manually is inefficient due to.

- Extremely large transaction volumes
- Highly imbalanced fraud datasets
- Rapidly evolving fraud patterns
- Delayed fraud detection processes
- Difficulty identifying hidden transaction anomalies

## Project Objective

The objective of the project was to develop a system capable of detecting fraud transactions with high accuracy, And also to determine which model performed best on the imbalanced fraud dataset. Beyond model development, the project aimed to create a backend API for serving predictions and a frontend interface that allows users to interact with the model easily.

Another major objective was to analyze transaction behavior patterns and understand the structure of the dataset through exploratory analysis.

## Project Workflow

The project started with dataset collection. The dataset was first analyzed to understand its structure, transaction distribution, and the relationships within the dataset. Exploratory Data Analysis was carried out to identify duplicate, check missing values, and visualize patterns in the dataset.

After the EDA phase, preprocessing was performed to clean the dataset and prepare it for training. The cleaned data was then divided into training and testing sets before multiple machine learning models were trained and evaluated.

The best-performing models were identified, and was integrated into a FastAPI backend system. The backend handled prediction requests and communicated with the Streamlit frontend interface allowing users to interact with the fraud detection system in real time.

## EDA and Pre-Processing Workflow

The dataset was inspected using Pandas functions such as:
```python
.head(), .info(), .describe(), .duplicated().sum(), df['Class'].value_counts()
```
to understand the number of rows, columns,statistical summary, check duplication and distribution. One of the observations was class imbalance within the dataset. Fraudulent transactions represented only a very small percentage of the total transactions, while legitimate transactions dominated the dataset.

About 1081 duplicate rows were identified. These values could negatively affect the learning process so it has to be removed.

## Model Training Workflow

The model training focused on identifying the most effective machine learning algorithm for detecting fraudulent transactions. Multiple models were trained and compared to understand their predictive strengths and weaknesses.

- Logistic Regression was first used as a baseline model because of its simplicity and interpretability. The model trained quickly and provided a useful benchmark for comparing other algorithms. The observations from training showed that Logistic Regression couldn't handle highly complex fraud patterns.

- Random Forest Classifier was then introduced as an ensemble learning model. During training, the model demonstrated stronger classification capability and better handling of non-linear relationships within the dataset. The observations from the Random Forest model showed improved fraud detection performance and reduced overfitting risk.

- XGBoost was also trained because of its strong performance in classification tasks involving imbalanced datasets. The training process showed that XGBoost handled fraud detection effectively and improved prediction performance significantly. The use of scale_pos_weight helped address class imbalance during training.

- Gradient Boosting Classifier was trained as another ensemble learning approach. The model demonstrated strong learning capability and competitive performance, although it required more computational resources compared to simpler algorithms.

## Key Finding and Model Evaluation


1. Importance of Recall
Because fraud detection prioritizes identifying fraudulent transactions, recall became a critical metric.
Observation:
High recall reduces undetected fraud cases.

2. Precision vs Recall Balance
A balance between precision and recall was necessary to:
Reduce false positives
Maintain fraud detection sensitivity

3. Ensemble Models Performed Better
Random Forest and XGBoost demonstrated stronger fraud detection capability compared to the baseline model.
Reason:
Better ability to learn complex fraud behavior patterns.

4. Feature Importance Analysis
- Feature importance analysis helped identify:
- High-impact transaction variables
- Influential predictive features
- Key fraud indicators

## API Workflow (FastAPI Backend)


The FastAPI backend served as the communication bridge between the machine learning model and the frontend application. Its primary responsibility was to receive transaction data from users, process prediction requests, load the trained model, and return prediction responses.

The backend workflow started when transaction details were submitted from the frontend interface. The FastAPI endpoint received the request and passed the transaction data through the trained machine learning model. After analysis, the backend generated a prediction indicating whether the transaction was fraudulent or legitimate and returned the result to the frontend.



















## Team
- Praise (Group Lead) — Backend / FastAPI
- Patrick — EDA & Preprocessing
- Paul — Model Training & Evaluation
- Blessing — Streamlit Frontend
- Francis — Documentation & Testing

