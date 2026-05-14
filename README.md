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

The primary objective of the project was to develop a system capable of detecting fraudulent transactions with high accuracy. The project also focused to determine which model performed best on the imbalanced fraud dataset. Beyond model development, the project aimed to create a backend API for serving predictions and a frontend interface that allows users to interact with the model easily.

Another major objective was to analyze transaction behavior patterns and understand the structure of the dataset through exploratory analysis.

## Project Workflow

The project started with dataset collection and inspection. The dataset was first analyzed to understand its structure, transaction distribution, and the relationships withing the dataset. Exploratory Data Analysis was carried out to identify duplicate records, check missing values, and visualize patterns in the dataset.

After the EDA phase, preprocessing was performed to clean the dataset and prepare it for machine learning training. The cleaned data was then divided into training and testing sets before multiple machine learning models were trained and evaluated.

The best-performing models were identified, and was integrated into a FastAPI backend system. The backend handled prediction requests and communicated with the Streamlit frontend interface allowing users to interact with the fraud detection system in real time.

## EDA and Pre-Processing Workflow

The dataset was inspected using Pandas functions such as 
'''python
df.head()

.head(), .info(), and .describe() to understand the number of rows, columns, feature distributions, and data types.

One of the most important observations made during the analysis was the severe class imbalance within the dataset. Fraudulent transactions represented only a very small percentage of the total transactions, while legitimate transactions dominated the dataset. This observation influenced many later decisions in the workflow, especially during model evaluation and training.

The EDA also revealed the presence of duplicate records. About 1081 duplicate rows were identified during the analysis stage. These duplicate values could negatively affect the learning process by causing bias in the models. To solve this issue, the duplicate records were removed before preprocessing and training.

Another important observation was that the dataset contained no missing values. This simplified preprocessing and improved confidence in the quality of the dataset.


## Team
- Praise (Group Lead) — Backend / FastAPI
- Patrick — EDA & Preprocessing
- Paul — Model Training & Evaluation
- Blessing — Streamlit Frontend
- Francis — Documentation & Testing

## Tech Stack
- Python, Scikit-learn, XGBoost, FastAPI, Streamlit
- Dataset: Credit Card Fraud Detection (Kaggle)
