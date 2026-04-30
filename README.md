# Social Media Sentiment Analysis Dashboard

## Overview
This project is an end-to-end Machine Learning system that performs sentiment analysis on social media-style text data. It classifies user input into three categories: Positive, Negative, and Neutral, and visualizes insights through an interactive Streamlit dashboard.

The system demonstrates a complete NLP pipeline including data preprocessing, feature extraction, model training, evaluation, and deployment.

---

## Problem Statement
Social media platforms generate large volumes of user-generated text daily. Manually analyzing this data is inefficient and time-consuming. This project automates sentiment classification to help understand public opinion, customer feedback, and brand perception.

---

## Objectives
- Build a sentiment classification model using machine learning
- Preprocess and clean text data for NLP tasks
- Extract meaningful features using TF-IDF
- Train a classification model for sentiment prediction
- Visualize insights using an interactive dashboard
- Enable real-time sentiment prediction for user input

---

## Tech Stack
- Python
- Pandas, NumPy
- NLTK (Natural Language Processing)
- Scikit-learn (Machine Learning)
- TF-IDF Vectorization
- Matplotlib, Seaborn (Visualization)
- Streamlit (Web Dashboard)

---

## Project Workflow
1. Data Collection (Synthetic social media dataset)
2. Text Preprocessing (cleaning, normalization, stopword removal)
3. Feature Extraction (TF-IDF vectorization)
4. Model Training (Logistic Regression classifier)
5. Model Evaluation (accuracy, classification report, confusion matrix)
6. Deployment (Streamlit dashboard for real-time predictions)

---

## Project Structure
Social-Media-Sentiment-Analysis-Dashboard/
│
├── data/ # Dataset files
├── src/ # Preprocessing and ML pipeline
├── app/ # Streamlit dashboard
├── models/ # Trained model and vectorizer
├── outputs/ # Evaluation results (confusion matrix)
├── images/ # Screenshots for documentation
├── requirements.txt # Dependencies
├── README.md # Project documentation
└── .gitignore # Ignored files

---

## Installation and Setup

### 1. Clone the repository
git clone https://github.com/bindu-n-87/social-media-sentiment-analysis-dashboard.git
cd social-media-sentiment-analysis-dashboard

---

### 2. Create virtual environment
python -m venv venv

Activate environment:

**Windows**
venv\Scripts\activate

**Mac/Linux**
source venv/bin/activate

---

### 3. Install dependencies
pip install -r requirements.txt

---

## How to Run the Project

### Step 1: Generate Dataset
python data/generate_dataset.py

---

### Step 2: Train Model
python src/train_model.py

This will:
- Train the model
- Generate evaluation metrics
- Save trained model and vectorizer

---

### Step 3: Run Streamlit Dashboard
streamlit run app/streamlit_app.py

---

## Features
- Real-time sentiment prediction
- Clean and interactive dashboard
- Dataset visualization (bar chart, pie chart)
- Machine learning model evaluation
- Modular and scalable project structure

---

## Model Performance
- Algorithm: Logistic Regression
- Feature Engineering: TF-IDF
- Evaluation Metrics: Accuracy, Precision, Recall, F1-score
- Visualization: Confusion Matrix

---

## Example Output
Input: I love this product, it is amazing

Output: Positive Sentiment

---

## Future Improvements
- Integration with real Twitter API data
- Use of deep learning models (LSTM, BERT)
- Time-series sentiment trend analysis
- Deployment on cloud (Streamlit Cloud / AWS)
- Advanced dashboard analytics

---

## Learning Outcomes
- Practical NLP pipeline development
- Machine learning model building
- Feature engineering using TF-IDF
- Streamlit dashboard development
- End-to-end project deployment workflow

---

## Author
Bindu P

---

## License
This project is for educational purposes.
