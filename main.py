import pandas as pd
from src.preprocess import clean_text
from src.features import get_tfidf_features

df = pd.read_csv("data/social_data.csv")

df['clean_text'] = df['text'].apply(clean_text)

X, vectorizer = get_tfidf_features(df['clean_text'])

print("Shape of TF-IDF matrix:", X.shape)