import pickle
from preprocess import clean_text

model = pickle.load(open("models/model.pkl", "rb"))
vectorizer = pickle.load(open("models/vectorizer.pkl", "rb"))

def predict_sentiment(text):
    # clean input
    cleaned_text = clean_text(text)

    # convert to TF-IDF
    vector = vectorizer.transform([cleaned_text])

    # predict
    prediction = model.predict(vector)[0]

    return prediction


if __name__ == "__main__":
    while True:
        user_input = input("\nEnter text (or type 'exit'): ")

        if user_input.lower() == "exit":
            break

        result = predict_sentiment(user_input)
        print("Predicted Sentiment:", result)