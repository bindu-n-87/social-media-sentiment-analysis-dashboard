import pandas as pd
import random

def generate_data(n=500):
    positive_texts = [
        "I love this product",
        "Amazing experience with the service",
        "Really happy with the quality",
        "This is excellent and works perfectly",
        "Super satisfied with everything",
        "Best purchase I have made",
        "Highly recommend this to everyone",
        "Very impressed with the performance",
        "Absolutely fantastic product",
        "Works better than expected"
    ]

    negative_texts = [
        "Worst experience ever",
        "I hate this product",
        "Very disappointed with the service",
        "It stopped working after one day",
        "Completely useless and waste of money",
        "Very poor quality",
        "Not worth buying at all",
        "Extremely frustrating experience",
        "Totally broken and useless",
        "Bad customer support and service"
    ]

    neutral_texts = [
        "It is okay nothing special",
        "Average experience overall",
        "Works as expected",
        "Not bad not great",
        "It is fine for basic use",
        "Decent product for the price",
        "Neither good nor bad",
        "Standard performance nothing extra",
        "It does the job",
        "Acceptable quality"
    ]

    data = []

    for _ in range(n):
        label = random.choice(["positive", "negative", "neutral"])

        if label == "positive":
            text = random.choice(positive_texts)
        elif label == "negative":
            text = random.choice(negative_texts)
        else:
            text = random.choice(neutral_texts)

        data.append([text, label])

    df = pd.DataFrame(data, columns=["text", "sentiment"])
    df.to_csv("data/social_data.csv", index=False)

    print(f"Dataset generated successfully with {n} rows")

if __name__ == "__main__":
    generate_data()