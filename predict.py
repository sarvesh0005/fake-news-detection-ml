import re
import string
import joblib

from pathlib import Path

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


# Download required NLTK resources (only the first time)
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)


# Load saved artifacts
artifact_path = Path("artifacts")

vectorizer = joblib.load(artifact_path / "tfidf_vectorizer.pkl")
model = joblib.load(artifact_path / "best_model.pkl")


# Initialize preprocessing tools
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))


def preprocess_text(text):
    """Clean the input text using the same preprocessing steps as training."""

    text = text.lower()

    text = re.sub(r"http\S+|www\S+", "", text)

    text = re.sub(r"<.*?>", "", text)

    text = text.translate(str.maketrans("", "", string.punctuation))

    text = re.sub(r"\d+", "", text)

    text = re.sub(r"\s+", " ", text).strip()

    words = text.split()

    words = [
        lemmatizer.lemmatize(word)
        for word in words
        if word not in stop_words
    ]

    return " ".join(words)


def predict_news(news):

    cleaned_news = preprocess_text(news)

    news_vector = vectorizer.transform([cleaned_news])

    prediction = model.predict(news_vector)[0]

    if prediction == 1:
        return "Fake News"
    else:
        return "Real News"


def main():

    print("=" * 50)
    print("        Fake News Detection System")
    print("=" * 50)

    while True:

        print("\nPaste a news article below:")
        news = input("\n> ")

        prediction = predict_news(news)

        print("\nPrediction:", prediction)

        choice = input("\nDo you want to test another article? (y/n): ").lower()

        if choice != "y":
            print("\nThank you for using the Fake News Detection System.")
            break


if __name__ == "__main__":
    main()