# sentiment_analysis.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn import metrics
import pickle

def train_model(data):
    X_train, X_test, y_train, y_test = train_test_split(data['Combined_News'], data['Label'], test_size=0.2, random_state=42)

    model = make_pipeline(CountVectorizer(), MultinomialNB())
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("Accuracy:", metrics.accuracy_score(y_test, y_pred))

    return model

def save_model(model, filename='sentiment_model.pkl'):
    with open(filename, 'wb') as file:
        pickle.dump(model, file)
    print(f"Model saved as {filename}")

def main():
    # Load your dataset (replace 'your_dataset.csv' with your actual file)
    data = pd.read_csv('Data.csv')

    # Assume your dataset has 'Combined_News' column for stock trade headlines and 'Label' column for sentiment (0 or 1)
    model = train_model(data)

    # Save the trained model
    save_model(model)

if __name__ == "__main__":
    main()
