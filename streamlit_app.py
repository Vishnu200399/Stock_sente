# streamlit_app.py

import streamlit as st
import pickle
import pandas as pd
import os

def load_model(filename='sentiment_model.pkl'):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, filename)
    
    with open(model_path, 'rb') as file:
        return pickle.load(file)

def main():
    st.title("Stock Trade Sentiment Analysis")

    # Load the trained model
    model = load_model()

    # Input text for sentiment analysis
    user_input = st.text_input("Enter a stock trade headline:")

    if st.button("Predict"):
        # Make prediction using the loaded model
        prediction = model.predict([user_input])[0]

        # Display the prediction
        st.write(f"Sentiment: {'Positive' if prediction == 1 else 'Negative'}")

if __name__ == "__main__":
    main()
