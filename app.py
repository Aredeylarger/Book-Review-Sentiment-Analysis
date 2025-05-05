
import streamlit as st
import joblib
import os

# Define paths to model and vectorizer
script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, "logistic_regression_model.pkl")
vectorizer_path = os.path.join(script_dir, "vectorizer.pkl")

# Check if files exist before loading
if os.path.exists(model_path) and os.path.exists(vectorizer_path):
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    st.write("Model & vectorizer loaded successfully!")
else:
    st.error(f"File not found. Model path: {model_path}, Vectorizer path: {vectorizer_path}")

# Function to predict sentiment
def predict_sentiment(text):
    cleaned_text = text
    transformed_text = vectorizer.transform([cleaned_text])
    prediction = model.predict(transformed_text)
    return prediction[0].upper()

# Streamlit UI
st.title("Sentiment Analysis Web App")
st.write("Enter a review below to analyze its sentiment.")

# User input text
user_input = st.text_area("Enter your text here:")

# Predict sentiment on button click
if st.button("Analyze Sentiment"):
    if user_input.strip():
        result = predict_sentiment(user_input)
        st.write(f"### Sentiment: {result}")
    else:
        st.warning("Please enter some text to analyze.")
