import numpy as np
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from flask import Flask, request, render_template

# Load NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Text Cleaning
def clean_text(text):
    if isinstance(text, str):
        # Remove special characters and punctuation
        text = re.sub(r'[^\w\s]', '', text)
        # Convert to lowercase
        text = text.lower()
        return text
    else:
        return ''
    
# Remove stopwords
stop_words = set(stopwords.words('english'))
def remove_stopwords(text):
    words = word_tokenize(text)
    filtered_words = [word for word in words if word not in stop_words]
    return ' '.join(filtered_words)

# Text Normalization (Lemmatization)
lemmatizer = WordNetLemmatizer()
def lemmatize_text(text):
    words = word_tokenize(text)
    lemmatized_words = [lemmatizer.lemmatize(word) for word in words]
    return ' '.join(lemmatized_words)


# Apply text cleaning to Review text column
def preprocess_text(review_text):
    cleaned_text = clean_text(review_text)
    cleaned_text = remove_stopwords(cleaned_text)
    normalized_text = lemmatize_text(cleaned_text)
    bow_features = bow_vectorizer.transform([normalized_text])
    return bow_features

# Load the model
import joblib
lr_model = joblib.load('lr_model.pkl')
# Load the CountVectorizer object
bow_vectorizer = joblib.load('vectorizer.pkl')


app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    # Get the input text from the request
    input_text = request.form['text']

    # Ensure that the CountVectorizer object is fitted with training data
    bow_features = preprocess_text(input_text)

    # Make predictions
    predictions = lr_model.predict(bow_features)

    return render_template('results.html',result = str(predictions[0]))

if __name__ == '__main__':
    app.run(host='0.0.0.0')
