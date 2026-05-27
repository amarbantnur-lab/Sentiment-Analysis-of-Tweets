import numpy as np
import pandas as pd
import re
import pickle
import nltk

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Download stopwords
nltk.download('stopwords')

# Load stopwords
stop_words = set(stopwords.words('english'))

# Load dataset
dataset = pd.read_csv(
    "training.1600000.processed.noemoticon.csv",
    encoding='ISO-8859-1',
    header=None
)

# Add column names
col_names = ['target', 'id', 'date', 'flag', 'user', 'text']
dataset.columns = col_names

# Show first 5 rows
print(dataset.head())

# Dataset shape
print("Dataset Shape:", dataset.shape)

# Check missing values
print(dataset.isnull().sum())

# Target distribution
print(dataset['target'].value_counts())

# Convert 4 -> 1 (Positive)
dataset['target'] = dataset['target'].map({0: 0, 4: 1})

print(dataset['target'].value_counts())

# Initialize stemmer
stemmer = PorterStemmer()

# Stemming function
def stemming(content):
    content = str(content)

    # Remove non-alphabet characters
    stemmed_content = re.sub('[^a-zA-Z]', ' ', content)

    # Convert to lowercase
    stemmed_content = stemmed_content.lower()

    # Split words
    stemmed_content = stemmed_content.split()

    # Remove stopwords and stem
    stemmed_content = [
        stemmer.stem(word)
        for word in stemmed_content
        if word not in stop_words
    ]

    # Join words
    stemmed_content = ' '.join(stemmed_content)

    return stemmed_content

# Apply stemming
dataset['text'] = dataset['text'].apply(stemming)

print(dataset.head())

# Separate data and labels
X = dataset['text']
Y = dataset['target']

# Train-test split
X_train, X_test, Y_train, Y_test = train_test_split(
    X,
    Y,
    test_size=0.2,
    random_state=42
)

# Convert text to numerical data
vectorizer = TfidfVectorizer()

X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

print(X_train)

# Train model
model = LogisticRegression(max_iter=1000)

model.fit(X_train, Y_train)

# Predict
Y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(Y_test, Y_pred)

print("Accuracy:", accuracy)

# Prediction function
def predict_sentiment(text):

    text = re.sub('[^a-zA-Z]', ' ', text)

    text = text.lower()

    text = text.split()

    text = [
        stemmer.stem(word)
        for word in text
        if word not in stop_words
    ]

    text = ' '.join(text)

    text = [text]

    text = vectorizer.transform(text)

    sentiment = model.predict(text)

    if sentiment[0] == 0:
        return "Negative"
    else:
        return "Positive"

# Test predictions
print(predict_sentiment("I hate you"))
print(predict_sentiment("I love you"))

# Save model
with open('model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)

# Save vectorizer
with open('vectorizer.pkl', 'wb') as vectorizer_file:
    pickle.dump(vectorizer, vectorizer_file)

print("Model and vectorizer saved successfully!")