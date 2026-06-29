import pandas as pd
import re
import nltk

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
df = pd.read_csv("Resume.csv")

# Keep required columns
df = df[['Resume_str', 'Category']]

# Remove null values
df.dropna(inplace=True)

# Text cleaning
def clean_text(text):
    text = str(text).lower()

    text = re.sub(r'http\S+', ' ', text)
    text = re.sub(r'www\S+', ' ', text)
    text = re.sub(r'@\S+', ' ', text)
    text = re.sub(r'#[A-Za-z0-9_]+', ' ', text)
    text = re.sub(r'[^a-zA-Z ]', ' ', text)
    text = re.sub(r'\s+', ' ', text)

    return text

df['Resume_str'] = df['Resume_str'].apply(clean_text)

# Features and Labels
X = df['Resume_str']
y = df['Category']

# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# TF-IDF + SVM Pipeline
model = Pipeline([
    ('tfidf', TfidfVectorizer(
        stop_words='english',
        max_features=5000
    )),
    ('clf', LinearSVC())
])

# Train
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
#save model
import joblib
joblib.dump(model, "resume_classifier.pkl")
print("Model Saved")
# Load & Predict
import joblib
model = joblib.load("resume_classifier.pkl")
resume_text = """
Python
Machine Learning
Deep Learning
TensorFlow
Data Analysis
SQL
"""
prediction = model.predict([resume_text])

print("Predicted Category:", prediction[0])