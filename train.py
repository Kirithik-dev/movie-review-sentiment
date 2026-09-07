"""
train.py
--------
This script:
1. Loads the review data
2. Converts text into numbers (TF-IDF)
3. Trains a Logistic Regression model
4. Saves the trained model so predict.py can use it later
"""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle

# 1. Load the data
df = pd.read_csv("data/reviews.csv")
print(f"Loaded {len(df)} reviews")

X = df["text"]          # the review text
y = df["label"]         # positive / negative

# 2. Split into train and test sets (so we can check how well it learned)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3. Convert text into numbers using TF-IDF
# TF-IDF gives each word a score based on how important it is in a sentence
# vs how common it is across all sentences.
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# 4. Train the model
model = LogisticRegression()
model.fit(X_train_vec, y_train)

# 5. Check accuracy on the test set
predictions = model.predict(X_test_vec)
accuracy = accuracy_score(y_test, predictions)
print(f"Test accuracy: {accuracy:.2f}")

# 6. Save the model AND the vectorizer (we need both to predict later)
with open("model.pkl", "wb") as f:
    pickle.dump((model, vectorizer), f)

print("Model saved as model.pkl")