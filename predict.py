"""
predict.py
----------
Loads the trained model + vectorizer, and predicts sentiment
for a piece of text passed in from the command line.

Usage:
    python predict.py "This movie was amazing!"
"""

import sys
import pickle

# 1. Load the saved model and vectorizer
with open("model.pkl", "rb") as f:
    model, vectorizer = pickle.load(f)

# 2. Get the text from the command line
text = sys.argv[1]

# 3. Convert the text into TF-IDF numbers (same way training data was converted)
text_vec = vectorizer.transform([text])

# 4. Predict
prediction = model.predict(text_vec)[0]

# 5. Show the result
print(f"Text: {text}")
print(f"Prediction: {prediction}")