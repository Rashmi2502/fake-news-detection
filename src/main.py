# --------------------------------------------
# FAKE NEWS DETECTION SYSTEM - MAJOR PROJECT
# Using:
# - NLP
# - Multiple ML Models
# - Streamlit UI
# - LIME Explainable AI
# - Matplotlib Visualizations
# --------------------------------------------

import pandas as pd
import numpy as np
import os
import string
import nltk
import streamlit as st
import matplotlib

# Prevent GUI backend issues
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import seaborn as sns

from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# ML Models
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier

# Explainable AI
from lime.lime_text import LimeTextExplainer

# --------------------------------------------
# DOWNLOAD NLTK DATA
# --------------------------------------------

nltk.download('stopwords')

# --------------------------------------------
# CREATE OUTPUT FOLDER
# --------------------------------------------

if not os.path.exists("outputs"):
    os.makedirs("outputs")

# --------------------------------------------
# TEXT CLEANING
# --------------------------------------------

stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = str(text).lower()

    # Remove punctuation
    text = ''.join(
        [char for char in text if char not in string.punctuation]
    )

    # Remove stopwords
    words = text.split()

    words = [
        word for word in words
        if word not in stop_words
    ]

    return " ".join(words)

# --------------------------------------------
# LOAD DATASETS
# --------------------------------------------

print("Loading datasets...")

true_data = pd.read_csv("src/True.csv")
fake_data = pd.read_csv("src/Fake.csv")

# Labels
true_data['label'] = 1
fake_data['label'] = 0

# Combine datasets
data = pd.concat([true_data, fake_data], axis=0)

# Remove duplicates
data = data.drop_duplicates()

# Shuffle data
data = data.sample(frac=1, random_state=42).reset_index(drop=True)

# Limit dataset size (optional)
data = data.sample(n=10000, random_state=42)

# Combine title + text
data['content'] = data['title'] + " " + data['text']

# Clean text
data['content'] = data['content'].apply(clean_text)

# --------------------------------------------
# FEATURE EXTRACTION
# --------------------------------------------

vectorizer = TfidfVectorizer(max_features=5000)

X = vectorizer.fit_transform(data['content'])
y = data['label']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# --------------------------------------------
# MACHINE LEARNING MODELS
# --------------------------------------------

models = {
    "Logistic Regression": LogisticRegression(),
    "Naive Bayes": MultinomialNB(),
    "SVM": LinearSVC(),
    "Random Forest": RandomForestClassifier(n_estimators=100)
}

results = {}

print("\nTraining Models...\n")

for name, model in models.items():

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    pre = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    results[name] = {
        "model": model,
        "accuracy": acc,
        "precision": pre,
        "recall": rec,
        "f1": f1,
        "y_pred": y_pred
    }

    print(f"{name} Accuracy: {acc:.2f}")

# --------------------------------------------
# BEST MODEL
# --------------------------------------------

best_model_name = max(
    results,
    key=lambda x: results[x]["accuracy"]
)

best_model = results[best_model_name]["model"]

print(f"\nBest Model: {best_model_name}")

# --------------------------------------------
# VISUALIZATIONS
# --------------------------------------------

# 1. Accuracy Comparison Graph

model_names = list(results.keys())

accuracies = [
    results[m]["accuracy"]
    for m in model_names
]

plt.figure(figsize=(8, 5))

plt.bar(model_names, accuracies)

plt.title("Model Accuracy Comparison")

plt.ylabel("Accuracy")

plt.xticks(rotation=20)

plt.tight_layout()

plt.savefig("outputs/accuracy_comparison.png")

plt.close()

# --------------------------------------------
# 2. Precision / Recall / F1 Graph
# --------------------------------------------

precision = [
    results[m]["precision"]
    for m in model_names
]

recall = [
    results[m]["recall"]
    for m in model_names
]

f1_scores = [
    results[m]["f1"]
    for m in model_names
]

x = np.arange(len(model_names))

plt.figure(figsize=(10, 5))

plt.bar(x - 0.2, precision, width=0.2, label='Precision')

plt.bar(x, recall, width=0.2, label='Recall')

plt.bar(x + 0.2, f1_scores, width=0.2, label='F1 Score')

plt.xticks(x, model_names, rotation=20)

plt.legend()

plt.title("Performance Metrics Comparison")

plt.tight_layout()

plt.savefig("outputs/metrics.png")

plt.close()

# --------------------------------------------
# 3. Confusion Matrix
# --------------------------------------------

cm = confusion_matrix(
    y_test,
    results[best_model_name]["y_pred"]
)

plt.figure(figsize=(6, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt='d'
)

plt.title(f"Confusion Matrix - {best_model_name}")

plt.xlabel("Predicted")

plt.ylabel("Actual")

plt.tight_layout()

plt.savefig("outputs/confusion_matrix.png")

plt.close()

print("Graphs saved in outputs folder")

# --------------------------------------------
# LIME EXPLAINABLE AI
# --------------------------------------------

explainer = LimeTextExplainer(
    class_names=["Fake", "Real"]
)

def predict_proba(texts):

    cleaned = [
        clean_text(t)
        for t in texts
    ]

    vectorized = vectorizer.transform(cleaned)

    try:
        return best_model.predict_proba(vectorized)

    except:

        preds = best_model.decision_function(vectorized)

        return np.vstack([
            1 - preds,
            preds
        ]).T

# --------------------------------------------
# PREDICTION FUNCTION
# --------------------------------------------

def predict_news(news_text):

    cleaned = clean_text(news_text)

    vectorized = vectorizer.transform([cleaned])

    prediction = best_model.predict(vectorized)[0]

    label = (
        "REAL NEWS ✅"
        if prediction == 1
        else "FAKE NEWS ❌"
    )

    # LIME Explanation

    exp = explainer.explain_instance(
        news_text,
        predict_proba,
        num_features=10
    )

    explanation = exp.as_list()

    # Save explanation as HTML

    exp.save_to_file(
        "outputs/lime_explanation.html"
    )

    return label, explanation

# --------------------------------------------
# STREAMLIT USER INTERFACE
# --------------------------------------------

st.title("📰 Fake News Detection System")

st.subheader("AI Powered Fake News Classifier")

st.write(f"✅ Best Model: {best_model_name}")

st.write("Enter a news article below:")

# User Input

user_input = st.text_area(
    "News Article",
    height=250
)

# --------------------------------------------
# BUTTON
# --------------------------------------------

if st.button("Check News"):

    if user_input.strip() == "":

        st.warning("Please enter news text!")

    else:

        label, explanation = predict_news(user_input)

        # Prediction
        st.subheader("Prediction Result")

        st.success(label)

        # Explainable AI
        st.subheader("Why this Prediction?")

        for word, score in explanation:

            st.write(f"{word} : {score:.3f}")

# --------------------------------------------
# DISPLAY SAVED GRAPHS
# --------------------------------------------

st.subheader("Model Accuracy Comparison")

st.image("outputs/accuracy_comparison.png")

st.subheader("Performance Metrics")

st.image("outputs/metrics.png")

st.subheader("Confusion Matrix")

st.image("outputs/confusion_matrix.png")