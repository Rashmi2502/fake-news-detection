# Fake News Detection System

## Overview

This project is a Fake News Detection System built using Natural Language Processing (NLP) and Machine Learning. The system analyzes news content and classifies it as **REAL** or **FAKE** based on learned patterns from the dataset.

---

## Features

- Text preprocessing using NLP techniques
- TF-IDF vectorization for feature extraction
- Multiple machine learning models:
  - Logistic Regression
  - Naive Bayes
  - Support Vector Machine (SVM)
  - Random Forest
- Model comparison and evaluation
- Confusion Matrix and accuracy visualization
- Explainable AI using LIME
- Interactive user interface using Streamlit

---

## Technologies Used

- Python
- Pandas, NumPy
- Scikit-learn
- NLTK
- Matplotlib, Seaborn
- Streamlit
- LIME

---

## Project Structure

```
fake-news-detection/
│── src/
│   ├── main.py
│   ├── True.csv
│   └── Fake.csv
│── outputs/          (generated during execution)
│── README.md
│── requirements.txt
└── .gitignore
```

---

## How to Run

**1. Create a virtual environment (optional but recommended):**
```bash
python -m venv .venv
.venv\Scripts\activate
```

**2. Install dependencies:**
```bash
pip install -r requirements.txt
```

**3. Run the application:**
```bash
streamlit run src/main.py
```

**4. Open in browser:**
```
http://localhost:8501
```

---

## Output

- Prediction: **REAL** or **FAKE**
- Model accuracy comparison graph
- Confusion matrix visualization
- LIME-based explanation of predictions

---

## Conclusion

This project demonstrates how NLP and machine learning techniques can be used to detect fake news. It also provides interpretability using LIME, making predictions more transparent and understandable.

---

## Note

The dataset files (`True.csv` and `Fake.csv`) are placed inside the `src/` folder.