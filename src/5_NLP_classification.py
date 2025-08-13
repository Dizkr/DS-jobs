import pandas as pd
import string
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix

from langdetect import detect
from langdetect.lang_detect_exception import LangDetectException

import stopwords
from src.utils import load_latest_file

DATA_DIR = "data/processed"
MODEL_PATH = "models/job_classifier_pipeline-LR.pkl"
TITLE_MAX_FEATURES = 100
DESC_MAX_FEATURES = 2000
RANDOM_STATE = 20
LOGREG_MAX_ITER = 1000
THRESHOLD_DEFAULT = 0.5

# Stopwords
english_stopwords = set(stopwords.get_stopwords('english')).union(
    set(TfidfVectorizer(stop_words='english').get_stop_words())
)
polish_stopwords = set(stopwords.get_stopwords('polish'))
additional_stopwords = set([
    "oraz", "english", "hands", "znajomość", "pracy",
    "experience", "skills", "doświadczenie", "rozwiązań",
    "modeli", "role", "years"
])
combined_stopwords = english_stopwords.union(polish_stopwords).union(additional_stopwords)

# Custom Transformers

class LanguageDetector(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        def safe_detect(text):
            try:
                return detect(text) if isinstance(text, str) and text.strip() else "unknown"
            except LangDetectException:
                return "unknown"

        X = X.copy()
        X['is_polish'] = X['Job Description'].apply(lambda txt: 1 if safe_detect(txt) == 'pl' else 0)
        return X

class TextCleaner(BaseEstimator, TransformerMixin):
    def __init__(self, stopwords):
        self.stopwords = stopwords

    def fit(self, X, y=None):
        return self

    def clean_text(self, text):
        if not isinstance(text, str):
            return ""
        # Remove punctuation
        nopunc = ''.join(char for char in text if char not in string.punctuation)
        # Remove stopwords
        filtered_words = [word for word in nopunc.split() if word.lower() not in self.stopwords]
        return ' '.join(filtered_words)

    def transform(self, X):
        X = X.copy()
        X['Job Title'] = X['Job Title'].apply(self.clean_text)
        X['Job Description'] = X['Job Description'].apply(self.clean_text)
        return X

# Feature extraction 
combined_features = ColumnTransformer([
    ('title', TfidfVectorizer(max_features=TITLE_MAX_FEATURES), 'Job Title'),
    ('desc', TfidfVectorizer(max_features=DESC_MAX_FEATURES), 'Job Description'),
    ('lang', 'passthrough', ['is_polish'])
])

# Build pipeline
pipeline = Pipeline([
    ('lang_detect', LanguageDetector()),
    ('text_cleaning', TextCleaner(combined_stopwords)),
    ('features', combined_features),
    ('clf', LogisticRegression(max_iter=LOGREG_MAX_ITER))
])

# Load data
jobs_df = load_latest_file("justjoinit_jobs", DATA_DIR)

# Split features & labels
X = jobs_df[['Job Title', 'Job Description']]
y = jobs_df['Category']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, random_state=RANDOM_STATE
)

# Drop "Unclassified" from training set only
train_mask = y_train != "Unclassified"
X_train = X_train[train_mask]
y_train = y_train[train_mask]

# --- Train model ---
model = pipeline.fit(X_train, y_train)

joblib.dump(pipeline, MODEL_PATH)

pred = pipeline.predict(X_test)

cm = confusion_matrix(y_test, pred, labels=model.classes_)

plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=model.classes_, yticklabels=model.classes_)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix Heatmap')
plt.show()

def threshold_predict(model, X, threshold=THRESHOLD_DEFAULT):
    probs = model.predict_proba(X)
    class_preds = model.classes_[np.argmax(probs, axis=1)]
    max_probs = np.max(probs, axis=1)

    # Replace low-confidence predictions with "Unclassified"
    final_preds = [pred if prob >= threshold else "Unclassified"
                   for pred, prob in zip(class_preds, max_probs)]
    
    return final_preds

jobs_df = load_latest_file("justjoinit_jobs", DATA_DIR)

# Make sure the manual classification column exists
assert 'Category_manual' in jobs_df.columns, "Manual classification missing"

X_full = jobs_df[['Job Title', 'Job Description']]
y_full = jobs_df['Category_manual']

X_full = X_test
y_full = y_test

pred = model.predict(X_full)

thresholded_preds = threshold_predict(model, X_full, threshold=0.5)

y_full = pd.Series(y_full)
thresholded_preds = pd.Series(thresholded_preds)

y_full = pd.Series(y_test).reset_index(drop=True)
thresholded_preds = pd.Series(thresholded_preds).reset_index(drop=True)

classified_mask = (y_full != "Unclassified") & (thresholded_preds != "Unclassified")

print("Metrics on classified samples only:")
print(classification_report(y_full[classified_mask], thresholded_preds[classified_mask]))

plt.figure(figsize=(8,6))
cm = confusion_matrix(y_full[classified_mask], thresholded_preds[classified_mask], labels=model.classes_)
sns.heatmap(cm, annot=True, fmt='d', xticklabels=model.classes_, yticklabels=model.classes_)
plt.title("Confusion matrix on classified samples only")
plt.show()

unclassified_mask_true = (y_full == "Unclassified")
unclassified_mask_pred = (thresholded_preds == "Unclassified")

true_positives = sum(unclassified_mask_true & unclassified_mask_pred)
false_positives = sum(~unclassified_mask_true & unclassified_mask_pred)
false_negatives = sum(unclassified_mask_true & ~unclassified_mask_pred)

print("Unclassified overlap:")
print(f"True Positives (correctly unclassified): {true_positives}")
print(f"False Positives (predicted unclassified but actually classified): {false_positives}")
print(f"False Negatives (manual unclassified but predicted classified): {false_negatives}")

