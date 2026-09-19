# 3 MODELS COMPARISON
# Naive Bayes vs Logistic Regression vs Linear SVM

import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score

df = pd.read_csv("spam.csv", encoding="latin-1")
df = df[['v1', 'v2']]
df.columns = ['label', 'message']
df = df.drop_duplicates()

nltk.download('stopwords')
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    words = [stemmer.stem(w) for w in text.split() if w not in stop_words]
    return ' '.join(words)

df['clean_message'] = df['message'].apply(clean_text)

vectorizer = TfidfVectorizer(max_features=3000)
X = vectorizer.fit_transform(df['clean_message'])
y = df['label'].map({'ham': 0, 'spam': 1})

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

models = {
    "Naive Bayes": MultinomialNB(),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Linear SVM": LinearSVC()
}

results = []

print("=" * 55)
print("MODEL COMPARISON RESULTS")
print("=" * 55)

for name, model in models.items():
    model.fit(X_train, y_train)              # train
    y_pred = model.predict(X_test)           # predict
    acc = accuracy_score(y_test, y_pred)     # accuracy
    f1 = f1_score(y_test, y_pred)            # f1-score

    results.append((name, acc, f1))
    print(f"\n{name}:")
    print(f"  Accuracy : {acc:.4f}")
    print(f"  F1-Score : {f1:.4f}")

print("\n" + "=" * 55)
best = max(results, key=lambda x: x[2])  # F1 ke hisaab se best
print(f"BEST MODEL: {best[0]} (F1 = {best[2]:4f})")
print("=" * 55)