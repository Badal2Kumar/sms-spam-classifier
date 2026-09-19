# SMS SPAM CLASSIFIER

import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score,
                             confusion_matrix, classification_report)


print("=" * 50)
print("STEP 1: Loading dataset...")
print("=" * 50)

df = pd.read_csv("spam.csv", encoding="latin-1")
df = df[['v1', 'v2']]
df.columns = ['label', 'message']

print(df.head())
print("\nDataset shape:", df.shape)

print("\n" + "=" * 50)
print("STEP 2: Exploring dataset...")
print("=" * 50)

print("\nTarget labels:", df['label'].unique())
print("\nClass distribution:\n", df['label'].value_counts())
print("\nMissing values:\n", df.isnull().sum())

print("Duplicates found:", df.duplicated().sum())
df = df.drop_duplicates()

df['msg_length'] = df['message'].apply(len)
print("\nMessage length stats:\n", df.groupby('label')['msg_length'].describe())
print("=" * 50)
print("STEP 3: Cleaning text data...")
print("=" * 50)

nltk.download('stopwords')

stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()                         
    text = re.sub(r'[^a-z\s]', '', text)         
    words = [stemmer.stem(w) for w in text.split() if w not in stop_words] 
    return ' '.join(words)

df['clean_message'] = df['message'].apply(clean_text)
print(df[['message', 'clean_message']].head())

print("\n" + "=" * 50)
print("STEP 4: Converting text to numerical features (TF-IDF)...")
print("=" * 50)

vectorizer = TfidfVectorizer(max_features=3000)
X = vectorizer.fit_transform(df['clean_message'])
y = df['label'].map({'ham': 0, 'spam': 1})

print("Feature matrix shape:", X.shape)

print("\n" + "=" * 50)
print("STEP 5: Splitting data (80% train, 20% test)...")
print("=" * 50)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Training samples:", X_train.shape[0])
print("Testing samples :", X_test.shape[0])

print("\n" + "=" * 50)
print("STEP 6: Training Multinomial Naive Bayes model...")
print("=" * 50)

model = MultinomialNB()
model.fit(X_train, y_train)
print("Model trained successfully!")

print("\n" + "=" * 50)
print("STEP 7 & 8: Predicting and evaluating...")
print("=" * 50)

y_pred = model.predict(X_test)

print("Accuracy :", round(accuracy_score(y_test, y_pred), 4))
print("Precision:", round(precision_score(y_test, y_pred), 4))
print("Recall   :", round(recall_score(y_test, y_pred), 4))
print("F1-Score :", round(f1_score(y_test, y_pred), 4))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\n", classification_report(y_test, y_pred, target_names=['Ham', 'Spam']))

print("\n" + "=" * 50)
print("STEP 9: Testing with custom messages...")
print("=" * 50)

def predict_message(text):
    cleaned = clean_text(text)
    features = vectorizer.transform([cleaned])
    pred = model.predict(features)[0]
    return "SPAM" if pred == 1 else "HAM"

test_messages = [
    "Hey, are we still meeting for lunch tomorrow?",
"Free recharge karne ke liye is link par click karo!!!",
"Mummy bula rahi hai, ghar jaldi aao",
"Bhai dekh mast offer aaya hai jaldi se iss link pr click karo or claim kro"
]

for msg in test_messages:
    print(f"\nMessage: {msg}")
    print(f"Prediction: {predict_message(msg)}")

print("\nDone! ✅")