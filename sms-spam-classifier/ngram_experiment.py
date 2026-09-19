import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
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

y = df['label'].map({'ham': 0, 'spam': 1})


settings = {
    "Unigrams only   ngram_range=(1,1)": (1, 1),
    "Uni+Bigrams     ngram_range=(1,2)": (1, 2),
}

print("=" * 55)
print("N-GRAM EXPERIMENT RESULTS")
print("=" * 55)

for name, ngram_range in settings.items():

    vectorizer = TfidfVectorizer(ngram_range=ngram_range)
    X = vectorizer.fit_transform(df['clean_message'])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = MultinomialNB()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"\n{name}:")
    print(f"  Features           : {X.shape[1]}")
    print(f"  Accuracy            : {acc:.4f}")
    print(f"  F1-Score            : {f1:.4f}")

print("\nLet's see if (1,2) performed better than (1,1)!")
