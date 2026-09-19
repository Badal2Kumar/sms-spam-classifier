import pandas as pd
import re
import nltk
import joblib
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

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

# --- TF-IDF + Train-Test Split ---
vectorizer = TfidfVectorizer(max_features=3000)
X = vectorizer.fit_transform(df['clean_message'])
y = df['label'].map({'ham': 0, 'spam': 1})

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = MultinomialNB()
model.fit(X_train, y_train)
print("Model trained successfully!")

joblib.dump(model, 'spam_model.pkl')
print("Model saved: spam_model.pkl")

joblib.dump(vectorizer, 'vectorizer.pkl')
print("Vectorizer saved: vectorizer.pkl")

print("\nDone! Ab aapke folder mein 2 nayi .pkl files ban gayi hain.")