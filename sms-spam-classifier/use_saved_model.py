# Using a saved model (without training).
# Running inference on a saved model.
import joblib
model = joblib.load('spam_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

print("Model loaded instantly! No training needed")

def predict_message(text):
    features = vectorizer.transform([text])
    pred = model.predict(features)[0]
    return "SPAM" if pred == 1 else "HAM"

messages = [
    "You won a free cruise! Call now!",
    "Where are you? Call me back.",
    "URGENT! Your mobile number has won $2000 prize. Claim now!",
    "Hi mummy, main ghar aa raha hoon."
]

for msg in messages:
    print(f"{msg}  -->  {predict_message(msg)}")

print("\nDone!")