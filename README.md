# SMS Spam Classifier

Machine Learning project that classifies SMS messages as **Spam** or **Ham** using NLP techniques.

## Tech Stack
- Python, pandas, scikit-learn, NLTK
- TF-IDF Vectorization, Naive Bayes, Logistic Regression, Linear SVM
- joblib for model deployment

## Project Files: 
File 					              Purpose
`spam_classifier.py` 		    Initial model training
`save_model.py`			        Model + vectorizer saving
`use_saved_model.py` 		    Instant prediction from saved model
`compare_models.py` 	      3 models comparisongram_experiment.py` - Bigram accuracy improvement |

## Results: 
- Linear SVM achieved best F1-Score
- Bigrams (ngram_range=(1,2)) improved spam detection

##  How to Run: 
pip install pandas scikit-learn nltk joblib
python save_model.py
python use_saved_model.py
