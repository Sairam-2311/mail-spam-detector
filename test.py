import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score

data = pd.read_csv("emails.csv")

X = data["text"]
y = data["label"]
model = make_pipeline(
    TfidfVectorizer(),
    MultinomialNB()
)
scores = cross_val_score(
    model,
    X,
    y,
    cv=5
)
print("Scores:", scores)
print("Average Accuracy:", scores.mean())