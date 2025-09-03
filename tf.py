import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
data=pd.read_csv("data.csv")
X_text = data["symptoms"].fillna("")
y = data["disease"]
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(X_text)

print("X shape :", X.shape)
print("y shape :", y.shape)


