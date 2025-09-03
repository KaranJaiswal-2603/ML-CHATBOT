import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score , classification_report , confusion_matrix
data = pd.read_csv("data.csv")
X_text = data["symptoms"].fillna("")
y = data["disease"]
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(X_text)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print("Train data shape:", X_train.shape)
print("Test data shape:", X_test.shape)

model=LogisticRegression(max_iter=1000)
model.fit(X_train,y_train)
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

from sklearn.model_selection import cross_val_score
scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
print("Cross-validation accuracy scores:", scores)
print("Mean accuracy:", scores.mean())
import pickle

with open("Disease_model.pkl", "wb") as file_1:
    pickle.dump(model, file_1)

with open("tfidf_vectorizer.pkl", "wb") as file_2:
    pickle.dump(vectorizer, file_2)

print("\nmodel save done")