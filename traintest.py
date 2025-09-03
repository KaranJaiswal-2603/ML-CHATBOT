import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
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