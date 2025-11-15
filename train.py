import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib
import os
import json


os.makedirs('model', exist_ok=True)

with open('params.json', 'r') as f:
    params = json.load(f)

poison_level = params["poison_level"]
del params["poison_level"]

file = "data/iris.csv"

if poison_level == 5:
    file = "data/poison_5/iris_poisoned.csv"
elif poison_level == 10:
    file =  "data/poison_10/iris_poisoned.csv"
elif poison_level == 50:
    file =  "data/poison_50/iris_poisoned.csv"

print("data file", file)
df = pd.read_csv(file)

X = df.drop('species', axis=1)
y = df['species']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



model = LogisticRegression(**params)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

joblib.dump(model, 'model/model.joblib')

with open('metrics.txt', 'w') as f:
    f.write(f'Accuracy: {accuracy}\n')

print(f"Model trained with accuracy: {accuracy}")
print("Model saved to model/model.joblib")
print("Metrics saved to metrics.txt")
