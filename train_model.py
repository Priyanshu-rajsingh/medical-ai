import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle

df = pd.read_csv("datasets/Training.csv")

X = df.iloc[:, :-1]
y = df.iloc[:, -1]

model = RandomForestClassifier()

model.fit(X, y)

pickle.dump(model, open("models/disease_model.pkl","wb"))

print("Model trained successfully")