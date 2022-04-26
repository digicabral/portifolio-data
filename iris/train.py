#Criação do ambiente virtual python -m venv .env
import pickle
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier

data = load_iris()

X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.DataFrame(data.target, columns=["target"])

model = DecisionTreeClassifier(max_depth=2)
model.fit(X,y)

with open("trained_classifier.pkl","wb") as file:
    pickle.dump(model, file)