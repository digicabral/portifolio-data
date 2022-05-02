import pandas as pd
import numpy as np
import joblib

df=pd.read_csv("data/winequality_predict.csv")

model=joblib.load("models/model_pipeline.pkl")

predictions = model.predict(df)

print(predictions)