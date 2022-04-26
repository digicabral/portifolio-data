import pandas as pd
import dask as dk
from sklearn.model_selection import train_test_split, GridSearchCV
from fbprophet import Prophet
from math import sqrt
from pathlib import Path

#Load Data
df = pd.read_excel(r'C:\Users\rodrigo.cabral\Documents\portifolio-data\d2g19\data\dataset.xlsx', header=0, index_col=0, parse_dates=True, squeeze=True, engine='openpyxl')

#Adjusting data
df = df.filter(['demitido','ativos','admissoes','attrition','attrition_18a27','attrition_27a30','attrition_0a35','attrition_35a40','attrition_40mais'])

#Separating datasets
split_point = int((len(df)/100)*80)
df_train = df[0:split_point]
df_test = df[split_point:]
df_validation = df

# write to disk
df_train.to_csv('./data/df_train.csv')
df_test.to_csv('./data/df_test.csv')