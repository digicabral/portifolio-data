import pandas as pd
import numpy as np
import dask.dataframe as dd
from dask.distributed import Client
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error
from fbprophet import Prophet
from fbprophet.diagnostics import cross_validation, performance_metrics
from fbprophet.plot import plot_cross_validation_metric
from math import sqrt
import itertools

# Load Data
def load_data():
    df = pd.read_excel(r'C:\Users\rodrigo.cabral\Documents\portifolio-data\d2g19\data\dataset.xlsx', header=0, index_col=0, parse_dates=True, squeeze=True, engine='openpyxl')
    return df

# Adjusting data
def adjust_data(dataframe):
    df = dataframe.filter(['demitido','ativos','admissoes','attrition','attrition_18a27','attrition_27a30','attrition_0a35','attrition_35a40','attrition_40mais'])
    # ordenando do mais antigo para o mais novo
    df = df.sort_index(axis=0)

    # Separating datasets
    split_point = int((len(df)/100)*80)
    df_train = df[0:split_point]
    df_test = df[split_point:]
    df_validation = df
    # write to disk
    df_train.to_csv('./data/df_train.csv')
    df_test.to_csv('./data/df_test.csv')
    # backup do df de treino
    df2 = df_train
    # Criando a coluna ds que é requisito obrigatório do prophet com as informações temporais e instanciando o modelo do prophet
    df3 = df_train.rename(columns={'attrition':'y'})
    df3['ds'] = df_train.index.values
    return df3

def modelling(df):
    print("Entrou na modelling")
    model = Prophet()
    # Adicionando as colunas do DataFrame aos regressores do modelo
    #Faço um for adicionando todas as colunas exceto a attrition e o y que é a target, e o ds que é a data
    for col in df.columns:
        if col not in["attrition","ds","y"]:
            model.add_regressor(col)

    model.fit(df)
    return model

def best_params(dataframe):
    # connect to the cluster
    client = Client()
    param_grid = {  
            'changepoint_prior_scale': [0.001, 0.01, 0.1, 0.5],
            'seasonality_prior_scale': [0.01, 0.1, 1.0, 10.0],
            }
    # Generate all combinations of parameters
    all_params = [dict(zip(param_grid.keys(), v)) for v in itertools.product(*param_grid.values())]
    rmses = []  # Store the RMSEs for each params here
    # Use cross validation to evaluate all parameters
    for params in all_params:
        # Fit model with given params
        m = Prophet(**params).fit(dataframe)  
        df_cv = cross_validation(m, horizon='90 days', parallel="dask")
        df_p = performance_metrics(df_cv, rolling_window=1)
        rmses.append(df_p['rmse'].values[0])
        # Find the best parameters
        tuning_results = pd.DataFrame(all_params)
        tuning_results['rmse'] = rmses
        print(tuning_results)

df = adjust_data(load_data())
best_params(df)