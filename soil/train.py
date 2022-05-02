"""
-------------------------------------Exercício 2 - Pipelines II:------------------------------------------------

 Utilizando o VSCode, crie um ambiente virtual com as dependências (bibliotecas com as devidas versões) do exercício 
desenvolvido anteriormente (Regressão Logística). Isto feito, salve o pickle do modelo no diretório models utilizando a
biblioteca joblib.

Passos:
1) Crie um ambiente vitual a partir do arquivo requirements.txt;
    Execute os seguintes comandos no terminal:

        1.1) Instalação:
            pip install virtualenv 

        1.2) Criação:
            virtualenv <nome_do_ambiente>

        1.3) Ativação
            1.3.1) Windows
                ./venv/Scripts/activate.bat	       
            1.3.2) Mac, Linux
                source venv/bin/activate    

        1.4) Instalação de libs
            pip install -r requirements.txt

2) Insira o código do modelo de regressão logística desenvolvido no exercício anterior

2) Salve o pickle do modelo no diretório models. Dica: utilize -> joblib.dump(modelo, caminho)

4) Execute o arquivo train.py no terminal -> python exercicio.py



-----------------------------------------------------------------------------------------------------------------
"""


import joblib
import pandas as pd 
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split



train = pd.read_csv('data/train.csv')

X = train.drop(['Id','Cover_Type'], axis = 1)
y = train.Cover_Type

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

cols_to_drop=['Soil_Type26','Soil_Type27','Soil_Type17','Soil_Type5',
 'Soil_Type1',
 'Soil_Type33',
 'Soil_Type31',
 'Soil_Type13',
 'Soil_Type28',
 'Soil_Type25',
 'Soil_Type19',
 'Soil_Type4',
 'Soil_Type8',
 'Soil_Type37',
 'Soil_Type11',
 'Soil_Type30',
 'Soil_Type9',
 'Soil_Type15',
 'Soil_Type40',
 'Soil_Type29',
 'Soil_Type35',
 'Soil_Type22',
 'Soil_Type20',
 'Soil_Type36',
 'Soil_Type23',
 'Soil_Type12',
 'Soil_Type32',
 'Soil_Type2',
 'Soil_Type21',
 'Wilderness_Area2',
 'Soil_Type16',
 'Soil_Type34',
 'Soil_Type7',
 'Soil_Type18',
 'Soil_Type6',
 'Soil_Type14',
 'Soil_Type24']

class DropUnecessaryFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, variables_to_drop=None):
        self.variables = variables_to_drop

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X = X.drop(self.variables, axis=1)

        return X

class AddFeatures(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = X.copy()
        X['Ele_minus_VDtHyd'] = X.Elevation-X.Vertical_Distance_To_Hydrology
        X['Ele_plus_VDtHyd'] = X.Elevation+X.Vertical_Distance_To_Hydrology
        X['Distanse_to_Hydrolody'] = (X['Horizontal_Distance_To_Hydrology']**2+X['Vertical_Distance_To_Hydrology']**2)**0.5
        X['Hydro_plus_Fire'] = X['Horizontal_Distance_To_Hydrology']+X['Horizontal_Distance_To_Fire_Points']
        X['Hydro_minus_Fire'] = X['Horizontal_Distance_To_Hydrology']-X['Horizontal_Distance_To_Fire_Points']
        X['Hydro_plus_Road'] = X['Horizontal_Distance_To_Hydrology']+X['Horizontal_Distance_To_Roadways']
        X['Hydro_minus_Road'] = X['Horizontal_Distance_To_Hydrology']-X['Horizontal_Distance_To_Roadways']
        X['Fire_plus_Road'] = X['Horizontal_Distance_To_Fire_Points']+X['Horizontal_Distance_To_Roadways']
        X['Fire_minus_Road'] = X['Horizontal_Distance_To_Fire_Points']-X['Horizontal_Distance_To_Roadways']
        return X

lr_pipe=Pipeline(steps=[
    ('add_features', AddFeatures()),
    ('drop_features', DropUnecessaryFeatures(cols_to_drop)),
    ('scaler', MinMaxScaler()),
    ('classifier', LogisticRegression(random_state=42,max_iter=1000))
    ])

lr_pipe.fit(X_train, y_train)

joblib.dump(lr_pipe, "models/pipeline_model.pkl")