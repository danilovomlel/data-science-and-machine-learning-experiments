"Funções utilizadas nos .ipynb"

import pandas as pd
from sklearn.linear_model import LogisticRegression
import pickle

IDADE_CLASSE = {1: 38.2, 2: 29.9, 3: 25.1}
def impute_age(cols):
    "Insere a idade baseado na média de idade por classe social"
    Age = cols[0]
    Pclass = cols[1]

    if pd.isna(Age):
        if Pclass == 1:
            return IDADE_CLASSE[1]
        elif Pclass == 2:
            return IDADE_CLASSE[2]
        else:
            return IDADE_CLASSE[3]
    else:
        return Age

def impute_age_model(cols) -> float:
    "Insere a idade baseado em modelo de regressão logística"
    AgeGroup = cols[0]
    X_features = {
        "Pclass": cols[1],
        "SibSp": cols[2],
        "Parch": cols[3]
    }    
    X_features = pd.DataFrame.from_dict([X_features])

    if pd.isna(AgeGroup):
        try:
            with open('../models/age_model.pickle', 'rb') as file:
                model = pickle.load(file)
        finally:
            pass
        AgeGroup = float(model.predict(X_features))
        return AgeGroup
    else:
        return AgeGroup

def fare_to_classgroup(fare) -> int:
    "Mapeia a classe social a partir da taxa paga"
    CLASSE_SOC = {3: [0, 50], 2: [50, 125], 1: [125, 1000]}
    class_group = 3
    for key, val in CLASSE_SOC.items():
        if fare < val[1] and fare >= val[0]:
            class_group = key
            break
    return class_group