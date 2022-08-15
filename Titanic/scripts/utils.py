"Funções utilizadas nos .ipynb"

from pathlib import Path
import pandas as pd

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