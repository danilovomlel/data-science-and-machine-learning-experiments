from pathlib import Path
import pandas as pd

def get_project_root() -> Path:
    #root = Path(__file__).parent.parent
    root = "C:/Users/Pristino/Downloads/Projetos-prog/Kaggle/Titanic"
    return root

#Insere a idade baseado na média de idade por classe social
IDADE_CLASSE = {1: 38.2, 2: 29.9, 3: 25.1}
def impute_age(cols):
    Age = cols[0]
    Pclass = cols[1]
    
    if pd.isnull(Age):
        if Pclass == 1:
            return IDADE_CLASSE[1]
        elif Pclass == 2:
            return IDADE_CLASSE[2]
        else:
            return IDADE_CLASSE[3]
    else:
        return Age