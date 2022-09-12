from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

from sklearn.feature_selection import mutual_info_regression
from sklearn.model_selection import cross_val_score
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
from category_encoders import MEstimateEncoder

TARGET = 'SalePrice'

five_levels = ["None", "Po", "Fa", "TA", "Gd", "Ex"]

ORDERED_LEVELS = {
    "ExterQual": five_levels,
    "ExterCond": five_levels,
    "BsmtQual": five_levels,
    "BsmtCond": five_levels,
    "HeatingQC": five_levels,
    "KitchenQual": five_levels,
    "FireplaceQu": five_levels,
    "GarageQual": five_levels,
    "GarageCond": five_levels,
    "PoolQC": five_levels,
    "LotShape": ["None", "IR3", "IR2", "IR1", "Reg"],
    "LandSlope": ["None", "Sev", "Mod", "Gtl"],
    "BsmtExposure": ["None", "No", "Mn", "Av", "Gd"],
    "BsmtFinType1": ["None", "Unf", "LwQ", "Rec", "BLQ", "ALQ", "GLQ"],
    "BsmtFinType2": ["None", "Unf", "LwQ", "Rec", "BLQ", "ALQ", "GLQ"],
    "Functional": ["None", "Sal", "Sev", "Maj1", "Maj2", "Mod", "Min2", "Min1", "Typ"],
    "GarageFinish": ["None", "Unf", "RFn", "Fin"],
    "PavedDrive": ["None", "N", "P", "Y"],
    "Utilities": ["None", "ELO", "NoSeWa", "NoSewr", "AllPub"],
    "CentralAir": ["None", "N", "Y"],
    "Electrical": ["None", "Mix", "FuseP", "FuseF", "FuseA", "SBrkr"],
    "Fence": ["None", "MnWw", "GdWo", "MnPrv", "GdPrv"],
}

CLUSTER_FEATURES = [
    "Spaciousness", #13
]

# PCA_FEATURES = [
#     "Foundation_NOM",
#     "BsmtQual_ORD",
#     "ExterQual_ORD",
# ]

def load_data() -> tuple:

    data_dir = Path("../data/")
    train = pd.read_csv(data_dir / "train.csv", index_col="Id")
    test = pd.read_csv(data_dir / "test.csv", index_col="Id")
    df = pd.concat([train, test])

    qualitative = [f for f in df.columns if df.dtypes[f] == 'object']
    df = clean(df)
    df = impute(df)
    df, _ = encode(df, qualitative)

    train = df.loc[train.index, :]
    test = df.loc[test.index, :]
    return train, test


def create_features(df, df_test=None):
    X = df.copy()
    y = X.pop("SalePrice")
    # mi_scores = make_mi_scores(X, y)

    encoder = CrossFoldEncoder(MEstimateEncoder, m=2)
    X = X.join(encoder.fit_transform(X, y, cols=["MSSubClass", "Neighborhood_NOM", 
                                                    "MSZoning_NOM"]))
    if df_test is not None:
        X_test = df_test.copy()        
        X_test.pop("SalePrice")
        X_test = X_test.join(encoder.transform(X_test))
        X = pd.concat([X, X_test])
    
    # X = drop_uninformative(X, mi_scores)
    X = X.join(mathematical_transforms(X))
    X = X.join(bool_feat(X))
    X = X.join(counts(X))
    X = X.join(group_transforms(X))
    X = X.join(cluster_labels(X, CLUSTER_FEATURES, n_clusters=13))
    # X = X.join(pca_components(X, PCA_FEATURES))


    if df_test is not None:
        X_test = X.loc[df_test.index, :]
        X.drop(df_test.index, inplace=True)

    if df_test is not None:
        return X, X_test
    else:
        return X

def clean(df):
    df["Exterior2nd"] = df["Exterior2nd"].replace({"Brk Cmn": "BrkComm"})
    return df

def impute(df):
    for name in df.select_dtypes("number"):
        df[name] = df[name].fillna(0)
    for name in df.select_dtypes("category"):
        df[name] = df[name].fillna("None")
    for name in df.select_dtypes("object"):
        df[name] = df[name].fillna("None")
    return df

def encode_nominative(df, feature, target=TARGET):
    ordering = pd.DataFrame()
    ordering['val'] = df[feature].unique()
    ordering.index = ordering.val
    ordering['spmean'] = df[[feature, target]].groupby(feature).median()[target]
    ordering = ordering.sort_values('spmean')
    ordering['ordering'] = range(1, ordering.shape[0]+1)
    ordering = ordering['ordering'].to_dict()
    
    for cat, ordinal in ordering.items():
        df.loc[df[feature] == cat, feature+'_NOM'] = ordinal

def encode_ordered(df, feature, levels):
    for ordinal, cat in enumerate(levels):
        df.loc[df[feature] == cat, feature+'_ORD'] = ordinal

def encode(df, qualitative):
    col_type = {}
    FINAL_TYPE = 'int32'
    for feature in qualitative:
        if feature in ORDERED_LEVELS.keys():
            encode_ordered(df, feature, ORDERED_LEVELS[feature])            
            col_type[feature+"_ORD"] = FINAL_TYPE
        else:
            encode_nominative(df, feature)
            col_type[feature+"_NOM"] = FINAL_TYPE
    df.drop(columns=qualitative, inplace=True)
    df = df.astype(col_type)
    return df, list(col_type.keys())


def score_dataset(model, X, y):
    score = cross_val_score(
        model, X, y, cv=5, scoring="neg_mean_squared_error",
    )
    score = -1 * score.mean()
    score = np.sqrt(score)
    return score

def error_RMSE(actual, predicted):
    return np.sqrt(np.sum(np.square(actual-predicted))/len(actual))

def make_mi_scores(X, y):
    X = X.copy()
    for colname in X.select_dtypes(["object", "category"]):
        X[colname], _ = X[colname].factorize()

    discrete_features = [pd.api.types.is_integer_dtype(t) for t in X.dtypes]
    mi_scores = mutual_info_regression(X, y, discrete_features=discrete_features, random_state=0)
    mi_scores = pd.Series(mi_scores, name="MI Scores", index=X.columns)
    mi_scores = mi_scores.sort_values(ascending=False)
    return mi_scores


def plot_mi_scores(scores):
    scores = scores.sort_values(ascending=True)
    width = np.arange(len(scores))
    ticks = list(scores.index)
    plot_mi = plt.figure(figsize=(8, 12), dpi=80)
    plt.barh(width, scores)
    plt.yticks(width, ticks)
    plt.title("Mutual Information Scores")
    plt.show(plot_mi)

def drop_uninformative(df, mi_scores, mi_treshold = 0.0):
    return df.loc[:, mi_scores > mi_treshold]


def label_encode(df):
    X = df.copy()
    for colname in X.select_dtypes(["category"]):
        X[colname] = X[colname].cat.codes
    return X


def mathematical_transforms(df):
    X = pd.DataFrame()
    X["TotalLivArea"] = (df.GrLivArea + df.TotalBsmtSF)
    X["Spaciousness"] = df['1stFlrSF'] + df['2ndFlrSF']
    X["MSZoningTargetLog"] = np.log(df.MSZoning_NOM_Target_E)
    # X["OverallQualExp"] = 2 ** df.OverallQual
    # X["OverallQualPow"] = df.OverallQual ** 2
    #Multiplicar features Qual*Area tende a causar overfit
    return X

def bool_feat(df):
    X = pd.DataFrame()
    X['HasGarage'] = df['GarageArea'].apply(lambda x: 1 if x > 0 else 0)
    X['Has2ndFloor'] = df['2ndFlrSF'].apply(lambda x: 1 if x > 0 else 0)
    return X


def counts(df):
    X = pd.DataFrame()
    X["PorchTypes"] = df[[
        "WoodDeckSF",
        "OpenPorchSF",
        "EnclosedPorch",
        "3SsnPorch",
        "ScreenPorch",
    ]].gt(0.0).sum(axis=1)
    return X


def group_transforms(df):
    X = pd.DataFrame()
    X["MedNhbdLivArea"] = df.groupby("Neighborhood_NOM")["GrLivArea"].transform("median")
    X["MedNhbdSpacArea"] = df.groupby("Neighborhood_NOM")["Spaciousness"].transform("median")
    return X

def cluster_labels(df, features, n_clusters=20):
    X = df.copy()
    X_scaled = X.loc[:, features]
    X_scaled = (X_scaled - X_scaled.mean(axis=0)) / X_scaled.std(axis=0)
    kmeans = KMeans(n_clusters=n_clusters, n_init=50, random_state=0)
    X_new = pd.DataFrame()
    X_new["Cluster"+str(n_clusters)] = kmeans.fit_predict(X_scaled)
    return X_new

def apply_pca(X, standardize=True):
    if standardize:
        X = (X - X.mean(axis=0)) / X.std(axis=0)

    pca = PCA()
    X_pca = pca.fit_transform(X)

    component_names = [f"PC{i+1}" for i in range(X_pca.shape[1])]
    X_pca = pd.DataFrame(X_pca, columns=component_names)

    loadings = pd.DataFrame(
        pca.components_.T,  # transpose the matrix of loadings
        columns=component_names,  # so the columns are the principal components
        index=X.columns,  # and the rows are the original features
    )
    return pca, X_pca, loadings


def plot_variance(pca, width=8, dpi=100):
    # Create figure
    fig, axs = plt.subplots(1, 2)
    n = pca.n_components_
    grid = np.arange(1, n + 1)
    # Explained variance
    evr = pca.explained_variance_ratio_
    axs[0].bar(grid, evr)
    axs[0].set(
        xlabel="Component", title="% Explained Variance", ylim=(0.0, 1.0)
    )
    # Cumulative Variance
    cv = np.cumsum(evr)
    axs[1].plot(np.r_[0, grid], np.r_[0, cv], "o-")
    axs[1].set(
        xlabel="Component", title="% Cumulative Variance", ylim=(0.0, 1.0)
    )
    # Set up figure
    fig.set(figwidth=8, dpi=100)
    return axs

def pca_components(df, features):
    X = df.loc[:, features]
    _, X_pca, _ = apply_pca(X)
    return X_pca

def corrplot(df, corr_method="pearson", annot=True, cluster_method="complete", **kwargs):
    corr = df.corr(corr_method)
    sns.clustermap(
        corr,
        vmin=-1.0,
        vmax=1.0,
        cmap="icefire",
        method=cluster_method,
        annot=annot,
        # mask=np.triu(np.ones_like(corr, dtype=bool)),
        **kwargs,
    )

def plot_defaults() -> None:
    plt.style.use("seaborn-whitegrid")
    plt.rc("figure", autolayout=True)
    plt.rc(
    "axes",
    labelweight="bold",
    labelsize="large",
    titleweight="bold",
    titlesize=14,
    titlepad=10,
    )

def plot_model_importances(feature_importances, index):
    importances = pd.Series(data=feature_importances, index=index)
    importances = importances.sort_values(ascending=True)
    width = np.arange(len(importances))
    ticks = list(importances.index)
    plot_mi = plt.figure(figsize=(8, 16), dpi=80)
    plt.barh(width, importances)
    plt.yticks(width, ticks)
    plt.title("Feature Importance")
    plt.show(plot_mi)

def plt_cross_val_predict(y, predicted):
    _, ax = plt.subplots()
    ax.scatter(y, predicted, edgecolors=(0, 0, 0))
    ax.plot([y.min(), y.max()], [y.min(), y.max()], "k--", lw=4)
    ax.set_xlabel("Measured SalePrice")
    ax.set_ylabel("Predicted SalePrice")
    plt.show()

""" Target encoding com CrossFold"""
class CrossFoldEncoder:
    def __init__(self, encoder, **kwargs):
        self.encoder_ = encoder
        self.kwargs_ = kwargs  # keyword arguments for the encoder
        self.cv_ = KFold(n_splits=5)

    # Fit an encoder on one split and transform the feature on the
    # other. Iterating over the splits in all folds gives a complete
    # transformation. We also now have one trained encoder on each
    # fold.
    def fit_transform(self, X, y, cols):
        self.fitted_encoders_ = []
        self.cols_ = cols
        X_encoded = []
        for idx_encode, idx_train in self.cv_.split(X):
            fitted_encoder = self.encoder_(cols=cols, **self.kwargs_)
            fitted_encoder.fit(
                X.iloc[idx_encode, :], y.iloc[idx_encode],
            )
            X_encoded.append(fitted_encoder.transform(X.iloc[idx_train, :])[cols])
            self.fitted_encoders_.append(fitted_encoder)
        X_encoded = pd.concat(X_encoded)
        X_encoded.columns = [name + "_Target_E" for name in X_encoded.columns]
        return X_encoded

    # To transform the test data, average the encodings learned from
    # each fold.
    def transform(self, X):
        from functools import reduce

        X_encoded_list = []
        for fitted_encoder in self.fitted_encoders_:
            X_encoded = fitted_encoder.transform(X)
            X_encoded_list.append(X_encoded[self.cols_])
        X_encoded = reduce(
            lambda x, y: x.add(y, fill_value=0), X_encoded_list
        ) / len(X_encoded_list)
        X_encoded.columns = [name + "_Target_E" for name in X_encoded.columns]
        return X_encoded