import warnings

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# random state
RANDOM_STATE = 123


def select_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.query("sqft_living <= sqft_lot")
    df = df.query("price != 7062500 and price != 0")

    return df.reset_index(drop=True)


def outliers_remove(df: pd.DataFrame) -> pd.DataFrame:
    X = df[["sqft_living", "price"]]
    X_scaled = StandardScaler().fit_transform(X)

    cluster = DBSCAN(eps=0.8, min_samples=10)
    cluster.fit(X_scaled)

    df["cluster"] = cluster.labels_

    df = df.query("cluster == 0").drop("cluster", axis=1).reset_index(drop=True)

    return df


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    df["basement"] = np.where(df["sqft_basement"] > 0, 1, 0)  # have a basement?
    df["renovated"] = np.where(df["yr_renovated"] > 0, 1, 0)  # has been renovated?

    bins = [1900, 1940, 1970, 2000, 2014 + 1]
    labels = [
        "Pre-World War II",
        "Post-World War II",
        "Early Modern",
        "Modern",
    ]  # categorization based on architecture style

    df["era_category"] = pd.cut(df["yr_built"], bins=bins, labels=labels, right=False)

    return df


def feature_extrction(train_street: pd.Series, test_street: pd.Series):
    vectorizer = TfidfVectorizer(min_df=0.01, max_df=0.90)

    vectorizer.fit(train_street)

    train_tfidf_matrix = vectorizer.transform(train_street)
    test_tfidf_matrix = vectorizer.transform(test_street)

    train_tfidf_df = pd.DataFrame(
        train_tfidf_matrix.toarray(), columns=vectorizer.get_feature_names_out()
    )
    test_tfidf_df = pd.DataFrame(
        test_tfidf_matrix.toarray(), columns=vectorizer.get_feature_names_out()
    )

    return train_tfidf_df, test_tfidf_df


def data_integration(df: pd.DataFrame, location: pd.DataFrame) -> pd.DataFrame:
    df = df.merge(location, on="city", how="left")
    df.drop("city", axis=1, inplace=True)

    return df


def reformatted(df: pd.DataFrame) -> pd.DataFrame:
    df["statezip"] = df["statezip"].str.split().str[1].astype(int)

    return df


def mean_encoded(df: pd.DataFrame, era_map) -> pd.DataFrame:
    df["era_category"] = df["era_category"].map(era_map)
    df["era_category"] = df["era_category"].astype(float)

    return df


def log_transform(df: pd.DataFrame) -> pd.DataFrame:
    df[["sqft_living", "sqft_lot", "sqft_above"]] = np.log(
        df[["sqft_living", "sqft_lot", "sqft_above"]]
    )

    return df


def save_data(df: pd.DataFrame, filepath: str):
    df.to_csv(filepath, index=False)
