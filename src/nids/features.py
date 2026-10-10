"""Feature preprocessing shared by every model."""

from __future__ import annotations

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler


def signed_log1p(X):
    """sign(x) * log(1 + |x|): tames heavy-tailed byte/packet/rate counters spanning many orders of magnitude."""
    X = np.asarray(X, dtype=float)
    return np.sign(X) * np.log1p(np.abs(X))


def build_preprocessor(numeric: list[str], categorical: list[str]) -> ColumnTransformer:
    """Median imputation + signed log + standardisation for numbers; one-hot for small categoricals."""
    numeric_steps = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("log", FunctionTransformer(signed_log1p, feature_names_out="one-to-one")),
        ("scale", StandardScaler()),
    ])
    transformers = [("num", numeric_steps, numeric)]
    if categorical:
        categorical_steps = Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=20, max_categories=30,
                                     sparse_output=False)),
        ])
        transformers.append(("cat", categorical_steps, categorical))
    return ColumnTransformer(transformers, remainder="drop", verbose_feature_names_out=False)
