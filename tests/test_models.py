import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.datasets import make_classification
from sklearn.metrics import f1_score

from nids.deep import TORCH_AVAILABLE, TorchMLPClassifier
from nids.models import build_pipeline
from nids.synthetic import COLUMNS, generate_flows


def test_synthetic_generator_schema_and_drift():
    df = generate_flows(3000, seed=1)
    assert list(df.columns) == list(COLUMNS.values())
    assert np.isinf(df["Flow Bytes/s"]).any()  # CIC-IDS style quirk is reproduced
    assert df["Label"].value_counts().idxmax() == "BENIGN"
    assert "Exfiltration" not in set(generate_flows(3000, seed=1, drift=False)["Label"])


@pytest.mark.parametrize("model", ["logreg", "random_forest", "hist_gb"])
@pytest.mark.parametrize("strategy", ["none", "class_weight", "smote", "hybrid"])
def test_every_classic_model_strategy_pair_fits(model, strategy):
    X, y = make_classification(n_samples=600, n_features=6, n_informative=4, n_classes=3,
                               weights=[0.85, 0.12, 0.03], random_state=0)
    X = pd.DataFrame(X, columns=[f"f{i}" for i in range(6)])
    y = pd.Series(np.array(["Benign", "DoS", "Bot"])[y])
    cfg = {"seed": 0, "models": {"random_forest": {"n_estimators": 20}, "hist_gb": {"max_iter": 30}}}
    pipe = build_pipeline(model, strategy, y, list(X.columns), [], cfg).fit(X, y)
    proba = pipe.predict_proba(X)
    assert proba.shape == (600, 3) and np.allclose(proba.sum(axis=1), 1)


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")
def test_torch_mlp_learns_and_is_sklearn_compatible():
    X, y = make_classification(n_samples=1500, n_features=10, n_informative=6, n_classes=3,
                               weights=[0.8, 0.15, 0.05], random_state=0)
    clf = TorchMLPClassifier(hidden_dims=(32, 16), max_epochs=15, class_weight="balanced", random_state=0)
    clf.fit(X, y)
    assert f1_score(y, clf.predict(X), average="macro") > 0.6  # majority-only guessing scores ~0.3
    assert clf.best_val_macro_f1_ > 0.5 and clf.history_
    assert np.allclose(clf.predict_proba(X[:10]).sum(axis=1), 1, atol=1e-5)
    assert clone(clf).get_params()["gamma"] == 2.0
