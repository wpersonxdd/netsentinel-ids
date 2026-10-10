import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

from nids.resampling import make_samplers, resampled_size, smote_targets, undersample_targets

COUNTS = pd.Series({"Benign": 10_000, "DoS": 1_000, "Bot": 50, "Tiny": 1})


def test_undersample_caps_only_the_majority():
    targets = undersample_targets(COUNTS, cap_ratio=4.0)
    assert targets["Benign"] == 4_000
    assert targets["DoS"] == 1_000 and targets["Bot"] == 50


def test_smote_targets_raise_minorities_and_skip_singletons():
    targets = smote_targets(COUNTS, ratio=0.15)
    assert targets == {"DoS": 1_500, "Bot": 1_500}


def test_strategies_build_expected_steps():
    y = pd.Series(np.repeat(["Benign", "DoS", "Bot"], [1000, 100, 6]))
    assert make_samplers("none", y, {}, 0) == [] and make_samplers("class_weight", y, {}, 0) == []
    smote = dict(make_samplers("smote", y, {"smote_target_ratio": 0.2}, 0))
    assert isinstance(smote["smote"], SMOTE) and smote["smote"].k_neighbors == 5
    hybrid = dict(make_samplers("hybrid", y, {"majority_cap_ratio": 2.0, "smote_target_ratio": 0.5}, 0))
    assert isinstance(hybrid["undersample"], RandomUnderSampler) and "smote" in hybrid


def test_smote_works_with_a_very_rare_class():
    rng = np.random.default_rng(0)
    y = pd.Series(np.repeat(["Benign", "Rare"], [500, 3]))
    X = rng.normal(size=(len(y), 4))
    (_, sampler), = make_samplers("smote", y, {"smote_target_ratio": 0.2}, 0)
    assert sampler.k_neighbors == 2  # must be below the class size
    _, y_res = sampler.fit_resample(X, y)
    assert (y_res == "Rare").sum() == 100
    assert resampled_size("smote", y, {"smote_target_ratio": 0.2}) == 600
