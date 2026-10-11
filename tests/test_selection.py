from nids.experiments import comparison_strategy, select_best


def _bench():
    return {
        "selection_metric": "macro_f1",
        "val_support": {"Benign": 1000, "DoS": 200, "Tiny": 1},
        "rows": [
            # wins plain macro-F1 only because it got the single "Tiny" validation flow right
            {"model": "rf", "strategy": "none", "macro_f1": 0.93, "rare_macro_recall": 0.5,
             "per_class_f1": {"Benign": 0.99, "DoS": 0.80, "Tiny": 1.0}},
            {"model": "rf", "strategy": "smote", "macro_f1": 0.65, "rare_macro_recall": 0.5,
             "per_class_f1": {"Benign": 0.99, "DoS": 0.95, "Tiny": 0.0}},
            {"model": "rf", "strategy": "hybrid", "macro_f1": 0.60, "rare_macro_recall": 0.5,
             "per_class_f1": {"Benign": 0.98, "DoS": 0.90, "Tiny": 0.0}},
            {"model": "gb", "strategy": "none", "error": "boom"},
        ],
    }


def test_selection_ignores_classes_with_tiny_validation_support():
    bench = select_best(_bench(), min_val_support=10)
    assert bench["best"]["strategy"] == "smote"
    assert bench["excluded_from_selection"] == ["Tiny"] and bench["selection_classes"] == ["Benign", "DoS"]
    # without the guard the single-flow class decides the winner
    assert select_best(_bench(), min_val_support=1)["best"]["strategy"] == "none"


def test_comparison_strategy_never_compares_a_model_with_itself():
    bench = select_best(_bench(), min_val_support=10)
    assert comparison_strategy("rf", "smote", bench) == "none"
    assert comparison_strategy("rf", "none", bench) == "smote"  # best imbalance strategy for the same model
    assert comparison_strategy("lr", "none", None) == "class_weight"
