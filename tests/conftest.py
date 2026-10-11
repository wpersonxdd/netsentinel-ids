from pathlib import Path

import pytest

from nids.config import load_config
from nids.synthetic import generate_flows


@pytest.fixture(scope="session")
def synthetic_csv(tmp_path_factory) -> Path:
    path = tmp_path_factory.mktemp("data") / "synthetic_flows.csv"
    generate_flows(8_000, seed=7).to_csv(path, index=False)
    return path


@pytest.fixture()
def small_cfg(tmp_path, synthetic_csv) -> dict:
    """A fast configuration: one quick model, two strategies, tiny explanation budget."""
    return load_config(None, {
        "paths": {"artifacts": str(tmp_path / "artifacts"), "reports": str(tmp_path / "reports")},
        "data": {"path": str(synthetic_csv)},
        "models": {"hist_gb": {"max_iter": 60}},
        "benchmark": {"models": ["hist_gb"], "strategies": ["none", "smote"]},
        "explain": {"permutation_rows": 400, "permutation_repeats": 1},
        "novelty": {"min_family_test_count": 5},
        "evaluation": {"bootstrap": 100},
    })
