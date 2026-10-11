"""Write a small "new traffic" CSV for the console's *Score a traffic file* demo.

Flows are drawn only from the **test split** of the given config, so the deployed model has never trained on them.
Every attack family is included (up to ``--per-family`` flows each), the rest is benign traffic.

    python scripts/make_demo_traffic.py --config configs/cicids2017.yaml --out data/raw/demo_cicids2017.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nids import BENIGN  # noqa: E402
from nids.config import load_config  # noqa: E402
from nids.data import load_dataset, split_indices  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--out", required=True)
    parser.add_argument("--rows", type=int, default=5000)
    parser.add_argument("--per-family", type=int, default=150)
    args = parser.parse_args()

    cfg = load_config(args.config)
    data = load_dataset(cfg["data"], cfg["seed"])
    split = split_indices(data.y, cfg["data"]["test_size"], cfg["data"]["val_size"], cfg["seed"])
    rng = np.random.default_rng(cfg["seed"])
    y_test = data.y.iloc[split.test]
    picks = []
    for family, rows in pd.Series(split.test, index=y_test.values).groupby(level=0):
        if family != BENIGN:
            picks.append(rng.choice(rows.to_numpy(), min(len(rows), args.per_family), replace=False))
    attacks = np.concatenate(picks) if picks else np.array([], dtype=int)
    benign_rows = split.test[(y_test == BENIGN).to_numpy()]
    benign = rng.choice(benign_rows, max(args.rows - len(attacks), 0), replace=False)
    rows = np.sort(np.concatenate([attacks, benign]))

    ctx = data.context.iloc[rows].reset_index(drop=True)
    out = pd.DataFrame(index=range(len(rows)))
    if "timestamp" in ctx:
        out["timestamp"] = pd.to_datetime(ctx["timestamp"]).dt.strftime("%Y-%m-%d %H:%M:%S")
    for col in ("src_ip", "dst_ip"):
        if col in ctx:
            out[col] = ctx[col].values
    out = pd.concat([out, data.X.iloc[rows].reset_index(drop=True)], axis=1)
    out["label"] = data.y.iloc[rows].values
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"Wrote {len(out):,} test-split flows to {args.out}: {out['label'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()
