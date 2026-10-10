"""Compare benign-only novelty detectors at the same false-alarm budget.

Each detector is trained on benign training flows, its threshold is set to the 99th percentile of
benign validation scores (1% false alarms), and we measure the share of each attack family in the test
set it flags, i.e. how much of that family it would catch if the classifier had never seen it.

    python scripts/compare_novelty_detectors.py [--config configs/default.yaml] [--fpr 0.01]

Writes reports/results/novelty_detectors.json and prints the comparison table.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nids import BENIGN  # noqa: E402
from nids.config import load_config, output_dirs  # noqa: E402
from nids.data import load_dataset, split_indices  # noqa: E402
from nids.features import build_preprocessor  # noqa: E402


def autoencoder_scores(Z_train, Z_eval_list, seed=0, epochs=25):
    import torch
    from torch import nn

    torch.manual_seed(seed)
    n = Z_train.shape[1]
    model = nn.Sequential(nn.Linear(n, 64), nn.ReLU(), nn.Linear(64, 16), nn.ReLU(),
                          nn.Linear(16, 64), nn.ReLU(), nn.Linear(64, n))
    opt = torch.optim.Adam(model.parameters(), 1e-3)
    T = torch.tensor(Z_train, dtype=torch.float32)
    for _ in range(epochs):
        perm = torch.randperm(len(T))
        for i in range(0, len(T), 512):
            batch = T[perm[i:i + 512]]
            opt.zero_grad()
            loss = ((model(batch) - batch) ** 2).mean()
            loss.backward()
            opt.step()
    with torch.no_grad():
        return [((model(torch.tensor(Z, dtype=torch.float32)) - torch.tensor(Z, dtype=torch.float32)) ** 2)
                .mean(1).numpy() for Z in Z_eval_list]


def main() -> None:
    from sklearn.ensemble import IsolationForest
    from sklearn.neighbors import LocalOutlierFactor

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--fpr", type=float, default=0.01)
    args = parser.parse_args()
    cfg = load_config(args.config)
    data = load_dataset(cfg["data"], cfg["seed"])
    split = split_indices(data.y, cfg["data"]["test_size"], cfg["data"]["val_size"], cfg["seed"])
    X, y = data.X, data.y
    benign_tr = X.iloc[split.train][(y.iloc[split.train] == BENIGN).to_numpy()]
    benign_va = X.iloc[split.val][(y.iloc[split.val] == BENIGN).to_numpy()]
    X_te, y_te = X.iloc[split.test], y.iloc[split.test].to_numpy()

    pre = build_preprocessor(data.numeric, data.categorical).fit(benign_tr)
    Z_tr, Z_va, Z_te = pre.transform(benign_tr), pre.transform(benign_va), pre.transform(X_te)
    rng = np.random.default_rng(cfg["seed"])
    Z_ref = Z_tr[rng.choice(len(Z_tr), min(20_000, len(Z_tr)), replace=False)]

    def evaluate(name, val_scores, test_scores, seconds):
        threshold = np.quantile(val_scores, 1 - args.fpr)
        flagged = test_scores > threshold
        row = {"detector": name, "benign_false_alarms": float(flagged[y_te == BENIGN].mean()),
               "seconds": round(seconds, 1)}
        for family in sorted(set(y_te) - {BENIGN}):
            row[family] = float(flagged[y_te == family].mean())
        return row

    rows = []
    t = time.perf_counter()
    forest = IsolationForest(n_estimators=200, max_samples=2048, random_state=cfg["seed"], n_jobs=-1).fit(Z_tr)
    rows.append(evaluate("Isolation Forest", -forest.score_samples(Z_va), -forest.score_samples(Z_te),
                         time.perf_counter() - t))
    for k in (10, 30):
        t = time.perf_counter()
        lof = LocalOutlierFactor(n_neighbors=k, novelty=True, n_jobs=-1).fit(Z_ref)
        rows.append(evaluate(f"LOF (k={k})", -lof.score_samples(Z_va), -lof.score_samples(Z_te),
                             time.perf_counter() - t))
    try:
        t = time.perf_counter()
        va, te = autoencoder_scores(Z_tr, [Z_va, Z_te], seed=cfg["seed"])
        rows.append(evaluate("Autoencoder", va, te, time.perf_counter() - t))
    except ImportError:
        print("PyTorch not installed: skipping the autoencoder")

    table = pd.DataFrame(rows).set_index("detector")
    families = [c for c in table.columns if c not in ("benign_false_alarms", "seconds")]
    table["mean_attack_detection"] = table[families].mean(axis=1)
    out = output_dirs(cfg)["results"] / "novelty_detectors.json"
    out.write_text(json.dumps({"fpr_budget": args.fpr, "rows": rows}, indent=2), encoding="utf-8")
    pd.set_option("display.width", 200)
    print((table * 1).round(3).to_string())
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
