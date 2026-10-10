import numpy as np
import pandas as pd
import pytest

from nids import BENIGN
from nids.data import (
    load_dataset,
    normalize_columns,
    normalize_labels,
    split_indices,
    stratified_subsample,
    to_family,
)


def test_normalize_columns_snake_case_and_dedup():
    df = pd.DataFrame(columns=[" Flow Bytes/s", "Fwd Header Length", "Fwd Header Length", "Label"])
    assert list(normalize_columns(df).columns) == ["flow_bytes_s", "fwd_header_length", "fwd_header_length_1", "label"]


@pytest.mark.parametrize("raw, family", [
    ("BENIGN", BENIGN),
    ("normal.", BENIGN),
    ("Web Attack � Brute Force", "WebAttack"),
    ("Web Attack – XSS", "WebAttack"),
    ("Brute Force -Web", "WebAttack"),
    ("DoS Hulk", "DoS"),
    ("DoS attacks-Slowloris", "DoS"),
    ("DDoS", "DDoS"),
    ("DDOS attack-HOIC", "DDoS"),
    ("FTP-Patator", "BruteForce"),
    ("SSH-Bruteforce", "BruteForce"),
    ("PortScan", "PortScan"),
    ("Bot", "Botnet"),
    ("Infilteration", "Infiltration"),
    ("Heartbleed", "Exploit"),
    ("neptune", "DoS"),           # NSL-KDD
    ("guess_passwd", "R2L"),      # NSL-KDD
    ("buffer_overflow", "U2R"),   # NSL-KDD
    ("Exploits", "Exploits"),     # UNSW-NB15
    ("Backdoors", "Backdoor"),    # UNSW-NB15
    ("Totally New Thing", "Totally New Thing"),
])
def test_to_family(raw, family):
    assert to_family(raw) == family


def test_binary_numeric_labels():
    out = normalize_labels(pd.Series([0, 1, 1, 0]))
    assert list(out) == [BENIGN, "Attack", "Attack", BENIGN]


def test_two_string_labels_are_not_treated_as_binary_numbers():
    out = normalize_labels(pd.Series(["BENIGN", "DDoS", "DDoS"]))
    assert list(out) == [BENIGN, "DDoS", "DDoS"]


def _flows(n_benign=60, n_attack=40, n_rare=3, seed=0):
    rng = np.random.default_rng(seed)
    n = n_benign + n_attack + n_rare
    return pd.DataFrame({
        "Src IP": [f"10.0.0.{i % 250}" for i in range(n)],
        "Dst IP": "10.0.1.1",
        "Timestamp": pd.date_range("2026-01-01", periods=n, freq="min").astype(str),
        "Destination Port": rng.integers(1, 1024, n),
        "Protocol": rng.choice(["TCP", "UDP"], n),
        "Flow Duration": rng.lognormal(10, 1, n),
        "Flow Bytes/s": rng.lognormal(5, 1, n),
        "Constant": 1.0,
        "Label": ["BENIGN"] * n_benign + ["DoS Hulk"] * n_attack + ["Heartbleed"] * n_rare,
    })


def test_load_dataset_cleans_and_drops(tmp_path):
    df = _flows()
    df.loc[0, "Flow Bytes/s"] = np.inf
    df = pd.concat([df, df.iloc[[5, 6]]])  # exact duplicates
    path = tmp_path / "flows.csv"
    df.to_csv(path, index=False)

    data = load_dataset({"path": str(path), "min_class_count": 10})
    assert {"src_ip", "dst_ip", "timestamp", "label", "constant"}.isdisjoint(data.X.columns)
    assert "protocol" in data.categorical and "flow_bytes_s" in data.numeric
    assert np.isfinite(data.X["flow_bytes_s"].dropna()).all() and data.X["flow_bytes_s"].isna().sum() == 1
    assert data.dropped_duplicates == 2
    assert data.dropped_rare == {"Exploit": 3}
    assert set(data.y) == {BENIGN, "DoS"}
    assert {"src_ip", "dst_ip", "timestamp", "dst_port"} <= set(data.context.columns)
    assert data.time is not None and len(data.X) == len(data.y) == len(data.context)


def test_unsw_style_blank_attack_cat_becomes_benign(tmp_path):
    df = pd.DataFrame({
        "dur": np.arange(40, dtype=float),
        "proto": ["tcp"] * 40,
        "attack_cat": [None] * 20 + ["Exploits"] * 20,
        "label": [0] * 20 + [1] * 20,
    })
    path = tmp_path / "unsw.csv"
    df.to_csv(path, index=False)
    data = load_dataset({"path": str(path), "min_class_count": 5})
    assert data.label_column == "attack_cat"
    assert data.y.value_counts().to_dict() == {BENIGN: 20, "Exploits": 20}
    assert "label" not in data.X.columns


def test_nslkdd_headerless(tmp_path):
    rng = np.random.default_rng(1)
    rows = []
    for i in range(40):
        feats = [int(rng.integers(0, 100)), ["tcp", "udp"][i % 2], "http", "SF"] + [float(rng.random()) for _ in range(37)]
        rows.append(feats + ["normal" if i < 20 else "neptune", 21])
    path = tmp_path / "KDDTrain+.txt"
    pd.DataFrame(rows).to_csv(path, header=False, index=False)
    data = load_dataset({"path": str(path), "min_class_count": 5})
    assert data.y.value_counts().to_dict() == {BENIGN: 20, "DoS": 20}
    assert "difficulty" not in data.X.columns and "protocol_type" in data.categorical


def test_split_is_stratified_and_disjoint():
    y = pd.Series(["Benign"] * 800 + ["A"] * 150 + ["B"] * 50)
    s = split_indices(y, 0.2, 0.1, seed=0)
    assert len(set(s.train) & set(s.test)) == 0 and len(set(s.val) & set(s.test)) == 0
    assert len(s.train) + len(s.val) + len(s.test) == len(y)
    for part in (s.train, s.val, s.test):
        assert set(y.iloc[part]) == {"Benign", "A", "B"}


def test_stratified_subsample_keeps_rare_classes():
    y = pd.Series(["Benign"] * 10_000 + ["Rare"] * 30)
    pos = stratified_subsample(y, 1_000, seed=0, min_per_class=100)
    sub = y.iloc[pos].value_counts()
    assert sub["Rare"] == 30 and sub["Benign"] < 10_000
