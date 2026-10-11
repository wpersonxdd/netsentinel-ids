"""Loading, cleaning, label normalisation and splitting of labelled network-flow datasets.

The loader is format-agnostic: it handles CIC-IDS2017/2018-style CSVs, UNSW-NB15, headerless
NSL-KDD files and any CSV/Parquet with one row per flow plus a label column. That way the same
pipeline runs on the synthetic demo data today and on the official hackathon dataset later.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from nids import BENIGN

log = logging.getLogger(__name__)

LABEL_CANDIDATES = ["attack_cat", "label", "class", "attack", "attack_type", "category", "target", "outcome"]
TIME_CANDIDATES = ["timestamp", "stime", "start_time", "time", "ts", "date_time", "datetime"]

# Addressing / bookkeeping columns. They identify *which* hosts took part in this particular capture
# rather than *how* the flow behaved, so a model trained on them memorises the lab setup and fails on
# a new network. They are kept as analyst context but never used as features.
ID_COLUMNS = {
    "flow_id", "id", "unnamed_0", "index", "no",
    "src_ip", "source_ip", "srcip", "src_addr", "saddr", "ip_src",
    "dst_ip", "destination_ip", "dstip", "dst_addr", "daddr", "ip_dst",
    "src_port", "source_port", "sport", "srcport",
    "ltime", "difficulty", "difficulty_level",
}
CONTEXT_ALIASES = {
    "src_ip": ["src_ip", "source_ip", "srcip", "src_addr", "saddr", "ip_src"],
    "dst_ip": ["dst_ip", "destination_ip", "dstip", "dst_addr", "daddr", "ip_dst"],
    "dst_port": ["destination_port", "dst_port", "dsport", "dport", "dstport"],
}
MAX_CATEGORIES = 100  # object columns with more distinct values than this are identifiers, not categories

NSLKDD_COLUMNS = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes", "land", "wrong_fragment",
    "urgent", "hot", "num_failed_logins", "logged_in", "num_compromised", "root_shell", "su_attempted",
    "num_root", "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds",
    "is_host_login", "is_guest_login", "count", "srv_count", "serror_rate", "srv_serror_rate",
    "rerror_rate", "srv_rerror_rate", "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate",
    "dst_host_count", "dst_host_srv_count", "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate", "dst_host_serror_rate",
    "dst_host_srv_serror_rate", "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "label", "difficulty",
]

# --------------------------------------------------------------------------------------------------
# Label normalisation: fine-grained attack names -> attack families
# --------------------------------------------------------------------------------------------------

_BENIGN_NAMES = {"benign", "normal", "0", "0 0", "false", "legitimate", "clean", "background"}

_NSLKDD_FAMILIES = {
    "DoS": ["back", "land", "neptune", "pod", "smurf", "teardrop", "apache2", "udpstorm",
            "processtable", "mailbomb"],
    "Probe": ["satan", "ipsweep", "nmap", "portsweep", "mscan", "saint"],
    "R2L": ["guess_passwd", "ftp_write", "imap", "phf", "multihop", "warezmaster", "warezclient", "spy",
            "xlock", "xsnoop", "snmpguess", "snmpgetattack", "httptunnel", "sendmail", "named", "worm"],
    "U2R": ["buffer_overflow", "loadmodule", "rootkit", "perl", "sqlattack", "xterm", "ps"],
}
_UNSW_FAMILIES = {
    "generic": "Generic", "exploits": "Exploits", "fuzzers": "Fuzzers", "reconnaissance": "Reconnaissance",
    "analysis": "Analysis", "backdoor": "Backdoor", "backdoors": "Backdoor", "shellcode": "Shellcode",
    "worms": "Worms",
}
# Applied in order; the first matching pattern wins (web attacks before brute force, DDoS before DoS).
_KEYWORD_FAMILIES = [
    (r"\bweb\b|webattack|\bxss\b|sql ?injection|\bsqli\b", "WebAttack"),
    (r"ddos|drdos", "DDoS"),
    (r"\bdos\b|hulk|goldeneye|slowloris|slowhttp", "DoS"),
    (r"scan", "PortScan"),
    (r"brute|patator|password", "BruteForce"),
    (r"\bbot", "Botnet"),
    (r"infilt", "Infiltration"),
    (r"exfil", "Exfiltration"),
    (r"heartbleed|exploit", "Exploit"),
]


def normalize_text(value: object) -> str:
    """Lower-case, replace punctuation (and mojibake such as 'Web Attack � XSS') with spaces."""
    text = re.sub(r"[^a-z0-9]+", " ", str(value).strip().lower())
    return re.sub(r"\s+", " ", text).strip()


_EXACT_FAMILIES = {normalize_text(name): fam for fam, names in _NSLKDD_FAMILIES.items() for name in names}
_EXACT_FAMILIES.update(_UNSW_FAMILIES)
_EXACT_FAMILIES.update({"bot": "Botnet", "infilteration": "Infiltration", "heartbleed": "Exploit"})


def to_family(label: object) -> str:
    """Map one raw label (any dataset vocabulary) to an attack family name."""
    norm = normalize_text(label)
    if norm in _BENIGN_NAMES:
        return BENIGN
    if norm in _EXACT_FAMILIES:
        return _EXACT_FAMILIES[norm]
    for pattern, family in _KEYWORD_FAMILIES:
        if re.search(pattern, norm):
            return family
    return str(label).strip()


def normalize_labels(labels: pd.Series, mode: str = "family") -> pd.Series:
    """Normalise a label column. ``mode='family'`` groups variants, ``'raw'`` only canonicalises benign."""
    values = labels.dropna().unique()
    as_numbers = pd.to_numeric(pd.Series(values), errors="coerce")
    if len(values) and as_numbers.notna().all() and set(as_numbers) <= {0, 1}:  # binary 0/1 labels
        mapping = {v: (BENIGN if float(v) == 0 else "Attack") for v in values}
    elif mode == "family":
        mapping = {v: to_family(v) for v in values}
    else:
        mapping = {v: (BENIGN if normalize_text(v) in _BENIGN_NAMES else str(v).strip()) for v in values}
    return labels.map(mapping)


# --------------------------------------------------------------------------------------------------
# Reading
# --------------------------------------------------------------------------------------------------


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """snake_case column names ('Flow Bytes/s' -> 'flow_bytes_s'), de-duplicated."""
    seen: dict[str, int] = {}
    names = []
    for col in df.columns:
        name = re.sub(r"[^0-9a-zA-Z]+", "_", str(col).strip()).strip("_").lower() or "col"
        if name in seen:
            seen[name] += 1
            name = f"{name}_{seen[name]}"
        else:
            seen[name] = 0
        names.append(name)
    out = df.copy()
    out.columns = names
    return out


def _looks_headerless(df: pd.DataFrame) -> bool:
    numeric_names = pd.to_numeric(pd.Series(df.columns.astype(str)), errors="coerce").notna().sum()
    return numeric_names > len(df.columns) / 2


def _read_one(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    df = pd.read_csv(path, low_memory=False, skipinitialspace=True, encoding_errors="replace")
    if _looks_headerless(df):
        df = pd.read_csv(path, header=None, low_memory=False, encoding_errors="replace")
        if df.shape[1] in (42, 43):  # NSL-KDD / KDD'99 layout
            df.columns = NSLKDD_COLUMNS[: df.shape[1]]
        else:
            df.columns = [f"col_{i}" for i in range(df.shape[1] - 1)] + ["label"]
        log.info("%s has no header row; assigned %d column names", path.name, df.shape[1])
    return df


def read_table(path: str | Path) -> pd.DataFrame:
    """Read one CSV/Parquet file, or concatenate every such file in a directory."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}. Generate the demo data with `python -m nids generate-data` "
            "or set data.path in the config."
        )
    suffixes = {".csv", ".parquet", ".txt"}
    files = sorted(p for p in path.rglob("*") if p.suffix.lower() in suffixes) if path.is_dir() else [path]
    if not files:
        raise FileNotFoundError(f"No CSV/Parquet files under {path}")
    frames = []
    for f in files:
        frame = normalize_columns(_read_one(f))
        log.info("Read %s: %d rows x %d columns", f.name, *frame.shape)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True, sort=False)


# --------------------------------------------------------------------------------------------------
# Preparation
# --------------------------------------------------------------------------------------------------


def detect_label_column(df: pd.DataFrame, preferred: str | None = None) -> str:
    if preferred:
        name = normalize_columns(pd.DataFrame(columns=[preferred])).columns[0]
        if name not in df.columns:
            raise ValueError(f"Label column '{preferred}' not found. Columns: {list(df.columns)[:50]}")
        return name
    present = [c for c in LABEL_CANDIDATES if c in df.columns]
    if not present:
        raise ValueError(f"No label column found; set data.label_column. Columns: {list(df.columns)[:50]}")
    multiclass = [c for c in present if df[c].nunique(dropna=True) > 2]
    return (multiclass or present)[0]


_DAYFIRST_FORMATS = ["%d/%m/%Y %H:%M:%S", "%d/%m/%Y %H:%M", "%d/%m/%Y %I:%M:%S %p", "%d/%m/%Y %I:%M %p"]


def parse_time(series: pd.Series, dayfirst: bool = False, add_12h_before_hour: int | None = None) -> pd.Series | None:
    """Parse epoch seconds/milliseconds or date strings; None when the column is not a usable time.

    ``dayfirst``: dates like 7/7/2017 are day/month/year (CIC-IDS2017). ``add_12h_before_hour``: repair a 12-hour
    clock written without AM/PM; hours below this value get +12 h. CIC-IDS2017 captured 08:00-18:00 and writes
    afternoon flows as 1:00-5:00, so ``8`` restores the true order.
    """
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().mean() > 0.9:
        unit = "ms" if numeric.median() > 1e12 else "s"
        parsed = pd.to_datetime(numeric, unit=unit, errors="coerce")
    elif dayfirst:
        text = series.astype(str).str.strip()
        parsed = pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns]")
        for fmt in _DAYFIRST_FORMATS:  # files of one dataset can mix formats
            missing = parsed.isna()
            if not missing.any():
                break
            parsed[missing] = pd.to_datetime(text[missing], format=fmt, errors="coerce")
        missing = parsed.isna()
        if missing.any():  # ISO timestamps (e.g. re-exported data) are unambiguous
            parsed[missing] = pd.to_datetime(text[missing], format="ISO8601", errors="coerce")
    else:
        parsed = pd.to_datetime(series, errors="coerce")
    if parsed.notna().mean() <= 0.5:
        return None
    if add_12h_before_hour is not None:
        parsed = parsed.where(parsed.dt.hour >= add_12h_before_hour, parsed + pd.Timedelta(hours=12))
    return parsed


def detect_time_column(df: pd.DataFrame, preferred: str | None = None, dayfirst: bool = False,
                       add_12h_before_hour: int | None = None) -> tuple[str | None, pd.Series | None]:
    candidates = [normalize_columns(pd.DataFrame(columns=[preferred])).columns[0]] if preferred else TIME_CANDIDATES
    for col in candidates:
        if col in df.columns:
            parsed = parse_time(df[col], dayfirst, add_12h_before_hour)
            if parsed is not None:
                return col, parsed
    return None, None


def extract_context(df: pd.DataFrame, time: pd.Series | None = None) -> pd.DataFrame:
    """Analyst-facing context (timestamp, IPs, port) aligned with the rows of ``df``."""
    ctx = pd.DataFrame(index=df.index)
    if time is not None:
        ctx["timestamp"] = time.values
    for name, aliases in CONTEXT_ALIASES.items():
        for alias in aliases:
            if alias in df.columns:
                ctx[name] = df[alias].values
                break
    return ctx


def as_category_text(col: pd.Series) -> pd.Series:
    """Categorical values as plain text with an explicit 'missing' level (same at train and inference)."""
    return col.astype(object).where(col.notna(), "missing").astype(str)


def infer_feature_types(X: pd.DataFrame) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Coerce mostly-numeric text columns to numbers; split into numeric and low-cardinality categorical."""
    X = X.copy()
    for col in X.columns:
        if pd.api.types.is_bool_dtype(X[col]):
            X[col] = X[col].astype(int)
        elif not pd.api.types.is_numeric_dtype(X[col]):  # object or pandas string dtype
            converted = pd.to_numeric(X[col], errors="coerce")
            if converted.notna().sum() >= 0.95 * X[col].notna().sum():
                X[col] = converted
    numeric = list(X.select_dtypes(include="number").columns)
    categorical, dropped = [], []
    for col in X.columns.difference(numeric, sort=False):
        if X[col].nunique(dropna=True) <= MAX_CATEGORIES:
            categorical.append(col)
        else:
            dropped.append(col)
    if dropped:
        log.info("Dropping high-cardinality text columns: %s", dropped)
    X = X[numeric + categorical]
    X[numeric] = X[numeric].astype(float).replace([np.inf, -np.inf], np.nan)
    for col in categorical:
        X[col] = as_category_text(X[col])
    useless = [c for c in X.columns if X[c].nunique(dropna=True) <= 1]
    if useless:
        log.info("Dropping %d constant/empty columns: %s", len(useless), useless)
        X = X.drop(columns=useless)
    numeric = [c for c in numeric if c in X.columns]
    categorical = [c for c in categorical if c in X.columns]
    return X, numeric, categorical


def stratified_subsample(y: pd.Series, max_rows: int, seed: int, min_per_class: int = 1000) -> np.ndarray:
    """Positions of a class-stratified subsample that never thins a class below ``min_per_class``."""
    if max_rows is None or len(y) <= max_rows:
        return np.arange(len(y))
    rng = np.random.default_rng(seed)
    frac = max_rows / len(y)
    picks = []
    for _, positions in pd.Series(np.arange(len(y)), index=y.values).groupby(level=0):
        pos = positions.to_numpy()
        k = max(int(round(len(pos) * frac)), min(len(pos), min_per_class))
        picks.append(rng.choice(pos, size=k, replace=False))
    return np.sort(np.concatenate(picks))


def class_order(labels) -> list[str]:
    """Benign first, then attack families alphabetically: a stable order for tables and plots."""
    uniq = sorted({str(v) for v in labels})
    return ([BENIGN] if BENIGN in uniq else []) + [c for c in uniq if c != BENIGN]


@dataclass
class PreparedData:
    X: pd.DataFrame
    y: pd.Series
    context: pd.DataFrame
    time: pd.Series | None
    numeric: list[str]
    categorical: list[str]
    label_column: str
    source: str
    dropped_duplicates: int = 0
    dropped_rare: dict[str, int] = field(default_factory=dict)
    sampling: dict | None = None  # set when a stratified subsample of the cleaned data is used

    @property
    def classes(self) -> list[str]:
        return class_order(self.y.unique())

    @property
    def feature_columns(self) -> list[str]:
        return self.numeric + self.categorical


LOADER_VERSION = 3  # bump when preparation logic changes, to invalidate cached datasets


def _cache_path(data_cfg: dict, seed: int) -> Path:
    """Cache file keyed by the data config, the seed and every source file's size and modification time."""
    path = Path(data_cfg["path"]).resolve()
    files = sorted(p for p in path.rglob("*") if p.is_file()) if path.is_dir() else [path]
    fingerprint = {  # independent of how the path is written (relative from the CLI, absolute from the console)
        "cfg": {k: v for k, v in data_cfg.items() if k not in ("cache", "cache_dir", "path")},
        "seed": seed,
        "loader": LOADER_VERSION,
        "files": [(f.relative_to(path).as_posix() if path.is_dir() else f.name, f.stat().st_size,
                   int(f.stat().st_mtime)) for f in files],
    }
    digest = hashlib.sha1(json.dumps(fingerprint, sort_keys=True, default=str).encode()).hexdigest()[:16]
    return Path(data_cfg.get("cache_dir", "data/processed")) / f"prepared_{digest}.joblib"


def load_dataset(data_cfg: dict, seed: int = 42) -> PreparedData:
    """Read, clean and label a dataset according to the ``data`` section of the config.

    With ``data.cache: true`` the prepared dataset is stored under ``data/processed`` and reused while the source
    files and the data config are unchanged (large datasets take minutes to parse).
    """
    if data_cfg.get("cache"):
        cache = _cache_path(data_cfg, seed)
        if cache.exists():
            log.info("Using cached prepared dataset %s", cache)
            return joblib.load(cache)
        prepared = _load_dataset(data_cfg, seed)
        cache.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(prepared, cache)
        log.info("Cached prepared dataset to %s", cache)
        return prepared
    return _load_dataset(data_cfg, seed)


def _load_dataset(data_cfg: dict, seed: int) -> PreparedData:
    raw = read_table(data_cfg["path"])
    label_col = detect_label_column(raw, data_cfg.get("label_column"))
    log.info("Label column: %s", label_col)

    labels = raw[label_col]
    # UNSW-NB15 leaves attack_cat empty for normal traffic; recover it from the binary label.
    if label_col == "attack_cat" and "label" in raw.columns:
        normal = pd.to_numeric(raw["label"], errors="coerce") == 0
        blank = labels.isna() | labels.astype(str).str.strip().isin(["", "-"])
        labels = labels.where(~(blank & normal), BENIGN)
    y = normalize_labels(labels, data_cfg.get("label_mode", "family"))

    time_col, time = detect_time_column(raw, data_cfg.get("time_column"), data_cfg.get("time_dayfirst", False),
                                        data_cfg.get("time_add_12h_before_hour"))
    context = extract_context(raw, time)

    label_like = {c for c in LABEL_CANDIDATES if c in raw.columns}
    extra = {normalize_columns(pd.DataFrame(columns=[c])).columns[0] for c in data_cfg.get("extra_drop_columns", [])}
    drop = (ID_COLUMNS | label_like | extra | ({time_col} if time_col else set())) & set(raw.columns)
    X, numeric, categorical = infer_feature_types(raw.drop(columns=sorted(drop)))
    log.info("Features: %d numeric, %d categorical (dropped as identifiers/labels: %s)",
             len(numeric), len(categorical), sorted(drop))

    keep = y.notna().to_numpy(copy=True)
    # Exact duplicate flows inflate scores (the same row lands in train and test); drop them.
    dupes = pd.concat([X, y.rename("__label__")], axis=1).duplicated().to_numpy()
    n_dupes = int((dupes & keep).sum())
    keep &= ~dupes
    if n_dupes:
        log.info("Dropped %d exact duplicate rows", n_dupes)

    counts = y[keep].value_counts()
    min_count = int(data_cfg.get("min_class_count", 10))
    rare = counts[counts < min_count]
    if len(rare):
        log.warning("Dropping classes with < %d samples (cannot be learned or evaluated): %s",
                    min_count, rare.to_dict())
        keep &= ~y.isin(rare.index).to_numpy()

    X, y, context = X[keep], y[keep], context[keep]
    time = time[keep] if time is not None else None

    sampling = None
    if data_cfg.get("max_rows") and len(y) > int(data_cfg["max_rows"]):
        min_per_class = int(data_cfg.get("min_per_class", 1000))
        sampling = {"rows_before": int(len(y)), "max_rows": int(data_cfg["max_rows"]), "min_per_class": min_per_class,
                    "class_counts_before": {str(k): int(v) for k, v in y.value_counts().items()}}
        pos = stratified_subsample(y, int(data_cfg["max_rows"]), seed, min_per_class)
        X, y, context = X.iloc[pos], y.iloc[pos], context.iloc[pos]
        time = time.iloc[pos] if time is not None else None

    X, y, context = X.reset_index(drop=True), y.reset_index(drop=True), context.reset_index(drop=True)
    time = time.reset_index(drop=True) if time is not None else None
    log.info("Prepared %d rows, %d classes: %s", len(y), y.nunique(), y.value_counts().to_dict())
    return PreparedData(X, y, context, time, numeric, categorical, label_col, str(data_cfg["path"]),
                        n_dupes, rare.to_dict(), sampling)


def prepare_for_inference(df: pd.DataFrame, numeric: list[str], categorical: list[str], label_mode: str = "family",
                          time_dayfirst: bool = False, time_add_12h_before_hour: int | None = None
                          ) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series | None]:
    """Turn new raw traffic into the trained feature layout. Returns (X, context, true labels or None)."""
    df = normalize_columns(df)
    _, time = detect_time_column(df, None, time_dayfirst, time_add_12h_before_hour)
    context = extract_context(df, time)
    y = None
    present = [c for c in LABEL_CANDIDATES if c in df.columns]
    if present:
        y = normalize_labels(df[detect_label_column(df)], label_mode)
    X = df.reindex(columns=numeric + categorical)
    X[numeric] = X[numeric].apply(pd.to_numeric, errors="coerce").astype(float).replace([np.inf, -np.inf], np.nan)
    for col in categorical:
        X[col] = as_category_text(X[col])
    missing = [c for c in numeric + categorical if c not in df.columns]
    if missing:
        log.warning("%d trained feature columns missing from input (imputed): %s", len(missing), missing[:10])
    return X, context, y


@dataclass
class Split:
    train: np.ndarray
    val: np.ndarray
    test: np.ndarray


def split_indices(y: pd.Series, test_size: float, val_size: float, seed: int) -> Split:
    """Stratified train/validation/test positions; every class appears in every part."""
    idx = np.arange(len(y))
    trval, test = train_test_split(idx, test_size=test_size, stratify=y, random_state=seed)
    train, val = train_test_split(trval, test_size=val_size / (1 - test_size), stratify=y.iloc[trval],
                                  random_state=seed)
    return Split(np.sort(train), np.sort(val), np.sort(test))
