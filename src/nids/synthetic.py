"""Synthetic labelled network-flow generator with a CIC-IDS2017-like schema.

Why it exists: the official hackathon dataset is released later, and the whole pipeline (benchmark,
training, novelty detection, drift, triage UI) should run end-to-end today. Numbers measured on this
data demonstrate the *method*; they are not evidence of real-world detection performance.

What it models on purpose:
* severe class imbalance (benign ~80%, the rarest attack ~0.15%),
* attacks arriving in time-bounded campaigns (a 2-hour DoS, a 30-minute scan) rather than uniformly,
* attack families that overlap with benign traffic (web attacks look like browsing, infiltration
  looks like SMB file sharing, encrypted C2 looks like HTTPS), so rare classes are genuinely hard,
* concept drift: in the last 35% of the timeline DoS shifts to slow variants, port scans become
  stealthy, botnet C2 moves to TLS on 443, and a brand-new family (Exfiltration) appears,
* CIC-IDS data-quality quirks: Infinity rates for zero-duration flows, missing values, duplicates.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

START = pd.Timestamp("2026-09-01")
DAYS = 10


@dataclass(frozen=True)
class Profile:
    """Traffic profile: lognormal(median, sigma) shapes for volumes, Poisson means for host counters."""

    weight: float
    ports: tuple
    proto: str
    duration: tuple            # microseconds
    fwd_pkts: tuple
    bwd_pkts: tuple
    fwd_size: tuple            # mean payload bytes per forward packet
    bwd_size: tuple
    entropy: tuple             # normal(mu, sd), clipped to [0, 8] bits/byte
    iat_cv: tuple              # coefficient of variation of inter-arrival times (low = periodic)
    uniq_ports: float          # distinct destination ports contacted by the source in 10 s
    conn_same_dst: float       # connections from the source to the same destination in 10 s
    syn: float = 1.0
    fin: float = 0.8
    rst: float = 0.1
    psh: float = 0.3
    windows: tuple = (65535, 64240, 29200, 8192)
    port_range: bool = False   # ports=(lo, hi) sampled uniformly


P = Profile
BENIGN_PROFILES = [
    P(0.50, (443,), "TCP", (4e5, 1.8), (9, 0.9), (11, 1.0), (220, 0.6), (950, 0.6), (7.3, 0.4), (1.1, 0.5), 1.3, 3),
    P(0.14, (80, 8080), "TCP", (3e5, 1.8), (8, 0.9), (10, 1.0), (200, 0.6), (900, 0.6), (5.2, 0.9), (1.1, 0.5), 1.3, 3),
    P(0.16, (53,), "UDP", (2e4, 1.0), (1, 0.1), (1, 0.1), (70, 0.3), (150, 0.4), (4.0, 0.6), (0.5, 0.5), 1.1, 4),
    P(0.03, (22,), "TCP", (3e7, 1.5), (60, 1.0), (60, 1.0), (90, 0.5), (130, 0.6), (7.5, 0.2), (1.5, 0.5), 1.0, 1.5),
    P(0.02, (25, 587), "TCP", (2e6, 1.0), (12, 0.6), (10, 0.6), (400, 0.8), (80, 0.4), (5.5, 0.7), (1.0, 0.5), 1.0, 1.5),
    P(0.04, (445, 139), "TCP", (1e6, 1.5), (20, 1.0), (22, 1.0), (300, 0.8), (600, 0.8), (6.0, 0.9), (1.2, 0.5), 1.5, 2),
    P(0.01, (3389,), "TCP", (6e7, 1.2), (300, 1.0), (500, 1.0), (120, 0.5), (400, 0.7), (7.6, 0.2), (1.4, 0.4), 1.0, 1.2),
    P(0.09, (1024, 65535), "TCP", (1e6, 2.0), (10, 1.2), (10, 1.2), (300, 1.0), (500, 1.0), (6.0, 1.2), (1.2, 0.6), 1.5, 2,
      port_range=True),
    # Legitimate bulk uploads (backups, cloud sync): encrypted, upload-heavy, long-lived.
    P(0.01, (443, 22), "TCP", (3e7, 1.0), (600, 0.8), (150, 0.8), (1200, 0.15), (70, 0.3), (7.8, 0.1), (0.5, 0.4), 1, 1),
]

ATTACK_PROFILES = {
    "DoS": [  # HTTP flood (Hulk) and slow-header attacks (Slowloris)
        P(0.7, (80,), "TCP", (1e5, 1.0), (5, 0.4), (4, 0.5), (120, 0.3), (350, 0.6), (4.8, 0.5), (0.4, 0.4), 1, 45,
          fin=0.3, rst=0.4, windows=(29200, 65535)),
        P(0.3, (80,), "TCP", (8e7, 0.5), (12, 0.4), (1, 0.5), (40, 0.3), (0, 0), (3.5, 0.5), (0.15, 0.3), 1, 25,
          fin=0.05, rst=0.05, psh=0.9, windows=(29200,)),
    ],
    "DDoS": [  # LOIC-style flood from many sources
        P(1.0, (80,), "TCP", (3e3, 1.5), (3, 0.4), (1, 0.8), (60, 0.5), (60, 0.5), (2.5, 0.8), (0.3, 0.4), 1, 140,
          fin=0.05, rst=0.6, psh=0.1, windows=(8192, 65535)),
    ],
    "PortScan": [  # nmap SYN scan: one probe per port, RST back, tiny window
        P(1.0, (1, 65535), "TCP", (60, 1.0), (1, 0.2), (1, 0.3), (0, 0), (0, 0), (0.2, 0.2), (0.1, 0.3), 85, 85,
          fin=0.0, rst=0.95, psh=0.0, windows=(1024,), port_range=True),
    ],
    "BruteForce": [  # SSH-Patator and FTP-Patator
        P(0.6, (22,), "TCP", (2.5e6, 0.8), (14, 0.5), (16, 0.5), (85, 0.4), (115, 0.4), (7.5, 0.2), (0.6, 0.5), 1, 9,
          fin=0.9, psh=0.4, windows=(29200, 64240)),
        P(0.4, (21,), "TCP", (1.2e6, 0.8), (8, 0.5), (10, 0.5), (15, 0.4), (30, 0.4), (4.2, 0.5), (0.6, 0.5), 1, 8,
          fin=0.9, psh=0.6, windows=(29200, 64240)),
    ],
    "WebAttack": [  # login brute force, XSS and SQL injection hidden in ordinary-looking HTTP sessions
        P(1.0, (80, 8080), "TCP", (1.5e6, 1.5), (7, 0.7), (8, 0.8), (330, 0.6), (700, 0.7), (4.8, 0.8), (0.9, 0.5), 1.2, 4.5,
          fin=0.85, psh=0.45, windows=(29200, 65535, 64240)),
    ],
    "Botnet": [  # C2 beacons (Ares-like): small, fairly regular exchanges
        P(1.0, (8080, 443), "TCP", (1.2e5, 0.8), (4, 0.4), (3, 0.4), (210, 0.4), (160, 0.4), (6.6, 0.6), (0.3, 0.6), 1, 5,
          fin=1.0, psh=0.5, windows=(8192, 65535)),
    ],
    "Infiltration": [  # compromised internal host quietly touching SMB/RPC/RDP of its neighbours
        P(1.0, (445, 139, 443, 3389), "TCP", (1.5e6, 1.6), (22, 1.0), (24, 1.0), (320, 0.8), (620, 0.8), (6.3, 0.9),
          (1.1, 0.5), 3.2, 2.5, fin=0.8, rst=0.15),
    ],
}

# Share of each attack class's rows that are really benign-looking flows inside the attack window
# (handshakes, keep-alives, the victim's normal traffic). Public IDS datasets carry the same noise,
# and it caps the recall any model can reach.
LABEL_NOISE = 0.02

# Concept drift in the "future" part of the timeline.
DRIFTED_PROFILES = {
    **ATTACK_PROFILES,
    "DoS": [  # attackers shift from floods to low-and-slow variants
        P(0.3, (80,), "TCP", (1e5, 1.0), (5, 0.4), (4, 0.5), (120, 0.3), (350, 0.6), (4.8, 0.5), (0.4, 0.4), 1, 45,
          fin=0.3, rst=0.4, windows=(29200, 65535)),
        P(0.7, (80, 443), "TCP", (9e7, 0.6), (10, 0.5), (2, 0.5), (60, 0.4), (40, 0.5), (4.0, 0.6), (0.2, 0.3), 1, 12,
          fin=0.05, rst=0.05, psh=0.8, windows=(29200, 65535)),
    ],
    "PortScan": [  # slow, stealthy scan of well-known ports with randomised windows
        P(1.0, (1, 1024), "TCP", (500, 1.0), (1, 0.3), (1, 0.4), (0, 0), (0, 0), (0.3, 0.3), (0.3, 0.4), 12, 12,
          fin=0.0, rst=0.9, psh=0.0, windows=(1024, 65535, 29200), port_range=True),
    ],
    "Botnet": [  # C2 moved to TLS on 443: looks like normal HTTPS except for its regular timing
        P(1.0, (443,), "TCP", (1.5e5, 0.8), (5, 0.4), (4, 0.4), (240, 0.4), (190, 0.4), (7.4, 0.3), (0.3, 0.6), 1, 5,
          fin=1.0, psh=0.5, windows=(8192, 65535)),
    ],
    "Exfiltration": [  # NEW family: bulk encrypted uploads and DNS tunnelling
        P(0.6, (443,), "TCP", (6e7, 0.8), (900, 0.6), (120, 0.6), (1300, 0.1), (60, 0.3), (7.85, 0.05), (0.3, 0.3), 1, 1,
          fin=1.0, psh=0.2),
        P(0.4, (53,), "UDP", (4e4, 0.5), (1, 0.1), (1, 0.1), (230, 0.1), (250, 0.1), (5.9, 0.2), (0.4, 0.3), 1, 40),
    ],
}

# Attacks arrive in campaigns, not as a uniform drizzle: (flows per campaign, campaign length in hours).
CAMPAIGNS = {"DoS": (400, 2), "DDoS": (300, 1), "PortScan": (300, 0.5), "BruteForce": (150, 1),
             "WebAttack": (80, 1.5), "Botnet": (150, 12), "Infiltration": (40, 3), "Exfiltration": (30, 2)}

PAST_PRIORS = {"BENIGN": 0.80, "DoS": 0.075, "DDoS": 0.05, "PortScan": 0.045, "BruteForce": 0.015,
               "WebAttack": 0.006, "Botnet": 0.005, "Infiltration": 0.0015}
FUTURE_PRIORS = {"BENIGN": 0.79, "DoS": 0.07, "DDoS": 0.05, "PortScan": 0.045, "BruteForce": 0.015,
                 "WebAttack": 0.007, "Botnet": 0.008, "Infiltration": 0.002, "Exfiltration": 0.004}

COLUMNS = {
    "src_ip": "Src IP", "dst_ip": "Dst IP", "timestamp": "Timestamp", "dst_port": "Destination Port",
    "protocol": "Protocol", "duration": "Flow Duration", "fwd_pkts": "Total Fwd Packets",
    "bwd_pkts": "Total Backward Packets", "fwd_bytes": "Total Length of Fwd Packets",
    "bwd_bytes": "Total Length of Bwd Packets", "fwd_size": "Fwd Packet Length Mean",
    "bwd_size": "Bwd Packet Length Mean", "size_std": "Packet Length Std", "bytes_s": "Flow Bytes/s",
    "pkts_s": "Flow Packets/s", "iat_mean": "Flow IAT Mean", "iat_std": "Flow IAT Std",
    "syn": "SYN Flag Count", "ack": "ACK Flag Count", "fin": "FIN Flag Count", "rst": "RST Flag Count",
    "psh": "PSH Flag Count", "urg": "URG Flag Count", "win_fwd": "Init_Win_bytes_forward",
    "win_bwd": "Init_Win_bytes_backward", "down_up": "Down/Up Ratio", "active": "Active Mean",
    "idle": "Idle Mean", "uniq_ports": "Unique Dst Ports 10s", "conn_same_dst": "Conn Count Same Dst 10s",
    "entropy": "Payload Entropy", "label": "Label",
}


def _lognormal(rng: np.random.Generator, shape: tuple, k: int) -> np.ndarray:
    median, sigma = shape
    if median <= 0:
        return np.zeros(k)
    return rng.lognormal(np.log(median), sigma, k)


def _generate_class(rng: np.random.Generator, label: str, profiles: list[Profile], n: int) -> pd.DataFrame:
    weights = np.array([p.weight for p in profiles], dtype=float)
    which = rng.choice(len(profiles), size=n, p=weights / weights.sum())
    cols: dict[str, np.ndarray] = {}
    parts = []
    for j, prof in enumerate(profiles):
        k = int((which == j).sum())
        if k == 0:
            continue
        if prof.port_range:
            port = rng.integers(prof.ports[0], prof.ports[1] + 1, k)
        else:
            port = rng.choice(prof.ports, k)
        tcp = prof.proto == "TCP"
        fwd = np.maximum(1, np.round(_lognormal(rng, prof.fwd_pkts, k))).astype(int)
        bwd = np.round(_lognormal(rng, prof.bwd_pkts, k)).astype(int)
        cols = {
            "dst_port": port,
            "protocol": np.full(k, prof.proto),
            "duration": np.round(_lognormal(rng, prof.duration, k)),
            "fwd_pkts": fwd,
            "bwd_pkts": bwd,
            "fwd_size": _lognormal(rng, prof.fwd_size, k),
            "bwd_size": np.where(bwd > 0, _lognormal(rng, prof.bwd_size, k), 0.0),
            "entropy": np.clip(rng.normal(*prof.entropy, k), 0, 8),
            "iat_cv": _lognormal(rng, prof.iat_cv, k),
            "uniq_ports": 1 + rng.poisson(max(prof.uniq_ports - 1, 0), k),
            "conn_same_dst": 1 + rng.poisson(max(prof.conn_same_dst - 1, 0), k),
            "syn": rng.binomial(1, prof.syn, k) * tcp,
            "fin": rng.binomial(1, prof.fin, k) * tcp,
            "rst": rng.binomial(1, prof.rst, k) * tcp,
            "psh": rng.binomial(fwd, prof.psh) * tcp,
            "ack": np.maximum(fwd + bwd - 1, 0) * tcp,
            "urg": rng.binomial(1, 0.001, k) * tcp,
            "win_fwd": rng.choice(prof.windows, k) * tcp,
            "win_bwd": rng.choice((65535, 29200, 28960, 0), k) * tcp,
        }
        parts.append(pd.DataFrame(cols))
    df = pd.concat(parts, ignore_index=True)
    df["label"] = label
    return df


def _campaign_days(rng: np.random.Generator, name: str, n: int, day_from: float, day_to: float) -> np.ndarray:
    """Day offsets: uniform for benign traffic, clustered into campaigns for attacks."""
    if name not in CAMPAIGNS:
        return rng.uniform(day_from, day_to, n)
    per_campaign, hours = CAMPAIGNS[name]
    length = hours / 24
    k = max(1, int(round(n / per_campaign)))
    starts = rng.uniform(day_from, max(day_from, day_to - length), k)
    return starts[rng.integers(0, k, n)] + rng.uniform(0, length, n)


def _derive(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    seconds = df["duration"] / 1e6
    df["fwd_bytes"] = np.round(df["fwd_pkts"] * df["fwd_size"])
    df["bwd_bytes"] = np.round(df["bwd_pkts"] * df["bwd_size"])
    with np.errstate(divide="ignore", invalid="ignore"):
        df["bytes_s"] = (df["fwd_bytes"] + df["bwd_bytes"]) / seconds
        df["pkts_s"] = (df["fwd_pkts"] + df["bwd_pkts"]) / seconds
    gaps = np.maximum(df["fwd_pkts"] + df["bwd_pkts"] - 1, 1)
    df["iat_mean"] = df["duration"] / gaps
    df["iat_std"] = df["iat_cv"] * df["iat_mean"]
    df["size_std"] = np.abs(df["fwd_size"] - df["bwd_size"]) / 2 * rng.uniform(0.6, 1.4, len(df))
    df["down_up"] = np.floor(df["bwd_pkts"] / df["fwd_pkts"])
    active_share = rng.beta(2, 2, len(df))
    df["active"] = df["duration"] * active_share
    df["idle"] = df["duration"] * (1 - active_share)
    return df


def _addresses(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    n = len(df)
    internal = np.array([f"192.168.10.{i}" for i in range(2, 255)])
    servers = np.array(["192.168.10.50", "192.168.10.51", "192.168.10.19", "192.168.10.25"])
    external = np.array([f"52.{a}.{b}.{c}" for a, b, c in rng.integers(1, 255, (300, 3))])
    src = rng.choice(internal, n).astype(object)
    dst = np.where(rng.random(n) < 0.5, rng.choice(servers, n), rng.choice(external, n)).astype(object)
    label = df["label"].to_numpy()
    attacker = "205.174.165.73"
    for name in ("DoS", "PortScan", "BruteForce", "WebAttack"):
        m = label == name
        src[m], dst[m] = attacker, "192.168.10.50"
    m = label == "DDoS"
    src[m] = [f"{a}.{b}.{c}.{d}" for a, b, c, d in rng.integers(1, 255, (int(m.sum()), 4))]
    dst[m] = "192.168.10.50"
    bots = np.array(["192.168.10.5", "192.168.10.8", "192.168.10.9", "192.168.10.14", "192.168.10.15"])
    m = label == "Botnet"
    src[m], dst[m] = rng.choice(bots, int(m.sum())), "205.174.165.73"
    m = label == "Infiltration"
    src[m], dst[m] = "192.168.10.8", rng.choice(internal, int(m.sum()))
    m = label == "Exfiltration"
    src[m], dst[m] = "192.168.10.12", rng.choice(external[:5], int(m.sum()))
    df["src_ip"], df["dst_ip"] = src, dst
    return df


def generate_flows(n_samples: int = 100_000, seed: int = 42, drift: bool = True) -> pd.DataFrame:
    """Generate a labelled flow table with CIC-IDS2017-style column names."""
    rng = np.random.default_rng(seed)
    past_days = DAYS * 0.65
    n_past = int(round(n_samples * (0.65 if drift else 1.0)))
    periods = [(PAST_PRIORS, ATTACK_PROFILES, n_past, 0.0, past_days if drift else DAYS)]
    if drift:
        periods.append((FUTURE_PRIORS, DRIFTED_PROFILES, n_samples - n_past, past_days, DAYS))

    frames = []
    for priors, profiles, n, day_from, day_to in periods:
        names = list(priors)
        probs = np.array([priors[k] for k in names])
        counts = rng.multinomial(n, probs / probs.sum())
        for name, count in zip(names, counts):
            if count == 0:
                continue
            if name == "BENIGN":
                part = _generate_class(rng, name, BENIGN_PROFILES, int(count))
            else:
                noisy = int(rng.binomial(int(count), LABEL_NOISE))
                pieces = [(profiles[name], int(count) - noisy), (BENIGN_PROFILES, noisy)]
                part = pd.concat([_generate_class(rng, name, profs, k) for profs, k in pieces if k > 0],
                                 ignore_index=True)
            offsets = _campaign_days(rng, name, len(part), day_from, day_to)
            part["timestamp"] = START + pd.to_timedelta(offsets, unit="D")
            frames.append(part)
    df = pd.concat(frames, ignore_index=True)
    df = _addresses(_derive(df, rng), rng)

    # CIC-IDS quirks: zero-duration probes produce Infinity rates; a few rates are missing.
    zero = (df["label"] == "PortScan").to_numpy() & (rng.random(len(df)) < 0.02)
    df.loc[zero, "duration"] = 0
    df.loc[zero, ["bytes_s", "pkts_s"]] = np.inf
    df.loc[rng.random(len(df)) < 0.0005, "bytes_s"] = np.nan

    df = df.sort_values("timestamp").reset_index(drop=True)
    df["timestamp"] = df["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
    dupes = df.sample(frac=0.005, random_state=seed)  # exact duplicate flows, as in real captures
    df = pd.concat([df, dupes]).sort_values("timestamp", kind="stable").reset_index(drop=True)

    float_cols = ["fwd_size", "bwd_size", "size_std", "bytes_s", "pkts_s", "iat_mean", "iat_std",
                  "active", "idle", "entropy"]
    df[float_cols] = df[float_cols].round(3)
    return df[list(COLUMNS)].rename(columns=COLUMNS)
