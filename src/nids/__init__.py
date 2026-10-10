"""NetSentinel: imbalance-aware network intrusion detection."""

__version__ = "0.1.0"

BENIGN = "Benign"
# Label given to flows the classifier calls benign but the anomaly detector finds unlike any
# benign traffic seen in training: the candidate zero-day / novel-attack bucket.
UNKNOWN = "Unknown-Anomaly"
