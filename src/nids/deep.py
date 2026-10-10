"""PyTorch multilayer perceptron with focal loss, wrapped as a scikit-learn classifier.

Focal loss (Lin et al., 2017) multiplies cross-entropy by (1 - p_t)^gamma. Flows the network already
classifies confidently (most benign traffic) contribute almost nothing, so gradient updates are
dominated by the hard, usually rare, attack flows. Optional per-class alpha weights add a
frequency-based rebalancing on top.

torch is an optional dependency: without it the ``deep_mlp`` model is skipped by the benchmark.
"""

from __future__ import annotations

import copy
import logging

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

log = logging.getLogger(__name__)

try:
    import torch
    import torch.nn.functional as F
    from torch import nn

    TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only when torch is absent
    TORCH_AVAILABLE = False


if TORCH_AVAILABLE:

    class FocalLoss(nn.Module):
        def __init__(self, gamma: float = 2.0, alpha: torch.Tensor | None = None):
            super().__init__()
            self.gamma = gamma
            self.register_buffer("alpha", alpha)

        def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
            log_p = F.log_softmax(logits, dim=1).gather(1, target.unsqueeze(1)).squeeze(1)
            loss = -((1.0 - log_p.exp()) ** self.gamma) * log_p
            if self.alpha is not None:
                loss = loss * self.alpha[target]
            return loss.mean()

    def _build_mlp(n_in: int, hidden: tuple[int, ...], n_out: int, dropout: float) -> nn.Module:
        layers: list[nn.Module] = []
        prev = n_in
        for width in hidden:
            layers += [nn.Linear(prev, width), nn.BatchNorm1d(width), nn.ReLU(), nn.Dropout(dropout)]
            prev = width
        layers.append(nn.Linear(prev, n_out))
        return nn.Sequential(*layers)


class TorchMLPClassifier(ClassifierMixin, BaseEstimator):
    """Feed-forward network trained with focal loss and early stopping on validation macro-F1."""

    def __init__(self, hidden_dims=(256, 128, 64), dropout=0.2, lr=1e-3, weight_decay=1e-4, batch_size=512,
                 max_epochs=30, patience=5, gamma=2.0, class_weight=None, val_fraction=0.1, random_state=42,
                 verbose=False):
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.gamma = gamma
        self.class_weight = class_weight
        self.val_fraction = val_fraction
        self.random_state = random_state
        self.verbose = verbose

    def fit(self, X, y):
        if not TORCH_AVAILABLE:
            raise ImportError("deep_mlp needs PyTorch: pip install torch --index-url https://download.pytorch.org/whl/cpu")
        torch.manual_seed(self.random_state)
        X = np.asarray(X, dtype=np.float32)
        self.classes_, y_idx = np.unique(np.asarray(y), return_inverse=True)
        n_classes = len(self.classes_)
        self.n_features_in_ = X.shape[1]

        idx = np.arange(len(y_idx))
        try:
            tr, va = train_test_split(idx, test_size=self.val_fraction, stratify=y_idx, random_state=self.random_state)
        except ValueError:  # a class too small to stratify
            tr, va = train_test_split(idx, test_size=self.val_fraction, random_state=self.random_state)

        alpha = None
        if self.class_weight == "balanced":
            counts = np.bincount(y_idx[tr], minlength=n_classes).astype(float)
            # Square root of inverse frequency: full inverse weights (x500 for the rarest class) destabilise training.
            w = np.sqrt(counts.sum() / (n_classes * np.maximum(counts, 1.0)))
            alpha = torch.tensor(w / w.mean(), dtype=torch.float32)

        model = _build_mlp(X.shape[1], tuple(self.hidden_dims), n_classes, self.dropout)
        loss_fn = FocalLoss(self.gamma, alpha)
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)

        X_tr, y_tr = torch.from_numpy(X[tr]), torch.from_numpy(y_idx[tr].astype(np.int64))
        X_va, y_va = X[va], y_idx[va]
        generator = torch.Generator().manual_seed(self.random_state)
        best_score, best_state, stale = -1.0, None, 0
        self.history_ = []
        for epoch in range(self.max_epochs):
            model.train()
            order = torch.randperm(len(X_tr), generator=generator)
            total = 0.0
            for start in range(0, len(order), self.batch_size):
                batch = order[start:start + self.batch_size]
                if len(batch) < 2:  # BatchNorm needs at least two rows
                    continue
                optimizer.zero_grad()
                loss = loss_fn(model(X_tr[batch]), y_tr[batch])
                loss.backward()
                optimizer.step()
                total += loss.item() * len(batch)
            score = f1_score(y_va, self._predict_idx(model, X_va), average="macro")
            scheduler.step(score)
            self.history_.append({"epoch": epoch + 1, "loss": total / len(order), "val_macro_f1": score})
            if self.verbose:
                log.info("epoch %d loss %.4f val macro-F1 %.4f", epoch + 1, total / len(order), score)
            if score > best_score + 1e-4:
                best_score, best_state, stale = score, copy.deepcopy(model.state_dict()), 0
            else:
                stale += 1
                if stale >= self.patience:
                    break
        model.load_state_dict(best_state)
        model.eval()
        self.model_ = model
        self.best_val_macro_f1_ = best_score
        return self

    @staticmethod
    def _logits(model, X: np.ndarray, batch: int = 8192):
        model.eval()
        with torch.no_grad():
            return torch.cat([model(torch.from_numpy(X[i:i + batch])) for i in range(0, len(X), batch)])

    def _predict_idx(self, model, X: np.ndarray) -> np.ndarray:
        return self._logits(model, X).argmax(dim=1).numpy()

    def predict_proba(self, X):
        X = np.asarray(X, dtype=np.float32)
        return torch.softmax(self._logits(self.model_, X), dim=1).numpy()

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(axis=1)]
