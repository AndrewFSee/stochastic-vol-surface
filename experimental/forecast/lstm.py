"""LSTM and LSTM-GARCH volatility forecasters (same interface as src.forecast.models).

* ``Lstm`` sees, over the last ``seq_len`` days, the inputs HAR + implied
  uses — log daily/weekly/monthly realised variance, the return, and log
  implied variance — and predicts log realised variance over the next *h*
  days.
* ``LstmGarch`` additionally sees GJR-GARCH's own *h*-day variance forecast
  at each step: the usual hybrid design, letting the network correct the
  econometric forecast rather than learn the dynamics from scratch.

Kept in experimental/ because torch is an optional extra and, on SPY
2013-2023, neither beat HAR + implied (see the forecast report).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from src.forecast.models import Garch, _implied_var


def _inputs(d: pd.DataFrame, h: int) -> pd.DataFrame:
    with np.errstate(divide="ignore", invalid="ignore"):
        X = pd.DataFrame({
            "log_rv_d": np.log(d["rv_d"]),
            "log_rv_w": np.log(d["rv_w"]),
            "log_rv_m": np.log(d["rv_m"]),
            "ret": d["ret"] * 10.0,
            "log_iv": np.log(_implied_var(d, h)),
        }, index=d.index)
    return X.replace([np.inf, -np.inf], np.nan)


@dataclass
class Lstm:
    name: str = "LSTM"
    seq_len: int = 22
    hidden: int = 16
    epochs: int = 60
    patience: int = 8
    lr: float = 2e-3
    seed: int = 0
    use_garch: bool = False
    net_: object = field(default=None, repr=False)
    mu_: Optional[np.ndarray] = field(default=None, repr=False)
    sd_: Optional[np.ndarray] = field(default=None, repr=False)
    ymu_: float = 0.0
    ysd_: float = 1.0
    smear_: float = 1.0
    garch_: Optional[Garch] = field(default=None, repr=False)

    # ── features ─────────────────────────────────────────────────────────
    def _features(self, history: pd.DataFrame, h: int) -> pd.DataFrame:
        X = _inputs(history, h)
        if self.use_garch:
            g = self.garch_.predict(history.iloc[1:], h, history)
            X["log_garch"] = np.log(g.reindex(history.index))
        return X

    def _windows(self, X: np.ndarray, ends: np.ndarray) -> np.ndarray:
        return np.stack([X[e - self.seq_len + 1: e + 1] for e in ends])

    # ── fit / predict ────────────────────────────────────────────────────
    def fit(self, train: pd.DataFrame, h: int) -> "Lstm":
        import torch
        from torch import nn

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        if self.use_garch:
            self.garch_ = Garch("gjr").fit(train, h)
        X = self._features(train, h)
        y = np.log(train[f"y_{h}"])
        Xv = X.to_numpy()
        ok_row = np.isfinite(Xv).all(axis=1)
        ends = np.array([e for e in range(self.seq_len - 1, len(X))
                         if ok_row[e - self.seq_len + 1: e + 1].all() and np.isfinite(y.iloc[e])])
        self.mu_ = np.nanmean(Xv[ok_row], 0)
        self.sd_ = np.nanstd(Xv[ok_row], 0) + 1e-8
        Z = (Xv - self.mu_) / self.sd_
        W = self._windows(Z, ends).astype(np.float32)
        Y = y.to_numpy()[ends]
        self.ymu_, self.ysd_ = float(Y.mean()), float(Y.std() + 1e-8)
        Yz = ((Y - self.ymu_) / self.ysd_).astype(np.float32)

        # Time-ordered validation: the last 20% of windows, with an h-day gap
        # so no validation target overlaps a training target.
        n_val = max(int(0.2 * len(W)), 50)
        cut = len(W) - n_val
        tr, va = slice(0, max(cut - h, 1)), slice(cut, len(W))

        class Net(nn.Module):
            def __init__(s, n_in, hid):
                super().__init__()
                s.lstm = nn.LSTM(n_in, hid, batch_first=True)
                s.drop = nn.Dropout(0.1)
                s.out = nn.Linear(hid, 1)

            def forward(s, x):
                o, _ = s.lstm(x)
                return s.out(s.drop(o[:, -1])).squeeze(-1)

        net = Net(W.shape[2], self.hidden)
        opt = torch.optim.Adam(net.parameters(), lr=self.lr, weight_decay=1e-4)
        Wt, Yt = torch.from_numpy(W), torch.from_numpy(Yz)
        best, best_state, bad = np.inf, None, 0
        for _ in range(self.epochs):
            net.train()
            perm = torch.randperm(tr.stop)
            for i in range(0, len(perm), 64):
                idx = perm[i:i + 64]
                opt.zero_grad()
                loss = nn.functional.mse_loss(net(Wt[idx]), Yt[idx])
                loss.backward()
                opt.step()
            net.eval()
            with torch.no_grad():
                v = float(nn.functional.mse_loss(net(Wt[va]), Yt[va]))
            if v < best - 1e-5:
                best, bad = v, 0
                best_state = {k: t.clone() for k, t in net.state_dict().items()}
            else:
                bad += 1
                if bad >= self.patience:
                    break
        net.load_state_dict(best_state)
        net.eval()
        self.net_ = net
        with torch.no_grad():
            fit_log = net(Wt).numpy() * self.ysd_ + self.ymu_
        self.smear_ = float(np.mean(np.exp(Y - fit_log)))
        return self

    def predict(self, rows: pd.DataFrame, h: int, history: pd.DataFrame) -> pd.Series:
        import torch

        X = self._features(history, h)
        Z = (X.to_numpy() - self.mu_) / self.sd_
        pos = history.index.get_indexer(rows.index)
        out = pd.Series(np.nan, index=rows.index)
        ok = [p for p in pos if p >= self.seq_len - 1
              and np.isfinite(Z[p - self.seq_len + 1: p + 1]).all()]
        if ok:
            W = torch.from_numpy(self._windows(Z, np.array(ok)).astype(np.float32))
            with torch.no_grad():
                pred = self.net_(W).numpy() * self.ysd_ + self.ymu_
            out.loc[history.index[ok]] = np.exp(pred) * self.smear_
        return out


def lstm() -> Lstm:
    return Lstm(name="LSTM")


def lstm_garch() -> Lstm:
    return Lstm(name="LSTM-GARCH", use_garch=True)
