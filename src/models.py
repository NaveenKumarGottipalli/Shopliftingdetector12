"""Zero-shot prompt scoring, Top-3 MIL MLP head, scene-mean centering, and Group-DRO."""
import numpy as np
import torch
import torch.nn as nn

def compute_zs_scores(F: np.ndarray, TP: np.ndarray, TN: np.ndarray, agg: str) -> np.ndarray:
    """Compute top-3 temporally pooled zero-shot cosine margin scores."""
    sp, sn = F @ TP.T, F @ TN.T
    z = (sp.max(axis=-1) - sn.max(axis=-1)) if agg == "max" else (sp.mean(axis=-1) - sn.mean(axis=-1))
    return np.sort(z, axis=1)[:, -3:].mean(axis=1)

def center_features(X: np.ndarray, keys: np.ndarray, doms: np.ndarray, min_n: int = 8) -> np.ndarray:
    """Label-free scene-mean subtraction; scenes with < min_n clips fall back to split domain mean."""
    sm = X.mean(axis=1)
    out = X.copy()
    dmu = {d: sm[doms == d].mean(axis=0) for d in np.unique(doms)}
    for k in np.unique(keys):
        ii = np.where(keys == k)[0]
        mu = sm[ii].mean(axis=0) if len(ii) >= min_n else dmu[doms[ii[0]]]
        out[ii] = X[ii] - mu
    return out

class TopKMILHead(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 1),
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)

def bag_pool(z: torch.Tensor, k: int = 3) -> torch.Tensor:
    return z.topk(k, dim=1).values.mean(dim=1)

def train_head(Xtr: np.ndarray, ytr: np.ndarray, gtr: np.ndarray, seed: int,
               epochs: int = 25, dro: bool = False, device: torch.device = None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    np.random.seed(seed)
    Xg = torch.from_numpy(Xtr).float().to(device)
    mu = Xg.reshape(-1, Xg.shape[-1]).mean(dim=0)
    sd = Xg.reshape(-1, Xg.shape[-1]).std(dim=0) + 1e-5
    Xg = (Xg - mu) / sd
    yt = torch.from_numpy(ytr).float().to(device)
    gt = torch.from_numpy(np.asarray(gtr)).long().to(device)

    m = TopKMILHead(Xg.shape[-1]).to(device)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-2)
    pw = torch.tensor(max(1.0, (ytr == 0).sum() / max(1, (ytr == 1).sum())) ** 0.5, device=device)
    lossf = nn.BCEWithLogitsLoss(pos_weight=pw, reduction="none")
    G = int(gt.max()) + 1 if len(gt) else 1
    q = torch.ones(G, device=device) / G
    n = len(Xg)

    for _ in range(epochs):
        m.train()
        perm = torch.randperm(n, device=device)
        for i in range(0, n, 64):
            b = perm[i : i + 64]
            l = lossf(bag_pool(m(Xg[b])), yt[b])
            if dro:
                gb = gt[b]
                gl = torch.zeros(G, device=device).scatter_add_(0, gb, l.detach()) / torch.bincount(gb, minlength=G).clamp(min=1)
                q = q * torch.exp(0.01 * gl)
                q = q / q.sum()
                l = l * (q[gb] * G).clamp(max=5.0)
            loss = l.mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

    m.eval()
    def predict(X_eval: np.ndarray) -> np.ndarray:
        with torch.inference_mode():
            Xe = (torch.from_numpy(X_eval).float().to(device) - mu) / sd
            if len(Xe) == 0:
                return np.zeros(0)
            return torch.cat([bag_pool(m(Xe[i : i + 1024])) for i in range(0, len(Xe), 1024)]).cpu().numpy()
    return predict

METHODS = {
    "base": dict(c=False, dro=False),
    "center": dict(c=True, dro=False),
    "dro": dict(c=False, dro=True),
    "center+dro": dict(c=True, dro=True),
}

def fit_predict(X: np.ndarray, y: np.ndarray, scene: np.ndarray, sid: np.ndarray,
                dom: np.ndarray, tr: np.ndarray, te: np.ndarray,
                method: str, seed: int, epochs: int = 25, device: torch.device = None) -> np.ndarray:
    cfg = METHODS[method]
    Xtr, Xte = X[tr], X[te]
    if cfg["c"]:
        Xtr = center_features(Xtr, scene[tr], dom[tr])
        Xte = center_features(Xte, scene[te], dom[te])
    gtr = sid[tr] if cfg["dro"] else np.zeros(len(tr), dtype=int)
    return train_head(Xtr, y[tr], gtr, seed, epochs=epochs, dro=cfg["dro"], device=device)(Xte)
