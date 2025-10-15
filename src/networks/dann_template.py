# dann_grl.py
import math
import itertools
from typing import Optional, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


# -------------------------
# Gradient Reversal Layer
# -------------------------
class _GRLFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambd * grad_output, None


class GRL(nn.Module):
    def __init__(self, lambd: float = 1.0):
        super().__init__()
        self.lambd = lambd

    def set_lambda(self, lambd: float):
        self.lambd = lambd

    def forward(self, x):
        return _GRLFn.apply(x, self.lambd)


# -------------------------
# Example backbones & heads
# Swap FeatureExtractor with your own model.
# -------------------------
class FeatureExtractor(nn.Module):
    """
    Example MLP feature extractor for vector inputs.
    Replace with your CNN/Transformer/etc. for images/sequences.
    """
    def __init__(self, input_dim: int, feat_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256), nn.BatchNorm1d(256), nn.ReLU(inplace=True),
            nn.Linear(256, 256), nn.BatchNorm1d(256), nn.ReLU(inplace=True),
            nn.Linear(256, feat_dim)
        )

    def forward(self, x):
        return self.net(x)


class LabelPredictor(nn.Module):
    """
    Task head. For classification set num_outputs = num_classes and use CrossEntropy.
    For regression set num_outputs = target_dim and use MSE.
    """
    def __init__(self, feat_dim: int, num_outputs: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feat_dim, 128), nn.ReLU(inplace=True),
            nn.Linear(128, num_outputs)
        )

    def forward(self, f):
        return self.net(f)


class DomainClassifier(nn.Module):
    """
    Domain head. Binary domain label: 0=source, 1=target (BCEWithLogits).
    """
    def __init__(self, feat_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feat_dim, 128), nn.ReLU(inplace=True),
            nn.Linear(128, 1)  # logits
        )

    def forward(self, f_rev):
        return self.net(f_rev).squeeze(1)  # (B,)


# -------------------------
# Full DANN model
# -------------------------
class DANN(nn.Module):
    def __init__(self,
                 feature_extractor: nn.Module,
                 label_predictor: nn.Module,
                 domain_classifier: nn.Module):
        super().__init__()
        self.f = feature_extractor
        self.y = label_predictor
        self.d = domain_classifier
        self.grl = GRL(lambd=0.0)

    @torch.no_grad()
    def set_lambda(self, lambd: float):
        self.grl.set_lambda(lambd)

    def forward(self, x, return_features: bool = False):
        feats = self.f(x)
        logits_y = self.y(feats)
        logits_d = self.d(self.grl(feats))
        if return_features:
            return logits_y, logits_d, feats
        return logits_y, logits_d


# -------------------------
# Lambda schedule from DANN paper
# p in [0, 1] over training progress; lambd ramps from ~0 to 1
# -------------------------
def dann_lambda(p: float) -> float:
    return 2.0 / (1.0 + math.exp(-10.0 * p)) - 1.0


# -------------------------
# Training step utilities
# -------------------------
def _make_domain_labels(n_source: int, n_target: int, device):
    y_dom_src = torch.zeros(n_source, device=device)  # 0 for source
    y_dom_tgt = torch.ones(n_target, device=device)   # 1 for target
    return y_dom_src, y_dom_tgt


def _task_loss(logits, y, task_type: Literal["classification", "regression"]):
    if task_type == "classification":
        return F.cross_entropy(logits, y)
    else:
        # regression: assume logits and y are float tensors with same shape
        return F.mse_loss(logits, y)


# -------------------------
# Training loop
# -------------------------
def train_dann(
    model: DANN,
    source_loader: DataLoader,    # batches of (x_s, y_s)
    target_loader: DataLoader,    # batches of (x_t, _) unlabeled target
    optimizer: torch.optim.Optimizer,
    epochs: int,
    device: str = "cuda",
    task_type: Literal["classification", "regression"] = "classification",
    domain_loss_weight: float = 1.0,
    max_steps_per_epoch: Optional[int] = None,
    eval_fn=None,                 # optional: fn(model) -> dict printed each epoch
    grad_clip: Optional[float] = None,
):
    model.to(device)
    model.train()

    total_steps = epochs * min(len(source_loader), len(target_loader))
    step_idx = 0

    for epoch in range(1, epochs + 1):
        # Iterate pairs of source/target batches
        s_iter = iter(source_loader)
        t_iter = iter(target_loader)

        num_batches = min(len(source_loader), len(target_loader))
        if max_steps_per_epoch:
            num_batches = min(num_batches, max_steps_per_epoch)

        running = {"task": 0.0, "domain": 0.0, "total": 0.0}

        for _ in range(num_batches):
            try:
                x_s, y_s = next(s_iter)
            except StopIteration:
                s_iter = iter(source_loader)
                x_s, y_s = next(s_iter)
            try:
                x_t = next(t_iter)[0]  # assume target loader yields (x_t, _)
            except StopIteration:
                t_iter = iter(target_loader)
                x_t = next(t_iter)[0]

            x_s = x_s.to(device, non_blocking=True)
            y_s = y_s.to(device, non_blocking=True)
            x_t = x_t.to(device, non_blocking=True)

            # Progress & lambda schedule
            p = step_idx / max(1, total_steps - 1)
            lambd = dann_lambda(p) * domain_loss_weight
            model.set_lambda(lambd)

            optimizer.zero_grad(set_to_none=True)

            # --- Source forward (task + domain) ---
            logits_y_s, logits_d_s = model(x_s)  # GRL applied inside for domain
            loss_task = _task_loss(logits_y_s, y_s, task_type)

            # --- Target forward (domain only) ---
            # We only need domain logits; task labels are not available
            with torch.no_grad():
                feats_t = model.f(x_t)
            logits_d_t = model.d(model.grl(feats_t))

            # Domain labels
            y_dom_src, y_dom_tgt = _make_domain_labels(len(x_s), len(x_t), device)
            loss_dom = F.binary_cross_entropy_with_logits(
                torch.cat([logits_d_s, logits_d_t], dim=0),
                torch.cat([y_dom_src, y_dom_tgt], dim=0)
            )

            loss = loss_task + loss_dom  # GRL already flips gradients wrt features

            loss.backward()
            if grad_clip is not None:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            running["task"] += loss_task.item()
            running["domain"] += loss_dom.item()
            running["total"] += loss.item()
            step_idx += 1

        epoch_stats = {k: v / num_batches for k, v in running.items()}
        msg = (f"[Epoch {epoch:03d}] "
               f"task={epoch_stats['task']:.4f}  "
               f"domain={epoch_stats['domain']:.4f}  "
               f"total={epoch_stats['total']:.4f}  "
               f"lambda~{lambd:.3f}")
        if eval_fn is not None:
            model.eval()
            with torch.no_grad():
                eval_stats = eval_fn(model)
            model.train()
            if isinstance(eval_stats, dict):
                msg += "  " + "  ".join(f"{k}={v:.4f}" for k, v in eval_stats.items())
        print(msg)


# -------------------------
# Minimal usage example
# -------------------------
if __name__ == "__main__":
    """
    This demo assumes:
      - vector inputs of dimension D=100
      - classification with C=10 classes
    Replace the FeatureExtractor with your own network and adapt loaders.
    """
    import os
    torch.manual_seed(0)

    D, FEAT, C = 100, 128, 10

    # Dummy datasets (replace with real ones)
    class DummySrc(torch.utils.data.Dataset):
        def __len__(self): return 2048
        def __getitem__(self, i):
            x = torch.randn(D)
            y = torch.randint(0, C, (1,)).item()
            return x, y

    class DummyTgt(torch.utils.data.Dataset):
        def __len__(self): return 2048
        def __getitem__(self, i):
            x = torch.randn(D) + 0.5  # slight domain shift
            y = -1  # unused
            return x, y

    source_loader = DataLoader(DummySrc(), batch_size=64, shuffle=True, num_workers=0)
    target_loader = DataLoader(DummyTgt(), batch_size=64, shuffle=True, num_workers=0)

    # Build model
    f = FeatureExtractor(input_dim=D, feat_dim=FEAT)
    y_head = LabelPredictor(feat_dim=FEAT, num_outputs=C)       # CrossEntropy
    d_head = DomainClassifier(feat_dim=FEAT)                    # BCEWithLogits
    model = DANN(f, y_head, d_head)

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-3, weight_decay=1e-4)

    # Optional evaluation on source (since we don't label target here)
    @torch.no_grad()
    def eval_on_source(m: DANN):
        m.eval()
        correct, total, loss = 0, 0, 0.0
        for x, y in source_loader:
            x, y = x.cuda(), y.cuda()
            logits_y, _ = m(x)
            loss += F.cross_entropy(logits_y, y).item()
            pred = logits_y.argmax(1)
            correct += (pred == y).sum().item()
            total += y.numel()
        return {"src_loss": loss / len(source_loader), "src_acc": correct / total}

    # Train
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_dann(
        model=model,
        source_loader=source_loader,
        target_loader=target_loader,
        optimizer=optimizer,
        epochs=5,
        device=device,
        task_type="classification",       # switch to "regression" + MSE if needed
        domain_loss_weight=1.0,           # global weight; schedule is applied inside
        max_steps_per_epoch=None,
        eval_fn=eval_on_source
    )
