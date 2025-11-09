import os
import math
import random
import time
from typing import List, Tuple, Optional, Dict

import numpy as np
import torch
import matplotlib.pyplot as plt


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class NoamLR(torch.optim.lr_scheduler._LRScheduler):
    """
    lr = d_model^-0.5 * min(step^-0.5, step * warmup^-1.5)
    """
    def __init__(self, optimizer, d_model, warmup_steps, last_epoch=-1):
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = max(1, self.last_epoch + 1)
        scale = (self.d_model ** -0.5) * min(step ** -0.5,
                                             step * (self.warmup_steps ** -1.5))
        return [base_lr * scale for base_lr in self.base_lrs]


class LabelSmoothingLoss(torch.nn.Module):
    def __init__(self, label_smoothing: float, tgt_vocab: int, ignore_index: int = 0):
        super().__init__()
        assert 0.0 <= label_smoothing < 1.0
        self.ignore_index = ignore_index
        self.confidence = 1.0 - label_smoothing
        self.smoothing = label_smoothing
        self.tgt_vocab = tgt_vocab

    def forward(self, pred, target):
        # pred: [B,L,V], target: [B,L]
        pred = pred.view(-1, pred.size(-1))
        target = target.view(-1)
        mask = target != self.ignore_index
        n_valid = mask.sum()

        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.tgt_vocab - 2))
            true_dist.scatter_(1, target.unsqueeze(1), self.confidence)
            true_dist[:, self.ignore_index] = 0
            true_dist[target == self.ignore_index] = 0

        loss = torch.sum(-true_dist[mask] * torch.log_softmax(pred[mask], dim=-1)) / n_valid.clamp(min=1)
        return loss


# --------- Metrics helpers ---------

def safe_bleu(hyps: List[str], refs: List[str]) -> float:
    try:
        import sacrebleu
        return sacrebleu.corpus_bleu(hyps, [refs]).score
    except Exception:
        match = sum(int(h.strip() == r.strip()) for h, r in zip(hyps, refs))
        return 100.0 * match / max(1, len(hyps))


def ensure_dir(path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)


def plot_series(xs: List[float], ys: List[float], ylabel: str, out_png: str, title: Optional[str] = None, marker: str = "o"):
    ensure_dir(out_png)
    plt.figure()
    if xs and ys:
        plt.plot(xs, ys, marker=marker)
    plt.xlabel("Epoch" if "steps" not in os.path.basename(out_png) else "Step")
    plt.ylabel(ylabel)
    if title:
        plt.title(title)
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_png, bbox_inches="tight")
    plt.close()


def plot_steps(xs_steps: List[int], ys: List[float], ylabel: str, out_png: str, title: Optional[str] = None, marker: str = ""):
    ensure_dir(out_png)
    plt.figure()
    if xs_steps and ys:
        if marker:
            plt.plot(xs_steps, ys, marker=marker)
        else:
            plt.plot(xs_steps, ys)
    plt.xlabel("Step")
    plt.ylabel(ylabel)
    if title:
        plt.title(title)
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_png, bbox_inches="tight")
    plt.close()


def append_csv(path: str, header: List[str], row: List):
    exist = os.path.isfile(path)
    ensure_dir(path)
    with open(path, "a", encoding="utf-8") as f:
        if not exist:
            f.write(",".join(header) + "\n")
        f.write(",".join(str(x) for x in row) + "\n")
