import torch
import torch.nn.functional as F
import numpy as np
from collections import defaultdict
from sklearn.preprocessing import LabelEncoder

# Compute per-sample errors based on softmax difference from one-hot truth
def compute_errors(model, loader, device):
    model.eval()
    errors, labels = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device).squeeze(-1)
            y = y.to(device)
            logits = model(x)
            probs = torch.softmax(logits, dim=1)
            onehot = F.one_hot(y, num_classes=logits.shape[1]).float()
            diff = torch.norm(probs - onehot, dim=1)  # L2 error
            errors.extend(diff.cpu().numpy())
            labels.extend(y.cpu().numpy())
    return np.array(errors), np.array(labels)

# Error history tracker for each sample ID
class ErrorHistory:
    def __init__(self, window_size):
        self.window_size = window_size
        self.history = defaultdict(list)

    def update(self, sample_ids, errors):
        for sid, e in zip(sample_ids, errors):
            self.history[sid].append(e)

    def compute_voe(self):
        voe = {}
        for sid, hist in self.history.items():
            if len(hist) >= self.window_size:
                recent = hist[-self.window_size:]
                mean = np.mean(recent)
                var = np.mean([(v - mean) ** 2 for v in recent])
                voe[sid] = var
        return voe

# Prune class-balanced using Eq.7
def compute_prune_limits(labels, n_rounds):
    le = LabelEncoder()
    y = le.fit_transform(labels)
    class_sizes = np.bincount(y)
    M = len(class_sizes)
    P = M * (M - 1) / 2
    b = np.mean([min(class_sizes[i], class_sizes[j]) / max(class_sizes[i], class_sizes[j]) 
                 for i in range(M) for j in range(i + 1, M)])
    N = int(n_rounds * (1 - b ** 2) + 1)
    limits = {i: int(sz / N) for i, sz in enumerate(class_sizes)}
    return limits, le

# Prune samples by VoE ranking (low or high)
def prune_voe(voe_scores, labels, limit_dict, prune_ratio, mode='low'):
    sorted_voe = sorted(voe_scores.items(), key=lambda x: x[1], reverse=(mode == 'high'))
    keep_ids = set()
    class_counts = defaultdict(int)
    labels = np.array(labels)

    total_to_keep = int(len(voe_scores) * (1 - prune_ratio))

    for sid, _ in sorted_voe:
        cls = labels[sid]
        if class_counts[cls] < limit_dict[cls]:
            keep_ids.add(sid)
            class_counts[cls] += 1
        if len(keep_ids) >= total_to_keep:
            break

    return list(keep_ids)