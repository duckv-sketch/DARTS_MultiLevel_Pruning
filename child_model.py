# sleep_nas/child_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from search_space import OPS

class MixedOp(nn.Module):
    def __init__(self, C):
        super().__init__()
        self._ops = nn.ModuleList()
        for primitive in OPS:
            op = OPS[primitive](C, stride=1, affine=True)
            self._ops.append(op)

    def forward(self, x, weights):
        return sum(w * op(x) for w, op in zip(weights, self._ops))

class Cell(nn.Module):
    def __init__(self, steps, C):
        super().__init__()
        self.steps = steps
        self.ops = nn.ModuleList()
        for i in range(self.steps):
            for j in range(i + 1):
                self.ops.append(MixedOp(C))

    def forward(self, x, weights):
        states = [x]
        offset = 0
        for i in range(self.steps):
            s = sum(self.ops[offset + j](h, weights[offset + j]) for j, h in enumerate(states))
            offset += len(states)
            states.append(s)
        return sum(states[-self.steps:])

class Network(nn.Module):
    def __init__(self, C_in, C, num_classes, layers, criterion, steps=4):
        super().__init__()
        self._criterion = criterion
        self.stem = nn.Sequential(
            nn.Conv1d(C_in, C, 3, padding=1, bias=False),
            nn.BatchNorm1d(C)
        )
        self.cells = nn.ModuleList()
        for i in range(layers):
            self.cells.append(Cell(steps, C))

        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(C, num_classes)

    def forward(self, x, weights):
        x = self.stem(x)
        for cell in self.cells:
            x = cell(x, weights)
        x = self.global_pool(x).squeeze(-1)
        return self.classifier(x)

    def loss(self, x, y, weights):
        logits = self(x, weights)
        loss = self._criterion(logits, y.long())
        pred = torch.argmax(logits, dim=1)
        acc = (pred == y).float().mean().item()
        return loss, acc
