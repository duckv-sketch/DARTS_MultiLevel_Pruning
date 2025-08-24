import torch
import torch.nn as nn
import torch.nn.functional as F
from operations import OPS

class Cell(nn.Module):
    def __init__(self, genotype, C_prev_prev, C_prev, C, reduction, reduction_prev):
        super(Cell, self).__init__()
        self.reduction = reduction

        if reduction_prev:
            self.preprocess0 = nn.Sequential(
                nn.ReLU(),
                nn.Conv1d(C_prev_prev, C, 1, stride=2, padding=0, bias=False),
                nn.BatchNorm1d(C)
            )
        else:
            self.preprocess0 = nn.Sequential(
                nn.ReLU(),
                nn.Conv1d(C_prev_prev, C, 1, stride=1, padding=0, bias=False),
                nn.BatchNorm1d(C)
            )

        self.preprocess1 = nn.Sequential(
            nn.ReLU(),
            nn.Conv1d(C_prev, C, 1, stride=1, padding=0, bias=False),
            nn.BatchNorm1d(C)
        )

        op_names, indices = zip(*genotype.reduce if reduction else genotype.normal)
        self._compile(C, op_names, indices, reduction)

    def _compile(self, C, op_names, indices, reduction):
        self._steps = len(op_names) // 2
        self._ops = nn.ModuleList()
        self._indices = indices
        for name, index in zip(op_names, indices):
            stride = 2 if reduction and index < 2 else 1
            op = OPS[name](C, stride, True)
            self._ops.append(op)

    def forward(self, s0, s1):
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)
        states = [s0, s1]
        for i in range(self._steps):
            h1 = states[self._indices[2 * i]]
            h2 = states[self._indices[2 * i + 1]]
            op1 = self._ops[2 * i]
            op2 = self._ops[2 * i + 1]

            out1 = op1(h1)
            out2 = op2(h2)

            # Resize if mismatch in time dimension
            if out1.size(2) != out2.size(2):
                min_len = min(out1.size(2), out2.size(2))
                out1 = out1[..., :min_len]
                out2 = out2[..., :min_len]

            s = out1 + out2
            states.append(s)
        return torch.cat(states[2:], dim=1)


class FinalNetwork(nn.Module):
    def __init__(self, C, num_classes, layers, genotype, dropout_prob=0.3):
        super(FinalNetwork, self).__init__()
        self.C = C
        self.num_classes = num_classes
        self.layers = layers
        self.genotype = genotype
        self.dropout_prob = dropout_prob

        # Stem layer
        self.stem = nn.Sequential(
            nn.Conv1d(1, C, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(C),
            nn.ReLU(inplace=True)
        )

        # Build cells
        C_prev_prev, C_prev, C_curr = C, C, C
        self.cells = nn.ModuleList()
        reduction_prev = False

        for i in range(layers):
            if i in [layers // 3, 2 * layers // 3]:
                C_curr *= 2
                reduction = True
            else:
                reduction = False
            cell = Cell(genotype, C_prev_prev, C_prev, C_curr, reduction, reduction_prev)
            reduction_prev = reduction
            self.cells.append(cell)
            C_prev_prev, C_prev = C_prev, cell._steps * C_curr

        # Pool + Dropout + Classifier
        self.global_pooling = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(p=self.dropout_prob)
        self.classifier = nn.Linear(C_prev, num_classes)

    def forward(self, x):
        s0 = s1 = self.stem(x)
        for cell in self.cells:
            s0, s1 = s1, cell(s0, s1)
        out = self.global_pooling(s1).squeeze(-1)
        out = self.dropout(out)  # prevent overfitting
        logits = self.classifier(out)
        return logits

