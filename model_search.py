import torch
import torch.nn as nn
import torch.nn.functional as F
from operations import OPS

class MixedOp(nn.Module):
    def __init__(self, C, stride):
        super().__init__()
        self.ops = nn.ModuleList()
        for primitive in OPS:
            op = OPS[primitive](C, stride, True)
            self.ops.append(op)

    def forward(self, x, weights):
        return sum(w * op(x) for w, op in zip(weights, self.ops))


class Cell(nn.Module):
    def __init__(self, steps, multiplier, C_prev_prev, C_prev, C, reduction, reduction_prev):
        super().__init__()
        self.reduction = reduction
        self.steps = steps
        self.multiplier = multiplier

        if reduction_prev:
            self.preprocess0 = FactorizedReduce(C_prev_prev, C)
        else:
            self.preprocess0 = ReLUConvBN1D(C_prev_prev, C, 1, 1, 0)
        self.preprocess1 = ReLUConvBN1D(C_prev, C, 1, 1, 0)

        self.ops = nn.ModuleList()
        for i in range(self.steps):
            for j in range(2 + i):
                stride = 2 if reduction and j < 2 else 1
                op = MixedOp(C, stride)
                self.ops.append(op)

    def forward(self, s0, s1, weights):
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)
        states = [s0, s1]
        offset = 0
        for i in range(self.steps):
            s = sum(self.ops[offset + j](h, weights[offset + j]) for j, h in enumerate(states))
            offset += len(states)
            states.append(s)
        return torch.cat(states[-self.multiplier:], dim=1)


class Network(nn.Module):
    def __init__(self, C, num_classes, layers, criterion, steps=7, multiplier=4, stem_multiplier=3):
        super().__init__()
        self._C = C
        self._num_classes = num_classes
        self._layers = layers
        self._criterion = criterion
        self._steps = steps
        self._multiplier = multiplier

        C_curr = stem_multiplier * C
        self.stem = nn.Sequential(
            nn.Conv1d(1, C_curr, 3, padding=1, bias=False),
            nn.BatchNorm1d(C_curr)
        )

        C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
        self.cells = nn.ModuleList()
        reduction_prev = False
        for i in range(layers):
            if i in [layers // 3, 2 * layers // 3]:
                C_curr *= 2
                reduction = True
            else:
                reduction = False
            cell = Cell(steps, multiplier, C_prev_prev, C_prev, C_curr, reduction, reduction_prev)
            reduction_prev = reduction
            self.cells.append(cell)
            C_prev_prev, C_prev = C_prev, multiplier * C_curr

        self.global_pooling = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(C_prev, num_classes)

        self._initialize_alphas()

    def new(self):
        model_new = Network(self._C, self._num_classes, self._layers, self._criterion).to(next(self.parameters()).device)
        return model_new

    def forward(self, input):
        s0 = s1 = self.stem(input)
        for i, cell in enumerate(self.cells):
            weights = F.softmax(self.alphas_reduce if cell.reduction else self.alphas_normal, dim=-1)
            s0, s1 = s1, cell(s0, s1, weights)
        out = self.global_pooling(s1).squeeze(-1)
        logits = self.classifier(out)
        return logits

    def _initialize_alphas(self):
        k = sum(1 for i in range(self._steps) for n in range(2 + i))
        num_ops = len(OPS)

        self.alphas_normal = nn.Parameter(1e-3 * torch.randn(k, num_ops))
        self.alphas_reduce = nn.Parameter(1e-3 * torch.randn(k, num_ops))

        self._arch_parameters = [
            self.alphas_normal,
            self.alphas_reduce,
        ]

    def arch_parameters(self):
        return self._arch_parameters


# Required: ReLUConvBN1D and FactorizedReduce
class ReLUConvBN1D(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super().__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv1d(C_in, C_out, kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm1d(C_out, affine=affine)
        )

    def forward(self, x):
        return self.op(x)


class FactorizedReduce(nn.Module):
    def __init__(self, C_in, C_out, affine=True):
        super().__init__()
        assert C_out % 2 == 0
        self.relu = nn.ReLU(inplace=False)
        self.conv1 = nn.Conv1d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
        self.conv2 = nn.Conv1d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
        self.bn = nn.BatchNorm1d(C_out, affine=affine)

    def forward(self, x):
        x = self.relu(x)
        out = torch.cat([self.conv1(x), self.conv2(x[:, :, 1:])], dim=1)
        out = self.bn(out)
        return out