# sleep_nas/train_final.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from search_space import OPS

class FinalCell(nn.Module):
    def __init__(self, C, cell_desc):
        super().__init__()
        self.ops = nn.ModuleList()
        for name, idx in cell_desc:
            op = OPS[name](C, stride=1, affine=True)
            self.ops.append((op, idx))

    def forward(self, x, prev_outputs):
        states = prev_outputs + [x]
        outputs = [op(states[idx]) for op, idx in self.ops]
        return sum(outputs)

class FinalModel(nn.Module):
    def __init__(self, normal_cell, reduction_cell, C_in, C, num_classes, depth):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(C_in, C, 3, padding=1, bias=False),
            nn.BatchNorm1d(C)
        )
        self.cells = nn.ModuleList()
        for i in range(depth):
            cell = FinalCell(C, normal_cell if i % 3 != 2 else reduction_cell)
            self.cells.append(cell)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(C, num_classes)

    def forward(self, x):
        x = self.stem(x)
        prev_outputs = []
        for cell in self.cells:
            x = cell(x, prev_outputs)
            prev_outputs.append(x)
        x = self.pool(x).squeeze(-1)
        return self.classifier(x)

def build_final_model(normal_cell, reduction_cell, num_classes, depth):
    return FinalModel(normal_cell, reduction_cell, C_in=1, C=32, num_classes=num_classes, depth=depth)

def train_final_model(model, train_loader, val_loader, test_loader, depth):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.001)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

    for epoch in range(50):
        model.train()
        total_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            x = x.unsqueeze(1)
            logits = model(x)
            loss = criterion(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()
        print(f"[Final Train] Depth={depth} | Epoch={epoch+1}/50 | Loss={total_loss/len(train_loader):.4f}")
