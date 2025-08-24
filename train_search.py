# sleep_nas/train_search.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from child_model import Network
from search_space import PRIMITIVES


def run_darts_search(train_loader, val_loader, num_classes):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = nn.CrossEntropyLoss()
    model = Network(C_in=1, C=32, num_classes=num_classes, layers=5, criterion=criterion).to(device)

    num_edges = sum(1 for i in range(5) for j in range(i + 1))
    k = len(PRIMITIVES)
    alpha = torch.randn(num_edges, k, requires_grad=True, device=device)

    optimizer_w = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.001)
    optimizer_a = optim.Adam([alpha], lr=0.0006, betas=(0.5, 0.999), weight_decay=0.001)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer_w, T_max=25)

    for epoch in range(25):
        model.train()
        total_loss, total_acc = 0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            x = x.unsqueeze(1)
            weights = F.softmax(alpha, dim=-1)
            loss, acc = model.loss(x, y, weights)

            optimizer_w.zero_grad()
            loss.backward()
            optimizer_w.step()

            total_loss += loss.item()
            total_acc += acc * x.size(0)

        avg_loss = total_loss / len(train_loader.dataset)
        avg_acc = total_acc / len(train_loader.dataset)
        print(f"[Epoch {epoch+1}/25] Train Loss: {avg_loss:.4f} | Train Acc: {avg_acc:.4f}")

        if epoch >= 10:
            model.eval()
            val_loss, val_acc, count = 0, 0, 0
            with torch.no_grad():
                for x_val, y_val in val_loader:
                    x_val, y_val = x_val.to(device), y_val.to(device)
                    x_val = x_val.unsqueeze(1)
                    weights = F.softmax(alpha, dim=-1)
                    loss_arch, acc_arch = model.loss(x_val, y_val, weights)

                    val_loss += loss_arch.item()
                    val_acc += acc_arch * x_val.size(0)
                    count += x_val.size(0)

            val_loss /= count
            val_acc /= count
            print(f"            Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

        scheduler.step()

    # Parse architecture
    weights = F.softmax(alpha, dim=-1).cpu().detach().numpy()
    normal_cell = []
    offset = 0
    for i in range(5):
        for j in range(i + 1):
            op_idx = weights[offset + j].argmax()
            normal_cell.append((PRIMITIVES[op_idx], j))
        offset += i + 1

    reduction_cell = normal_cell  # use same for now
    return normal_cell, reduction_cell
