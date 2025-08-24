import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

class SleepDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32).unsqueeze(1)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def load_sleepedf_npz(data_dir="./PSG"):
    all_data, all_labels = [], []
    label_map = {'W': 0, 'N1': 1, 'N2': 2, 'N3': 3, 'REM': 4}

    for fname in sorted(os.listdir(data_dir)):
        if fname.endswith(".npz"):
            path = os.path.join(data_dir, fname)
            npz = np.load(path)
            X, y = npz['x'], npz['y']

            # Keep only known classes
            mask = np.isin(y, list(label_map.values()))
            X = X[mask]
            y = y[mask]

            print(f"[DEBUG] Loaded {path} → {len(y)} samples")
            all_data.append(X)
            all_labels.append(y)

    if not all_data:
        raise RuntimeError("No valid .npz files found!")

    X = np.concatenate(all_data, axis=0)
    y = np.concatenate(all_labels, axis=0)
    return X, y, len(label_map)

def get_dataloaders_simple(batch_size=32):
    X, y, num_classes = load_sleepedf_npz()

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.5, stratify=y, random_state=47)

    train_set = SleepDataset(X_train, y_train)
    val_set = SleepDataset(X_val, y_val)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, num_classes
