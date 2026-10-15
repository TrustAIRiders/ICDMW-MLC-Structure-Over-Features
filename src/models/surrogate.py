import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.loader import DataLoader
from sklearn.metrics import f1_score
from src.train.sampler import BalancedBatchSampler


class SurrogateGCN(nn.Module):

    def __init__(self, node_feat_dim: int, hidden_dim: int = 128, num_classes: int = 2,
                 dropout: float = 0.3):
        super().__init__()
        self.conv1 = GCNConv(node_feat_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        x = F.relu(self.norm1(self.conv1(x, edge_index)))
        x = self.dropout(x)
        x = F.relu(self.norm2(self.conv2(x, edge_index)))

        x = global_mean_pool(x, batch)
        return self.classifier(x)

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def train_surrogate(
    surrogate: SurrogateGCN,
    train_dataset,
    val_dataset,
    epochs: int = 50,
    lr: float = 0.001,
    batch_size: int = 256,
    device: str = "cuda",
    early_stop_patience: int = 10,
    verbose: bool = True,
) -> SurrogateGCN:

    surrogate = surrogate.to(device)

    labels = train_dataset.get_labels()
    sampler = BalancedBatchSampler(labels, batch_size=batch_size)
    train_loader = DataLoader(
        train_dataset, batch_sampler=sampler, num_workers=0
    )
    val_loader = DataLoader(val_dataset, batch_size=512, shuffle=False, num_workers=0)

    optimizer = torch.optim.Adam(surrogate.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    best_val_f1 = -1.0
    best_state = None
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        surrogate.train()
        total_loss = 0.0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            logits = surrogate(batch)
            loss = criterion(logits, batch.y.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        surrogate.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                logits = surrogate(batch)
                preds = logits.argmax(dim=-1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(batch.y.view(-1).cpu().numpy())

        val_f1 = float(f1_score(all_labels, all_preds, average="macro", zero_division=0))

        if verbose:
            avg_loss = total_loss / len(train_loader)
            print(f"  Epoch {epoch:3d}/{epochs}  loss={avg_loss:.4f}  val_F1={val_f1:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = {k: v.cpu().clone() for k, v in surrogate.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= early_stop_patience:
                if verbose:
                    print(f"  Early stop at epoch {epoch} (patience={early_stop_patience})")
                break

    if best_state is not None:
        surrogate.load_state_dict(best_state)
    surrogate.to(device)
    surrogate.eval()

    if verbose:
        print(f"  Best val F1-macro: {best_val_f1:.4f}")

    return surrogate