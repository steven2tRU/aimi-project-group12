import torch
import torch.nn.functional as F
from model import euclidean_dist, cosine_dist
from utils.device import get_device
from tqdm import tqdm
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
from config import config

device = get_device()

# validate model on few-shot batches
def validate_protonet(model, dataloader, plot_tsne=False, epoch = 0):
    model.eval()
    all_embs = []
    all_labels = []

    total_acc = 0
    with torch.no_grad():
        for support_x, support_y, query_x, query_y in dataloader:
            support_x = support_x.view(-1, *support_x.shape[2:]).to(device)
            query_x = query_x.view(-1, *query_x.shape[2:]).to(device)
            support_y = support_y.view(-1).to(device)
            query_y = query_y.view(-1).to(device)

            # encode support and query
            emb_support = model(support_x, normalize=True)
            emb_query = model(query_x, normalize=True)

            # compute prototypes
            prototypes = []
            proto_labels = []
            for cls in torch.unique(support_y):
                prototypes.append(emb_support[support_y == cls].mean(0))
                proto_labels.append(cls)
            prototypes = torch.stack(prototypes)
            proto_labels = torch.stack(proto_labels)

            # predict using closest prototype
            dists = torch.cdist(emb_query, prototypes)
            min_dists, min_indices = torch.min(dists, dim=1)
            preds = proto_labels[min_indices]

            acc = (preds == query_y).float().mean().item()
            total_acc += acc

            all_embs.append(emb_query.cpu())
            all_labels.append(query_y.cpu())

    model.train()

    all_embs = torch.cat(all_embs)
    all_labels = torch.cat(all_labels)

    # tsne visualization at epoch 10
    if plot_tsne and epoch == 9:
        emb_np = all_embs.numpy()
        labels_np = all_labels.numpy()

        tsne = TSNE(n_components=2, random_state=42)
        emb_2d = tsne.fit_transform(emb_np)

        plt.figure(figsize=(8, 6))
        for label in np.unique(labels_np):
            idx = labels_np == label
            plt.scatter(emb_2d[idx, 0], emb_2d[idx, 1], label=f"Class {label}", alpha=0.7)
        plt.legend()
        plt.title("t-SNE visualization of embeddings")
        plt.show()

    return total_acc / len(dataloader)

# trains ProtoNet on few-shot episodes and validates each epoch
def train_protonet(model, train_loader, val_loader, epochs=20, lr=1e-3):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_val_acc = 0.0
    temperature = 100  # 0.1 / 0.5 / 1.0

    for epoch in range(epochs):
        model.train()
        total_loss, total_acc = 0, 0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}", leave=False)
        for support_x, support_y, query_x, query_y in progress_bar:
            support_x = support_x.view(-1, *support_x.shape[2:]).to(device)
            query_x = query_x.view(-1, *query_x.shape[2:]).to(device)
            support_y = support_y.view(-1).to(device)
            query_y = query_y.view(-1).to(device)

            # encode inputs
            emb_support = model(support_x, normalize=True)
            emb_query = model(query_x, normalize=True)

            # compute prototypes
            prototypes = []
            proto_labels = []
            for cls in torch.unique(support_y):
                prototypes.append(emb_support[support_y == cls].mean(0))
                proto_labels.append(cls)
            prototypes = torch.stack(prototypes)
            proto_labels = torch.stack(proto_labels)

            # distance-based classification
            if config["distance_metric"] == "cosine":
                dists = cosine_dist(emb_query, prototypes)
            else:
                dists = euclidean_dist(emb_query, prototypes)
            log_p_y = F.log_softmax(-dists * temperature, dim=1)

            # predictions and masked loss
            min_dists, min_indices = torch.min(dists, dim=1)
            preds = proto_labels[min_indices]
            valid_mask = torch.ones_like(query_y, dtype=torch.bool)

            if valid_mask.sum() > 0:
                loss = F.nll_loss(log_p_y[valid_mask], query_y[valid_mask])
                acc = (preds[valid_mask] == query_y[valid_mask]).float().mean().item()
            else:
                # no valid query, skip loss update
                loss = torch.tensor(0.0, requires_grad=True)
                acc = 0.0

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_acc += acc
            progress_bar.set_postfix(loss=loss.item(), acc=acc)

        progress_bar.close()
        train_acc = total_acc / len(train_loader)
        val_acc = validate_protonet(model, val_loader, plot_tsne=True, epoch=epoch)

        tqdm.write(f"Epoch {epoch + 1} — Train Loss: {total_loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_protonet.pt')
            tqdm.write("Best model saved!")