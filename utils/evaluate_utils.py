import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from model import batched_forward

# this function evaluates a trained model on full slide data using a few-shot support-query split.
# computes accuracy, precision, recall, F1, and plots confusion matrix.
def evaluate_on_full_slide(model, slide_data, slide_labels, n_support=5, normalize=True):
    model.eval()
    all_preds = []
    all_targets = []

    # prepare data
    slide_data = torch.tensor(slide_data, dtype=torch.float32) / 255.0
    slide_data = slide_data.permute(0, 1, 2, 3)  # [B, C, H, W]
    slide_labels = np.array(slide_labels)

    prototype_classes = [0, 1, 2]  # model learns prototypes for class 0, 1, 2
    unique_labels = np.unique(slide_labels)
    unique_labels = np.array([cls for cls in unique_labels if cls in prototype_classes])

    # split into support and query
    support_idxs = []
    query_idxs = []
    for label in unique_labels:
        idxs = np.where(slide_labels == label)[0]
        if len(idxs) < n_support:
            print(f"Warning: Not enough support samples for class {label}. Found {len(idxs)} samples.")
            selected = idxs
        else:
            selected = np.random.choice(idxs, n_support, replace=False)
        support_idxs.extend(selected)
        query_idxs.extend([i for i in idxs if i not in selected])

    support_x = slide_data[support_idxs].to(next(model.parameters()).device)
    support_y = slide_labels[support_idxs]
    query_x = slide_data[query_idxs].to(next(model.parameters()).device)
    query_y = slide_labels[query_idxs]

    with torch.no_grad():
        emb_support = batched_forward(model, support_x, normalize=normalize)
        emb_query = batched_forward(model, query_x, batch_size=64, normalize=normalize)

        # compute prototypes
        prototypes = []
        proto_labels = []
        for label in unique_labels:
            indices = np.where(support_y == label)[0]
            proto = emb_support[indices].mean(0)
            prototypes.append(proto)
            proto_labels.append(label)

        prototypes = torch.stack(prototypes)
        proto_labels = torch.tensor(proto_labels).to(emb_query.device)

        # match queries to closest prototype
        dists = torch.cdist(emb_query, prototypes)
        min_dists, min_indices = torch.min(dists, dim=1)
        preds = proto_labels[min_indices]

        all_preds = preds.cpu().numpy()
        all_targets = query_y

    # evaluation
    acc = accuracy_score(all_targets, all_preds)
    precision = precision_score(all_targets, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_targets, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_targets, all_preds, average='weighted', zero_division=0)
    conf_mat = confusion_matrix(all_targets, all_preds)

    print(f"Final Accuracy:  {acc:.4f}")
    print(f"Precision:       {precision:.4f}")
    print(f"Recall:          {recall:.4f}")
    print(f"F1 Score:        {f1:.4f}")
    print("\n" + classification_report(all_targets, all_preds, digits=4))

    # plot confusion matrix
    plt.figure(figsize=(6, 5))
    sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues')
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.show()