import torch
import h5py
from sklearn.metrics import classification_report
from model import ProtoNet
from utils.evaluate_utils import evaluate_on_full_slide
from utils.device import get_device
from config import config

device = get_device()

# initialize and load best model
model = ProtoNet(in_channels=4, embedding_dim=config["embedding_dim"]).to(device)
model.load_state_dict(torch.load('best_protonet.pt', map_location=device))
model.eval()

# load full validation split from the .h5 file
with h5py.File(config["file_path"], 'r') as f:
    val_data = f[config["val_institutions"][0]]['data'][:]
    val_labels = f[config["val_institutions"][0]]['labels'][:, 0]

# run full-slide evaluation
all_targets, all_preds = evaluate_on_full_slide(
    model,
    slide_data=val_data,
    slide_labels=val_labels,
    n_support=config["k_shot"],
    normalize=True
)

# Print evaluation metrics
print(classification_report(all_targets, all_preds, digits=4))
