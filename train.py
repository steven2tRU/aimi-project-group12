from data import FewShotH5Dataset
from model import ProtoNet
from utils.train_utils import train_protonet
from utils.device import get_device
import torch
from config import config

# load training dataset
train_dataset = FewShotH5Dataset(
    config["file_path"],
    institutions=config["train_institutions"],
    n_way=config["n_way"],
    k_shot=config["k_shot"],
    q_query=config["q_query"],
    mask_dilate_iter=config["mask_dilate_iter"],
    episodes=config["episodes"]
)

# load validation dataset
val_dataset = FewShotH5Dataset(
    config["file_path"],
    institutions=config["val_institutions"],
    n_way=config["n_way"],
    k_shot=config["k_shot"],
    q_query=config["q_query"],
    mask_dilate_iter=config["mask_dilate_iter"],
    episodes=config["episodes"]
)

# wrap datasets in dataloaders
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False)

# initialize ProtoNet model
model = ProtoNet(in_channels=4, embedding_dim=config["embedding_dim"]).to(get_device())

# train model
train_protonet(
    model,
    train_loader,
    val_loader,
    epochs=config["epochs"],
    lr=config["learning_rate"],
)