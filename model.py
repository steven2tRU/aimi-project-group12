import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights

class ProtoNet(nn.Module):
    def __init__(self, in_channels=4, embedding_dim=64):
        super().__init__()
        # load pretrained ResNet18
        self.encoder = resnet18(weights=ResNet18_Weights)

        # replace first conv layer to accept 4 channels to include the binary mask
        self.encoder.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # freeze all layers
        for param in self.encoder.parameters():
            param.requires_grad = False
        # unfreeze last ResNet block and final FC for fine-tuning
        for param in self.encoder.layer4.parameters():
            param.requires_grad = True
        for param in self.encoder.fc.parameters():
            param.requires_grad = True

        # replace final FC layer to output the embedding dimension
        self.encoder.fc = nn.Linear(self.encoder.fc.in_features, embedding_dim)

    def forward(self, x, normalize=False):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        if normalize:
            x = F.normalize(x, p=2, dim=1)
        return x

# compute pairwise euclidean distance between embeddings
def euclidean_dist(a, b):
    n = a.size(0)
    m = b.size(0)
    return ((a.unsqueeze(1) - b.unsqueeze(0))**2).sum(2)

# compute pairwise cosine distance between embeddings
def cosine_dist(a, b):
    a = F.normalize(a, p=2, dim=1)
    b = F.normalize(b, p=2, dim=1)
    return 1 - torch.mm(a, b.T)

# forward pass for in batches
def batched_forward(model, x, batch_size=64, normalize=False):
    model.eval()
    embeddings = []
    with torch.no_grad():
        for i in range(0, len(x), batch_size):
            batch = x[i:i + batch_size].to(x.device)
            emb = model(batch, normalize=normalize)
            embeddings.append(emb)
    return torch.cat(embeddings, dim=0)