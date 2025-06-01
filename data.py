import torch
from torch.utils.data import Dataset
import numpy as np
import h5py
import random
from tqdm import tqdm
from scipy.ndimage import binary_dilation

# custom PyTorch dataset for few-shot learning from H5 file
class FewShotH5Dataset(Dataset):
    def __init__(self, h5_path, institutions, n_way=3, k_shot=5, q_query=5, mask_dilate_iter=1, episodes=100):
        self.file = h5py.File(h5_path, 'r')
        self.n_way = n_way
        self.k_shot = k_shot
        self.q_query = q_query
        self.mask_dilate_iter = mask_dilate_iter
        self.episodes = episodes

        self.data = []
        self.labels = []

        # load data and labels from selected institutions
        for group in tqdm(institutions, desc=f"Indexing data from {institutions}"):
            self.data.append(self.file[f'{group}/data'][:])
            self.labels.append(self.file[f'{group}/labels'][:, 0])  # Shape (N,)

        self.data = np.concatenate(self.data, axis=0)
        self.labels = np.concatenate(self.labels, axis=0)

        # dilate mask channel
        for i in tqdm(range(len(self.data)), desc="Precomputing dilated masks"):
            self.data[i][3] = self.expand_mask(self.data[i][3], iterations=self.mask_dilate_iter)

        # define prototype classes (0: lymphocyte, 1: monocyte, 2: other)
        self.prototype_classes = [0, 1, 2]

        # index samples per class that have enough data for support + query
        self.class_indices = {
            label: np.where(self.labels == label)[0]
            for label in np.unique(self.labels)
            if len(np.where(self.labels == label)[0]) >= k_shot + q_query
        }

        # use all available classes for query phase
        self.query_classes = list(self.class_indices.keys())

    def __len__(self):
        # number of episodes to sample
        return self.episodes

    def __getitem__(self, idx):
        support_x, support_y = [], []
        query_x, query_y = [], []

        # sample support set from prototype classes
        for i, cls in enumerate(self.prototype_classes):
            idxs = np.random.choice(self.class_indices[cls], self.k_shot, replace=False)
            for idx in np.sort(idxs):
                img = self.data[idx].copy()
                support_x.append(img)
                support_y.append(i)

        # sample query set from all available classes
        for cls in self.query_classes:
            idxs = np.random.choice(self.class_indices[cls], self.q_query, replace=False)
            for idx in np.sort(idxs):
                img = self.data[idx].copy()
                query_x.append(img)

                query_y.append(self.prototype_classes.index(cls))

        # convert to torch tensors
        support_x = torch.tensor(np.array(support_x), dtype=torch.float32)
        query_x = torch.tensor(np.array(query_x), dtype=torch.float32)

        # normalize per channel
        mean = torch.tensor([0.5, 0.5, 0.5, 0.5]).view(1, 4, 1, 1)
        std = torch.tensor([0.25, 0.25, 0.25, 0.25]).view(1, 4, 1, 1)

        support_x = (support_x / 255.0 - mean) / std
        query_x = (query_x / 255.0 - mean) / std

        # Move channel dimension to [B, C, H, W]
        support_x = support_x.permute(0, 1, 2, 3)
        query_x = query_x.permute(0, 1, 2, 3)

        support_y = torch.tensor(support_y)
        query_y = torch.tensor(query_y)

        return support_x, support_y, query_x, query_y

    # expands a binary mask using morphological dilation.
    def expand_mask(self, mask, dilation_size=3, iterations=2):
        SE = np.ones((dilation_size, dilation_size), dtype=bool)
        expanded_mask = binary_dilation(mask, structure=SE, iterations=iterations)
        return expanded_mask.astype(np.uint8)
