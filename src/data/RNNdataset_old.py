import torch
from torch.utils.data import Dataset
from pathlib import Path
from matplotlib import pyplot as plt
from torchvision import transforms
import numpy as np
import random

class RNNDataset(Dataset):
    def __init__(self, fingerprint_dir):
        print("RNNDataset.__init__")

        self.slices = []
        self.fingerprints = []
        for p in Path(fingerprint_dir).iterdir():
            if p.is_file:
                self.slices.append(p)
        self.idxs = torch.randperm(len(self.slices))
        self.counter = 0

        self.tf = transforms.Compose([
            transforms.Lambda(self.normalise),
            transforms.Lambda(self.noise)
        ])

    @staticmethod
    def normalise(x):
        # Preprocessing to normalise each fingerprint 
        mu = x.mean(dim=0)
        std = x.std(dim=0)
        y = (x - mu)/(std + 1e-7)
        return y

    @staticmethod
    def noise(x):
        # Adds gaussian noise to each fingerprint
        y = torch.randn_like(x)*0.1
        z = x + y
        return z

    def __len__(self):
        # Number of images in dataset (consists of fingerprint map with ground truth concatenated if using the .pt files)
        return len(self.slices) * 65536

    def __getitem__(self, index):
        # Retrieve an image from the training data
        # Load a slice
        if index % 65536 == 6500:
            p = self.slices[self.idxs[counter]]
            counter += 1
            full_map = torch.load(p)
            training_map = full_map[:1000, :, :]
            ground_map = full_map[1000:-1, :, :]
            x = self.tf(training_map).view(1000, -1).contiguous().permute(1, 0)
            print("x shape", x.shape)
            to_add = []
            to_add.extend(x.unbind())
            random.shuffle(to_add)
            self.fingerprints.extend(to_add)
        # Retreive the time series images (fingerprints) from the loaded images
        tmp = self.fingerprints[index]
        self.fingerprints[index] = None
        return tmp
