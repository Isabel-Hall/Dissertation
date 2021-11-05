import torch
from torch.utils.data import Dataset
from pathlib import Path
#from scipy.io import loadmat
from matplotlib import pyplot as plt
from torchvision import transforms
import numpy as np
from torchvision.transforms.transforms import RandomCrop


class GroundTruthDataset(Dataset):
    def __init__(self, fingerprint_dir,  limit=-1):
        print("GroundTruthDataset.__init__")    
        # Used for training autoencoder to load ground truth maps only, no MRF time signals

        self.paths = []
        for p in Path(fingerprint_dir).iterdir():
            if p.is_file:
                self.paths.append(p)
            if limit > -1 and len(self.paths) >= limit:
                break
        # Add random rotation and pad as autoencoder needs 256 x 256 spatial dimensions
        self.tf1 = transforms.Compose([
            transforms.RandomRotation(45),
            transforms.Pad(13)
        ])


    def __len__(self):
        # Number of images in dataset (ground truth)
        return len(self.paths)

    def __getitem__(self, index):
        # Retrieve an image from the training data
        p = self.paths[index]
        full_map = torch.from_numpy(np.load(p)).float()
        full_map = full_map.permute(2, 0, 1)

        x = self.tf1(full_map) # adds random rotation
        
        ground_map = x[:-1, :, :]

        # Returns ground truth map used for training autoencoder
        return (ground_map.float())
