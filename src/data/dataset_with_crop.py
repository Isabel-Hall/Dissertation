import torch
from torch.utils.data import Dataset
from pathlib import Path
#from scipy.io import loadmat
from matplotlib import pyplot as plt
from torchvision import transforms
import numpy as np
from torchvision.transforms.transforms import RandomCrop


class FingerprintDataset(Dataset):
    def __init__(self, fingerprint_dir,  random_crop, limit=-1, seq_length=1000):
        print("FingerprintDataset.__init__")
        # Makes list of files from directory of MRF data (fingerprint_dir) for dataloader   
        # This dataset takes a random crop (int) to reduce the spatial dimension of the MRF maps

        self.fingerprints = []
        self.seq_length = seq_length
        self.random_crop = random_crop
        for p in Path(fingerprint_dir).iterdir():
            if p.is_file:
                self.fingerprints.append(p)
            if limit > -1 and len(self.fingerprints) >= limit:
                break
        #self.fingerprints = self.fingerprints[:5]

        self.tf1 = transforms.Compose([
            transforms.RandomRotation(45),
            transforms.RandomCrop(self.random_crop)
        ])

        self.tf2 = transforms.Compose([
            transforms.Lambda(self.normalise),
            transforms.Lambda(self.noise)
        ])

    @staticmethod
    def normalise(x):
        # Preprocessing to normalise each fingerprint 
        mu = x.mean(dim=0)
        std = x.std(dim=0)
        y = (x - mu)/(std + 1e-7)
        return torch.tanh(y)

    @staticmethod
    def noise(x):
        # Adds gaussian noise to each fingerprint
        y = torch.randn_like(x)*0.1
        z = x + y
        return z.clamp(min=-1.0, max=1.0)

    def __len__(self):
        # Number of images in dataset (consists of fingerprint map with ground truth concatenated to end)
        return len(self.fingerprints)

    def __getitem__(self, index):
        # Retrieve an image from the training data
        p = self.fingerprints[index]
        full_map = torch.load(p)
        full_map = full_map[: , 12:-12, 12:-12]
        x = self.tf1(full_map) # adds random rotation and random crop
        # Split into the training data consiting of the 232 x 232 x SEQ_LENGTH MRF data and the ground truth
        training_map = x[:self.seq_length, :, :]
        ground_map = x[1000:-1, :, :]

        # Apply tf2 to training data to normalise values and add noise
        x_tf = self.tf2(training_map)

        # Returns tuple of transformed maps (training data, ground truth)
        return (x_tf.float(), ground_map.float())
