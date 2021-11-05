import torch
from torch.utils.data import Dataset
from pathlib import Path
#from scipy.io import loadmat
from matplotlib import pyplot as plt
from torchvision import transforms
import numpy as np
from torchvision.transforms.transforms import RandomCrop


class FingerprintDataset(Dataset):
    def __init__(self, fingerprint_dir,  limit=-1, seq_length=1000):
        print("FingerprintDataset.__init__")    
        # Creates list of files to load from fingerprint_dir 
        # Default sequence length is whole 1000 long fingerprint signals, can be reduced to shorter length signals

        self.fingerprints = []
        self.seq_length = seq_length
        for p in Path(fingerprint_dir).iterdir():
            if p.is_file:
                self.fingerprints.append(p)
            if limit > -1 and len(self.fingerprints) >= limit:
                break

        self.tf1 = transforms.RandomRotation(45) # Adds random rotation
        # Normalises fingerprint signals and adds noise
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
        # Number of images in dataset (consists of fingerprint map with ground truth concatenated at end)
        return len(self.fingerprints)

    def __getitem__(self, index):
        # Retrieve an image from the training data
        p = self.fingerprints[index]
        full_map = torch.load(p)
        x = self.tf1(full_map) # adds random rotation
        # Separate into training data and ground truth. Spatial dimensions 232 x 232 x SEQ_LENGTH
        training_map = x[:self.seq_length, 12:-12, 12:-12] 
        ground_map = x[1000:-1, 12:-12, 12:-12]
        # Apply tf2 to training data to normalise values and add noise
        x_tf = self.tf2(training_map)

        # Returns tuple of transformed maps (training data, ground truth)
        return (x_tf.float(), ground_map.float())
