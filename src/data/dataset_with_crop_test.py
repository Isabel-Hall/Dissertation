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
            #transforms.RandomRotation(45),
            transforms.RandomCrop(self.random_crop)
        ])

        self.tf2 = transforms.Compose([
            transforms.Lambda(self.normalise),
            transforms.Lambda(self.noise)
        ])

    @staticmethod
    def normalise(x):
        # Preprocessing to normalise each fingerprint 
        #print("normalise x", type(x), x.shape)
        mu = x.mean(dim=0)
        std = x.std(dim=0)
        #print("mu:", mu.shape)
        #print("std:", std.shape)
        y = (x - mu)/(std + 1e-7)
        #print("y shape:", y.shape)
        #print("100,100 mean:", y[:,100,150].mean())
        #print("100,100 std:", y[:,100,150].std())
        return torch.tanh(y)

    @staticmethod
    def noise(x):
        # Adds gaussian noise to each fingerprint
        y = torch.randn_like(x)*0.1
        z = x + y
        return z.clamp(min=-1.0, max=1.0)

    def __len__(self):
        # Number of images in dataset (consists of fingerprint map with ground truth concatenated if using the .npy files)
        return len(self.fingerprints)

    def __getitem__(self, index):
        # Retrieve an image from the training data
        p = self.fingerprints[index]
        full_map = torch.load(p)
        full_map = full_map[: , 12:-12, 12:-12]
        x = self.tf1(full_map) # adds random rotation
        #print("shape after tf1:", x.shape)
        training_map = x[:self.seq_length, :, :] #original [:1000, : , : ], for mainfang_updated, use [:600, : , :]
        ground_map = x[1000:-1, :, :]
        #print("training shape:", training_map.shape)
        #print("ground shape", ground_map.shape)

        # Apply tf2 to training data to normalise values and add noise
        x_tf = self.tf2(training_map)

        # Returns tuple of transformed maps (training data, ground truth)
        return (x_tf.float(), ground_map.float())
