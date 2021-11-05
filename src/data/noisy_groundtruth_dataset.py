import torch
from torch.utils.data import Dataset
from pathlib import Path
#from scipy.io import loadmat
from matplotlib import pyplot as plt
from torchvision import transforms
import numpy as np
from torchvision.transforms.transforms import RandomCrop


class GroundTruthDataset(Dataset):
    def __init__(self, fingerprint_dir,  limit=-1, seq_length=1000):
        print("GroundTruthDataset.__init__")    


        self.paths = []
        for p in Path(fingerprint_dir).iterdir():
            if p.is_file:
                self.paths.append(p)
            if limit > -1 and len(self.paths) >= limit:
                break
        #self.paths = self.paths[:5]

        # self.tf1 = transforms.Compose([
        #     transforms.RandomRotation(45),
        #     transforms.RandomCrop(self.random_crop)
        # ])
        self.tf1 = transforms.Compose([
            transforms.RandomRotation(45),
            transforms.Pad(13)
        ])
        self.tf2 = transforms.Lambda(self.noise)

    @staticmethod
    def noise(x):
        # Adds gaussian noise
        y = torch.randn_like(x)*0.3
        y[1] = y[1] * 0.135
        z = x + y
        return z

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
        noisy_map = self.tf2(ground_map)
        #print("training shape:", training_map.shape)
        #print("ground shape", ground_map.shape)


        
        return (ground_map.float(), noisy_map.float())
