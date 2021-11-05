import torch
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from pathlib import Path
from torchvision import transforms
import numpy as np

# For loading balanced individual fingerprint signals to train RNN

def make_dataset_and_sampler(path_file, seq_length=1000):
    # Makes weighted random sampler to ensure even spread of fingerprint signals in training data
    # path_file contains paths to fingerprints divided into classes based on T1 and T2 values
    paths = []
    classes = []
    with open(path_file, "r") as f:
        for line in f.readlines():
            path, line_class = line.split(",")
            paths.append(path)
            classes.append(int(line_class))
    num_per_class = np.bincount(classes)

    weights = 1. / num_per_class
    samples_weight = torch.as_tensor([weights[c] for c in classes]).double()

    sampler = WeightedRandomSampler(samples_weight, len(samples_weight))
    # Create dataset using class below
    dataset = RNNDataset(paths, classes, seq_length)
    # Return newly created dataset with corresponding sampler
    return dataset, sampler

class RNNDataset(Dataset):
    def __init__(self, paths, classes, seq_length):
        # Creating dataset of individual fingerprint signals. Normalise and add noise to each fingerprint
        print("RNN dataset init")
        self.fingerprints = paths
        self.classes = classes
        self.seq_length = seq_length
        print("num files:", len(self.fingerprints))

        self.tf = transforms.Compose([
            transforms.Lambda(self.normalise),
            transforms.Lambda(self.noise),
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
        # Number of individual fingerprints in dataset
        return len(self.fingerprints)

    def __getitem__(self, index):
        # Retrive a fingerprint from the dataset. Has 3 ground truths concat on end (T1, T2, PD)
        p = self.fingerprints[index]
        c = self.classes[index]
        whole = torch.from_numpy(np.load(p))
        x = whole[:self.seq_length]
        gt = whole[1000:-1]
        x_tf = self.tf(x)
        # Return tuple of fingperprint signal and T1, T2 ground truth
        return (x_tf.float(), gt.float(), c)


