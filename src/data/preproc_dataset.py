import torch
from torch.utils.data import Dataset
from pathlib import Path
from scipy.io import loadmat
from matplotlib import pyplot as plt
from torchvision import transforms
import numpy as np


class FingerprintDataset(Dataset):

    def __init__(self, fingerprint_dir, ground_truth_dir):
        print("preproc_FingerprintDataset.__init__")
        # Pre-processing to save MRF data together with corresponding ground truth values
        self.fingerprints = []
        # Lists all the MRF maps and ground truth maps from fingerprint_dir
        for p in Path(fingerprint_dir).iterdir():
            if p.is_file():
                filestem = p.stem                
                ground_truth_path = Path(ground_truth_dir + filestem + ".npy")
                if ground_truth_path.exists():
                    self.fingerprints.append((p, ground_truth_path))
             
        self.tf1 = transforms.Compose([
            transforms.ToTensor(),
            transforms.Pad((13,13))           
        ])
                

    def __len__(self):
        # Number of MRF files
        return len(self.fingerprints)

    def __getitem__(self, index):
        (p, g) = self.fingerprints[index]
        matlab_file = loadmat(p)
        # loads as a dictionary with 4 entries: ['__header__', '__version__', '__globals__', 'fingerprint_map']
        # header: matlab file and date/time created
        # version: ['1.0']
        # globals: []
        # fingerprint_map: tensor with fingerprint data

        # Concatenate the imaginary part of the MRF data and ground truth to apply transformations
        imaginary_map = matlab_file["fingerprint_map"].imag
        ground_truth_map = np.load(g)
        concat_maps = np.concatenate((imaginary_map, ground_truth_map), axis=2)

        # apply self.tf1 here. Becomes a 1003 x 256 x 256 tensor
        x = self.tf1(concat_maps)


        # Returns the training data and ground truth concatentated together, with appropriate padding and transformed to tensor
        return x

