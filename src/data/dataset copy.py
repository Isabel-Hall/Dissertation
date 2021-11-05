import torch
from torch.utils.data import Dataset
from pathlib import Path
from scipy.io import loadmat
from matplotlib import pyplot as plt
from torchvision import transforms
import numpy as np


class FingerprintDataset(Dataset):

    def __init__(self, fingerprint_dir, ground_truth_dir):
        print("FingerprintDataset.__init__")

        self.fingerprints = []
        # using fingerprint_dir, find all the fingerprints
        for p in Path(fingerprint_dir).iterdir():
            if p.is_file():
                filestem = p.stem                
                #print("p:", p)
                #print(filestem)
                ground_truth_path = Path(ground_truth_dir + filestem + ".npy")
                if ground_truth_path.exists():
                    #print(ground_truth_path)
                    self.fingerprints.append((p, ground_truth_path))
        #print("length:", len(self.fingerprints))   

        # Load the maps into memory
        self.loaded_maps = []
        for (tp, gp) in self.fingerprints:
            mat_file = loadmat(tp)
            # loads as a dictionary with 4 entries: ['__header__', '__version__', '__globals__', 'fingerprint_map']
            # header: matlab file and date/time created
            # version: ['1.0']
            # globals: []
            # fingerprint_map: tensor with fingerprint data
            training_map = np.array(mat_file["fingerprint_map"].imag)
            del mat_file
            truth_map = np.load(gp)
            print("init training shape:", training_map.shape)
            print("init truth shape:", truth_map.shape)
            self.loaded_maps.append((training_map, truth_map))

        self.tf1 = transforms.Compose([
            transforms.ToTensor(),
            transforms.Pad((13,13)),
            transforms.RandomRotation(45)            
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
        return y

    @staticmethod
    def noise(x):
        # Adds gaussian noise to each fingerprint
        y = torch.randn_like(x)*0.1
        z = x + y
        return z
                

    def __len__(self):
        # how many datapoints we have. number of slides?
        return len(self.fingerprints)

    def __getitem__(self, index):
        #print("__getitem__", index)
        #(p, g) = self.fingerprints[index]
        #matlab_file = loadmat(p)

        # print(matlab_file["fingerprint_map"].shape)
        # print(matlab_file["fingerprint_map"].min())
        # print(matlab_file["fingerprint_map"].max())
        # print(type(matlab_file["fingerprint_map"]))
        # test = matlab_file["fingerprint_map"][100, 100, :]
        # print("test shape:", test.shape)
        # plt.plot(test.imag)
        # plt.savefig("testplot.png")


        
        #imaginary_map = matlab_file["fingerprint_map"].imag
        #print("imaginary map shape:", imaginary_map.shape)
        #ground_truth_map = np.load(g)
        #print("ground truth shape:", ground_truth_map.shape)

        # Concatenate the input data and ground truth to apply transformations
        (training_map, truth_map) = self.loaded_maps[index]
        concat_maps = np.concatenate((training_map, truth_map), axis=2)
        #print("concat shape", concat_maps.shape)
        # test2 = imaginary_map[100, 100, :]
        # print("fingerprint shape:", test2.shape)
        # plt.plot(test2)
        # plt.savefig("test2plot.png")
        # plt.close()

        # apply self.tf1 here. Becomes a 1003x256x256 tensor
        x = self.tf1(concat_maps)
        print("x shape:", x.shape)
        # Divide rotated tensor into training data and ground truth:
        training = x[:1000, :, :]
        ground = x[1000:, :, :]

        #print("transformed training shape:", training.shape)
        #print("transformed ground shape:", ground.shape)

        # Finish transformations for training data (normalise and noise):
        y = self.tf2(training)
        #print(y.shape)

        #print("max val:", x.max())
        #print("min val:", x.min())

        # test3 = x[:, 100, 100]
        # print("fingerprint processed shape:", test3.shape)
        # plt.plot(test3)
        # plt.savefig("test3plot.png")
        # plt.close()

        #x = torch.from_numpy(imaginary_map).float()
        #x = x.permute(2, 0, 1).contiguous()

        return (y, ground)

