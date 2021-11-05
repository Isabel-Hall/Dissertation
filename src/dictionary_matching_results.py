import numpy as np
import torch
from torch import nn, use_deterministic_algorithms
from tqdm import tqdm
import time
from matplotlib import pyplot as plt
from pathlib import Path



"""
Args and constants
"""
SEQ_LENGTH = 600
USE_OFF_RES = False

"""
Directories
"""
fingerprintdir = "/media/DataSSD/datasets/MRI_Issie/test"
if USE_OFF_RES:
    dictfile = "/home/issie/data/dict_tables/big_dic.npy"
    lutfile = "/home/issie/data/dict_tables/big_lut.npy"
else:
    dictfile = "/home/issie/data/dict_tables/dic.npy"
    lutfile = "/home/issie/data/dict_tables/lut.npy"



"""
Dictionary pre-processing and Matching function
"""
# Loading, formatting and normalising dictionary and data
lut = torch.from_numpy(np.load(lutfile))
dic = np.load(dictfile)
# Get imaginary part of fingerprints
imag_dic = torch.from_numpy(dic.imag)
# Format dictionary to correct length
imag_dic = imag_dic.permute(1, 0)
short_dict = imag_dic[:, :600]

# Normalise dictionary entries
short_dict_norm = torch.norm(short_dict, dim=1)
short_dict = short_dict / short_dict_norm.unsqueeze(1)

def template_matching(dict, lut, map):
    # Takes a dictionary, look-up table and MRF map
    # Calcuates pixelwise dot product with dictionary entries
    # Returns map of T1/T2 values with the highest dot product for each pixel from LUT
    input = map.view(600, -1).permute(1, 0)
    reconstructed = []
    for pixel in tqdm(input, total=len(input)):
        dotprods = torch.mm(dict, pixel.unsqueeze(1))
        idx = torch.argmax(dotprods, axis=0)
        reconstructed.append(lut[idx])
    reconstructed = torch.cat(reconstructed)
    return reconstructed[:, :-1].permute(1, 0).view(-1, 232, 232)

"""
Losses
"""
mse_criterion = nn.MSELoss()
mae_criterion = nn.L1Loss()

"""
Testing loop
"""
times = []

for i, p in enumerate(Path(fingerprintdir).iterdir()):
    # Load file, split into MRF signal and ground truth
    MRF_file = torch.load(p)
    # Ground truth in ms to match dictionary entries
    ground_truth = MRF_file[1000: -1, 12:-12, 12:-12] * 1000
    MRF_map = MRF_file[:SEQ_LENGTH, 12:-12, 12:-12]
    # Normalise MRF signals
    MRF_norm = torch.norm(MRF_map, dim=0)
    MRF_map = MRF_map / MRF_norm.unsqueeze(0)

    # Template matching
    start = time.time()
    matched_map = template_matching(short_dict, lut, MRF_map)
    end = time.time()
    time_taken = end - start
    times.append(time_taken)

    # Losses, results converted from ms to s to match deep learning models
    matched_map = matched_map / 1000
    ground_truth = ground_truth / 1000

    loss_value_mse = mse_criterion(matched_map, ground_truth)
    t1_mse_loss = mse_criterion(matched_map[0, :, :], ground_truth[0, :, :])
    t2_mse_loss = mse_criterion(matched_map[1, :, :], ground_truth[1, :, :])

    loss_value_mae = mae_criterion(matched_map, ground_truth)
    t1_mae_loss = mae_criterion(matched_map[0, :, :], ground_truth[0, :, :])
    t2_mae_loss = mae_criterion(matched_map[1, :, :], ground_truth[1, :, :])

    t1_difference = torch.abs(matched_map[0,:,:] - ground_truth[0,:,:])
    t2_difference = torch.abs(matched_map[1,:,:] - ground_truth[1,:,:])

    torch.save(t1_difference, f"./results_torch/temp_match/{i}_t1.pt")
    torch.save(t2_difference, f"./results_torch/temp_match/{i}_t2.pt")

    t1_rel_diff = 100 - ((t1_difference + 1) / (ground_truth.squeeze()[0,:,:] + 1)) * 100
    t2_rel_diff = 100 - ((t2_difference + 1) / (ground_truth.squeeze()[1,:,:] + 1)) * 100

    fig, ax = plt.subplots(1, 2)
    plt.tight_layout()
    ax[0].matshow(t1_rel_diff, vmin=0, vmax=100)
    ax[0].set_title("Template Matching \nT1 error")
    im = ax[1].matshow(t2_rel_diff, vmin=0, vmax=100)
    ax[1].set_title("Template Matching \nT2 error")
    fig.colorbar(im, ax=ax.ravel().tolist(), shrink=0.5)
    ax[0].axis("off")
    ax[1].axis("off")
    plt.savefig(f"./results_images/baseline/{i}tm.png", bbox_inches="tight")
    plt.close(fig)

    fig, (ax1, ax2) = plt.subplots(1, 2)
    plt.tight_layout()
    im1 = ax1.matshow(t1_difference)
    ax1.set_title("Template Matching \nT1 error")
    im2 = ax2.matshow(t2_difference)
    ax2.set_title("Template Matching \nT1 error")
    fig.colorbar(im1, ax=ax1, shrink=0.5)
    fig.colorbar(im2, ax=ax2, shrink=0.5)
    ax1.axis("off")
    ax2.axis("off")
    plt.savefig(f"./results_images/abs{i}tm.png", bbox_inches="tight")
    plt.close(fig)

times_np = np.array(times)
print("times total", times_np.sum(), "mean=", times_np.mean()) 
print("dropping first element")
times_short_np = times_np[1:]
print("times total", times_short_np.sum(), "mean=", times_short_np.mean())