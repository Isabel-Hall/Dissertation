import numpy as np
import torch
from torch import nn
from tqdm import tqdm
import time
from matplotlib import pyplot as plt


#If using dict with no off res
dic = np.load("/home/issie/data/dict_tables/dic.npy")
lut = torch.from_numpy(np.load("/home/issie/data/dict_tables/lut.npy"))
imag_dic = torch.from_numpy(dic.imag)
imag_dic = imag_dic.permute(1, 0)
short_dic = imag_dic[:, :600]
print("short dict:", short_dic.shape)


short_dic_norm = torch.norm(short_dic, dim=1)
print("short dict norm shape:", short_dic_norm.shape)
short_dic = short_dic / short_dic_norm.unsqueeze(1)
print("600 norm:", torch.norm(short_dic, dim=1))

#print("dict shape", dic.shape) #(1000, 314160), with small dict: (1000, 8296)

print(lut.shape) #(314160, 3), (8296, 3)


# Larger dict with off res
large_dic = np.load("/home/issie/data/dict_tables/big_dic.npy")
large_lut = torch.from_numpy(np.load("/home/issie/data/dict_tables/big_lut.npy"))
imag_large = torch.from_numpy(large_dic.imag)
imag_large = imag_large.permute(1, 0)
large_dic_shorter = imag_large[:, :600]
print("larger dict short seq", large_dic_shorter.shape)

large_dic_norm = torch.norm(large_dic_shorter, dim=1)
large_dic_shorter = large_dic_shorter / large_dic_norm.unsqueeze(1)
print("larger norm:", torch.norm(large_dic_shorter, dim=1))



fst_d = dic[:, 0]
fst_l = lut[0]

#print("avg:", np.average(fst_d))
#print("std:", np.std(fst_d))
# Finding entries with 0 off res
#print(fst_l)
#print(lut[0, 2])
#zero_off = np.argwhere(lut[:, 2] == 0)
#print("num entries with off res zero:", len(zero_off))
#idx = zero_off[0]
#print(idx)
#print(dic[0, idx])
#getting imaginary part
#print("imag")
#data points in first dic entry [point in fingerprint, entry]
#print(dic[:5, 0].imag)
#print(imag_dic[:5, 0])

test_slice = torch.load("/media/DataSSD/datasets/MRI_Issie/train/0.pt").float()

ground_truth = test_slice[1000: -1, 12:-12, 12:-12]
ground_truth = ground_truth * 1000
test_slice = test_slice[:600, 12:-12, 12:-12]
test_slice_norm = torch.norm(test_slice, dim=0)
print("norm shape", test_slice_norm.shape)
test_slice = test_slice / test_slice_norm.unsqueeze(0)
#test_slice = test_slice[:600, :, :]
print("test slice")
print(test_slice.shape)
print(ground_truth.shape)


test_slice_flat = test_slice.view(600, -1)
print("test_slice_flat", test_slice_flat.shape)
dotprods = torch.mm(short_dic, test_slice_flat)
print("dotprods", dotprods.shape)
idx = torch.argmax(dotprods, axis=0)
print("idx", idx.shape)
#sample = idx[23300: 23306]
#print(sample)
#print(lut[sample])
reconstructed = lut[idx]
reconstructed = reconstructed[:, :-1].permute(1, 0).view(-1, 232, 232)
print("reconstructed shape:", reconstructed.shape)


def template_matching(dict, lut, map):
    input = map.view(600, -1).permute(1, 0)
    reconstructed = []
    #print("fx reconstructed size", reconstructed.shape)
    for pixel in tqdm(input, total=len(input)):
        #print("pixel size", pixel.shape) #600
        dotprods = torch.mm(dict, pixel.unsqueeze(1))
        #print("fx dotprod shape:", dotprods.shape)
        idx = torch.argmax(dotprods, axis=0)
        #print(idx)
        reconstructed.append(lut[idx])
        #print(pixel)
    reconstructed = torch.cat(reconstructed)
    return reconstructed[:, :-1].permute(1, 0).view(-1, 232, 232)


#Timings and loss to check validity between dicts
short_start = time.time()
short_matches = template_matching(short_dic, lut, test_slice)
short_end = time.time()
print("Time to match with short dict:", short_end - short_start)
#matches = matches[:, :-1].permute(1, 0).view(-1, 232, 232)

larger_start = time.time()
matches = template_matching(large_dic_shorter, large_lut, test_slice)
larger_end = time.time()
print("time to match with larger dict:", larger_end - larger_start)

print("matches (larger):", matches.shape)
print("reconstructed (short):", reconstructed.shape)

results_mse = nn.functional.mse_loss(matches / 1000, reconstructed / 1000)
print("mse between larger and shorter:", results_mse) #  tensor(0.3809, dtype=torch.float64)

gt_mse_mat = nn.functional.mse_loss(matches / 1000, ground_truth / 1000)
print("gt mse larger", gt_mse_mat) # tensor(0.0029, dtype=torch.float64)

gt_mse_rec = nn.functional.mse_loss(reconstructed / 1000, ground_truth / 1000)
print("gt mse shorter", gt_mse_rec) # tensor(0.3783, dtype=torch.float64)

# Saving results
"""mse_criterion = nn.MSELoss()
mae_criterion = nn.L1Loss()




fig, ax = plt.subplots(1, 2)
plt.tight_layout()
ax[0].matshow(t1_rel_diff, vmin=0, vmax=100)
ax[0].set_title("Temporal Convolution \nT1 error")
im = ax[1].matshow(t2_rel_diff, vmin=0, vmax=100)
ax[1].set_title("Temporal Convolution \nT2 error")
fig.colorbar(im, ax=ax.ravel().tolist(), shrink=0.5)
ax[0].axis("off")
ax[1].axis("off")
plt.savefig(f"./results_images/baseline/{i}conv1d.png", bbox_inches="tight")
plt.close(fig)

fig, (ax1, ax2) = plt.subplots(1, 2)
plt.tight_layout()
im1 = ax1.matshow(t1_difference)
ax1.set_title("Temporal Convolution \nT1 error")
im2 = ax2.matshow(t2_difference)
ax2.set_title("Temporal Convolution \nT1 error")
fig.colorbar(im1, ax=ax1, shrink=0.5)
fig.colorbar(im2, ax=ax2, shrink=0.5)
ax1.axis("off")
ax2.axis("off")
plt.savefig(f"./results_images/abs{i}conv1d.png", bbox_inches="tight")
plt.close(fig)"""


#print("random point")
#print(test_slice[:5, 100, 100])

#dot_prod = torch.dot(imag_dic[0, :], test_slice[:, 100, 100])
#print(dot_prod)

def single_template_matching(dict, fingerprint):
    # Takes a dictionary and an individual fingerprint to calculate the max dot product
    # Returns the index with the highes dot product to use in the LUT
    # Fingerprint must be the same length as the dictionary entries
    idx = -1
    max_dot_prod = 0
    for i, entry in enumerate(dict):
        #fingerprint = fingerprint / torch.norm(fingerprint)
        #print("fp", fingerprint.shape, "entry", entry.shape)
        #print("fp norm", torch.norm(fingerprint), "entry norm", torch.norm(entry))
        #input("")
        dot_prod = torch.dot(fingerprint, entry)
        if dot_prod > max_dot_prod:
            max_dot_prod = dot_prod
            idx = i
    return idx


# for j in range(100, 106):
#     id, max_dot_prod = template_matching(short_dic, test_slice[:, 100, j])
#     print("idx", id)
#     #print("max_dot_prod", max_dot_prod)

#     print("ground truth", ground_truth[:, 100, j])
#     print("LUT:", lut[id])
#print(lut[2091])
#print(torch.dot(test_slice[:, 100, 100], short_dic[2091]))
