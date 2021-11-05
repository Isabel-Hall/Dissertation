import torch
from torch.utils.data import DataLoader
from data.preproc_dataset import FingerprintDataset

# Saves the data as concatenated maps of fingerprints and ground truth in pytprch format for faster loading

"""
dataset and dataloader
"""
dataset = FingerprintDataset(
    "/data/matlab_test_fingerprints/",
    "/data/numpy_test_data/"
)

dataloader = DataLoader(
    dataset,
    batch_size=1,
    num_workers=1,
    shuffle=True
)

for i, concat_map in enumerate(dataloader):
    concat_map = concat_map.squeeze()
    print("map shape", i, ":", concat_map.shape)
    torch.save(concat_map, f"/pytorch_test_data/{i}.pt")
    

# for i, (batch_of_data, ground_truth) in enumerate(dataloader):
#     print("batch", batch_of_data.shape, "ground_truth", ground_truth.shape)
#     output_tensor = torch.cat([
#         batch_of_data,
#         ground_truth
#     ], dim=1).squeeze()

#     torch.save(output_tensor, f"/data/pytorch_format/{i}.pt")
    
    #break
