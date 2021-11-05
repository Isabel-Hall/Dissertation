import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from data.ground_truth_dataset import GroundTruthDataset
from model.autoencoder import AutoEncoder
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from tqdm import tqdm

# dataset / dataloader
dataset = GroundTruthDataset(
    "/media/DataSSD/datasets/MRI_Issie/numpy/" 
)

dataloader = DataLoader(
    dataset,
    batch_size=8,
    shuffle=True,
    num_workers=8
)

# model + optimiser
autoencoder = AutoEncoder(2, 2).float().cuda()
optimiser = optim.Adam(autoencoder.parameters(), lr=0.0001)

# loss
#criterion = nn.MSELoss()
criterion = nn.SmoothL1Loss()


# training
losses = []

for epoch in tqdm(range(1000)):

    for i, label in enumerate(dataloader):

        label = label.cuda().float()

        # mean = torch.FloatTensor([0.5113866338173979, 0.06698109627281001]).view(1, 2, 1, 1).cuda()
        # std = torch.FloatTensor([0.7184752571113431, 0.12903516669471898]).view(1, 2, 1, 1).cuda()
        # label = (label - mean) / std
        # label = torch.tanh(label)

        optimiser.zero_grad()
        reconstructed_label, z = autoencoder(label)

        loss = criterion(reconstructed_label, label)
        loss.backward()

        optimiser.step()

        losses.append(loss.item())

    fig, ax = plt.subplots()
    ax.plot(losses)
    ax.set_title("Reconstruction Loss")
    plt.savefig("./artifacts/autoencoder_stuff/reconstruction_loss_3.png")
    plt.close(fig)

    # 2 by 2 figure
    fig, ax = plt.subplots(2, 2, figsize=(6.4, 8))
    plt.tight_layout()
    
    # plot the real T1 
    ax[0, 0].matshow(label[0, 0].detach().cpu().squeeze()) 
    ax[0, 0].set_title("Real T1")

    # plot the real T2
    ax[0, 1].matshow(label[0, 1].detach().cpu().squeeze()) 
    ax[0, 1].set_title("Real T2")

    # plot the reconstructed T1 
    ax[1, 0].matshow(reconstructed_label[0, 0].detach().cpu().squeeze()) 
    ax[1, 0].set_title("Reconstructed T1")

    # plot the reconstructed T2
    ax[1, 1].matshow(reconstructed_label[0, 1].detach().cpu().squeeze()) 
    ax[1, 1].set_title("Reconstructed T2")

    plt.savefig("./artifacts/autoencoder_stuff/image_3.png")
    plt.close(fig)

    if epoch % 111 == 0:
        torch.save(autoencoder.state_dict(), f"./artifacts/autoencoder_stuff/autoencoder_3.state")


        
