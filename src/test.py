"""
imports
"""
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from data.dataset import FingerprintDataset
from model.u_net import UNet 
from model.fang.unet_fang import Unet_3ds_rcab
from model.coarse_model import Coarse_net
from matplotlib import pyplot as plt
from torchvision import utils
from itertools import chain
import mlflow
from mlflow import log_metric, log_param, log_artifacts, log_image
from pathlib import Path



torch.manual_seed(0)
CUDA = torch.cuda.is_available()

"""
args and constants
"""
BATCH_SIZE = 1



"""
dataset and dataloader
"""
dataset = FingerprintDataset(
    "/pytorch_data/test"
)

dataloader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    num_workers=1,
    shuffle=False
)




"""
model
"""
#model = UNet()
model = Unet_3ds_rcab(64, 2).float()
model.load_state_dict(
    torch.load("/app/artifacts/0/2ed76a490e9c4502a144a7179f6d7fbb/States/model-99.state")
)
coarse_model = Coarse_net().float()
coarse_model.load_state_dict(
    torch.load("/app/artifacts/0/2ed76a490e9c4502a144a7179f6d7fbb/States/coarse_model-99.state")
)

if CUDA:
    model = model.to(torch.device("cuda:0"))
    coarse_model = coarse_model.to(torch.device("cuda:0"))



"""
optimiser and loss
"""
criterion = nn.MSELoss()


"""
test loop
"""

for (batch_of_data, ground_truth) in dataloader:

    if CUDA:
        batch_of_data = batch_of_data.to(torch.device("cuda:0"))
        ground_truth = ground_truth.to(torch.device("cuda:0"))
    
    #print(batch_of_data.shape)

    # utils.save_image(
    #     batch_of_data[:, :3, :, :],
    #     "./data.png",
    #     normalize=True
    # )

    #print(batch_of_data[0,100,100,:])
    z1 = coarse_model(batch_of_data.float())
    z = model(z1)
    #print("z shape:", z.shape, z.min().item(), z.max().item())
    #print("ground truth shiz: ", ground_truth.shape, ground_truth.min().item(), ground_truth.max().item())

    loss_value = criterion(z, ground_truth)
    print("loss", loss_value)

    t1_difference = torch.abs(z.squeeze()[0,:,:] - ground_truth.squeeze()[0,:,:])
    print("z shape", z.shape)
    print("differnce:", t1_difference.shape)
    # utils.save_image(
    #     t1_difference,
    #     Path("./saved_images/t1_pred_dif.png"),
    #     normalize=True
    # )
    plt.matshow(t1_difference.detach().numpy())
    plt.savefig("./tmp/t1dif.jpg")
    plt.clf()
    t2_difference = torch.abs(z.squeeze()[1,:,:] - ground_truth.squeeze()[1,:,:])
    print("z shape", z.shape)
    print("differnce:", t2_difference.shape)
    # utils.save_image(
    #     t2_difference,
    #     Path("./saved_images/t2_pred_dif.png"),
    #     normalize=True
    # )
    plt.matshow(t2_difference.detach().numpy())
    plt.savefig("./tmp/t2dif.jpg")
    plt.clf()
    break




        # print("z2 shape:", z2.shape)
        #break


        # z = z.detach().cpu()
        # ground_truth = ground_truth.detach().cpu()

        # utils.save_image(
        #     z[:, 0, :, :].unsqueeze(1),
        #     T1_PRED_DIR / f"{epoch}.png",
        #     normalize=True
        # )   
        # utils.save_image(
        #     z[:, 1, :, :].unsqueeze(1),
        #     T2_PRED_DIR / f"{epoch}.png",
        #     normalize=True
        # )   
        # utils.save_image(
        #     ground_truth[:, 0, :, :].unsqueeze(1),
        #     T1_GT_DIR / f"{epoch}.png",
        #     normalize=True
        # ) 
        # utils.save_image(
        #     ground_truth[:, 1, :, :].unsqueeze(1),
        #     T2_GT_DIR / f"{epoch}.png",
        #     normalize=True
        # )

        #torch.save(coarse_model.state_dict(), STATE_DIR / f"coarse_model-{epoch}.state")
        #torch.save(model.state_dict(), STATE_DIR / f"model-{epoch}.state")
        #log_artifacts(ARTIFACT_DIR)