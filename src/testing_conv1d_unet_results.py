from typing import Sequence
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from data.dataset_test import FingerprintDataset
from model.fang.unet_fang import Unet_3ds_rcab
from model.pretrained_rnn import PretrainedRNN
from model.coarse_model_conv1d import CoarseModel
from model.hoppeRNN import Hoppernn
from matplotlib import pyplot as plt
from torchvision import utils
from itertools import chain
import mlflow
from mlflow import log_metric, log_param, log_artifacts, log_image
from pathlib import Path
from tqdm import tqdm
import time
import numpy as np

try:
    experiment_id = mlflow.create_experiment("testing_conv1d_baseline_unet")
    experiment = mlflow.get_experiment(experiment_id)
except:
    experiment = mlflow.get_experiment_by_name("testing_conv1d_baseline_unet")
    experiment_id = experiment.experiment_id


torch.manual_seed(0)
CUDA = torch.cuda.is_available()

"""
args and constants
"""
BATCH_SIZE = 1
SEQ_LENGTH = 600
#CROP = 96

"""
dataset and dataloader
"""
dataset = FingerprintDataset(
    "/media/DataSSD/datasets/MRI_Issie/test/",
    seq_length=SEQ_LENGTH
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
model = Unet_3ds_rcab(75, 2).float()
model.load_state_dict(
    torch.load("/home/issie/code/py/MRI/src/artifacts/11/a16558c05f7546329b34dcf612096bcf/States/model-99.state")
)
model.eval()
coarse_model = CoarseModel(input_channels=SEQ_LENGTH).float()
coarse_model.load_state_dict(
    torch.load("/home/issie/code/py/MRI/src/artifacts/11/a16558c05f7546329b34dcf612096bcf/States/coarse_model-99.state")
)
coarse_model.eval()

if CUDA:
    model = model.to(torch.device("cuda:0"))
    coarse_model = coarse_model.to(torch.device("cuda:0"))


"""
loss
"""
mse_criterion = nn.MSELoss()
mae_criterion = nn.L1Loss()

"""
test loop
"""
with mlflow.start_run(experiment_id=experiment_id) as run:
    run_id = run.info.run_id
    # make required directories
    ARTIFACT_DIR = f"./artifacts/{experiment_id}/{run_id}/"
    T1_GT_DIR = Path(ARTIFACT_DIR) / "T1_GT"
    T1_PRED_DIR = Path(ARTIFACT_DIR) / "T1_PRED"
    T2_GT_DIR = Path(ARTIFACT_DIR) / "T2_GT"
    T2_PRED_DIR = Path(ARTIFACT_DIR) / "T2_PRED"
    T1_DIFF = Path(ARTIFACT_DIR) / "T1_DIFF"
    T2_DIFF = Path(ARTIFACT_DIR) / "T2_DIFF"

    paths = [T1_GT_DIR, T1_PRED_DIR, T2_GT_DIR, T2_PRED_DIR, T1_DIFF, T2_DIFF]
    for p in paths:
        p.mkdir(exist_ok=True, parents=True)


    log_param("batch_size", BATCH_SIZE)
    log_param("sequence_length", SEQ_LENGTH)

    times = []
    for i, (batch_of_data, ground_truth) in tqdm(enumerate(dataloader), total=len(dataloader)):
        start = time.time()

        if CUDA:
            batch_of_data = batch_of_data.to(torch.device("cuda:0"))
            ground_truth = ground_truth.to(torch.device("cuda:0"))


        with torch.no_grad():
            #print(batch_of_data[0,100,100,:])
            z1 = coarse_model(batch_of_data.float())
            #print("z1 post prelim", z1.shape)
            z = model(z1)
            end = time.time()
            times.append(end - start)
        #print("z shape:", z.shape, z.min().item(), z.max().item())
        #print("ground truth shiz: ", ground_truth.shape, ground_truth.min().item(), ground_truth.max().item())

        loss_value_mse = mse_criterion(z, ground_truth)
        t1_loss_value_mse = mse_criterion(z.squeeze()[0, : , : ], ground_truth.squeeze()[0, : , : ])
        t2_loss_value_mse = mse_criterion(z.squeeze()[1, : , : ], ground_truth.squeeze()[1, : , : ])
        #print("loss", loss_value)
        loss_value_mae = mae_criterion(z, ground_truth)
        t1_loss_value_mae = mae_criterion(z.squeeze()[0, : , : ], ground_truth.squeeze()[0, : , : ])
        t2_loss_value_mae = mae_criterion(z.squeeze()[1, : , : ], ground_truth.squeeze()[1, : , : ])

        log_metric("MSE_loss", loss_value_mse.item())
        log_metric("T1_MSE_loss", t1_loss_value_mse.item())
        log_metric("T2_MSE_loss", t2_loss_value_mse.item())
        log_metric("MAE_loss", loss_value_mae.item())
        log_metric("T1_MAE_loss", t1_loss_value_mae.item())
        log_metric("T2_MAE_loss", t2_loss_value_mae.item())

        z = z.detach().cpu()
        ground_truth = ground_truth.detach().cpu()

        t1_difference = torch.abs(z.squeeze()[0,:,:] - ground_truth.squeeze()[0,:,:])
        t2_difference = torch.abs(z.squeeze()[1,:,:] - ground_truth.squeeze()[1,:,:])

        torch.save(t1_difference, f"./results_torch/conv1d/{i}_t1.pt")
        torch.save(t2_difference, f"./results_torch/conv1d/{i}_t2.pt")

        """t1_rel_diff = 100 - ((t1_difference + 1) / (ground_truth.squeeze()[0,:,:] + 1)) * 100
        t2_rel_diff = 100 - ((t2_difference + 1) / (ground_truth.squeeze()[1,:,:] + 1)) * 100

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
        plt.close(fig)

        utils.save_image(
            z[:, 0, :, :].unsqueeze(1),
            T1_PRED_DIR / f"{i}.png",
            normalize=True
        )   
        utils.save_image(
            z[:, 1, :, :].unsqueeze(1),
            T2_PRED_DIR / f"{i}.png",
            normalize=True
        )   
        utils.save_image(
            ground_truth[:, 0, :, :].unsqueeze(1),
            T1_GT_DIR / f"{i}.png",
            normalize=True
        ) 
        utils.save_image(
            ground_truth[:, 1, :, :].unsqueeze(1),
            T2_GT_DIR / f"{i}.png",
            normalize=True
        )
        utils.save_image(
            t1_difference,
            T1_DIFF / f"{i}.png",
            normalize=True
        )
        utils.save_image(
            t2_difference,
            T2_DIFF / f"{i}.png",
            normalize=True
        )"""


    #log_artifacts(ARTIFACT_DIR)
    times_np = np.array(times)
    print("times total", times_np.sum(), "mean=", times_np.mean()) # 0.5921916961669922 mean= 0.03947944641113281
    print("dropping first element")
    times_short_np = times_np[1:]
    print("times total", times_short_np.sum(), "mean=", times_short_np.mean())
    # total 0.34842538833618164 mean= 0.02488752773829869