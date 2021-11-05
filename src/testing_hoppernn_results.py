from typing import Sequence
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from data.dataset_with_crop import FingerprintDataset
#from data.dataset import FingerprintDataset
from model.fang.unet_fang import Unet_3ds_rcab
from model.pretrained_rnn import PretrainedRNN
from model.coarse_model import Coarse_net
from model.hoppeRNN import Hoppernn
from matplotlib import pyplot as plt
from torchvision import utils
from itertools import chain
import mlflow
from mlflow import log_metric, log_param, log_artifacts, log_image
from pathlib import Path
from tqdm import tqdm

try:
    experiment_id = mlflow.create_experiment("testing_hoppe_rnn")
    experiment = mlflow.get_experiment(experiment_id)
except:
    experiment = mlflow.get_experiment_by_name("testing_hoppe_rnn")
    experiment_id = experiment.experiment_id


torch.manual_seed(0)
CUDA = torch.cuda.is_available()

"""
args and constants
"""
BATCH_SIZE = 1
SEQ_LENGTH = 1000
CROP = 200

"""
dataset and dataloader
"""
# dataset = FingerprintDataset(
#     "/media/DataSSD/datasets/MRI_Issie/test/",
#     random_crop=CROP,
#     seq_length=SEQ_LENGTH
# )
dataset = FingerprintDataset(
    "/media/DataSSD/datasets/MRI_Issie/test/",
    random_crop=CROP,
    seq_length=SEQ_LENGTH
)

dataloader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    num_workers=1,
    shuffle=True
)

"""
model
"""
model = Hoppernn(CUDA)
model.load_state_dict(
    torch.load("/home/issie/code/py/MRI/src/mlruns/3/cb1df3bf55c94ca18a3b6ce2cc5cdcf4/artifacts/States/model-60.state")
)
model.eval()


if CUDA:
    model = model.to(torch.device("cuda:0"))



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
    log_param("crop", CROP)


    for i, (batch_of_data, ground_truth) in tqdm(enumerate(dataloader), total=len(dataloader)):

        if CUDA:
            batch_of_data = batch_of_data.to(torch.device("cuda:0"))
            ground_truth = ground_truth.to(torch.device("cuda:0"))

        batch_of_data = batch_of_data.squeeze()
        #print("squeezed", batch_of_data.shape)
        reshaped = batch_of_data.view(SEQ_LENGTH, -1)
        #print("reshaped1", reshaped.shape)
        reshaped = reshaped.permute(1, 0)
        #print("reshaped2", reshaped.shape)
        with torch.no_grad():
            #print(batch_of_data[0,100,100,:])
            z = model(reshaped.float())
            #print("z1 post prelim", z1.shape)
            z = z.permute(1, 0)
            #z1 = z1.view(-1, 232, 232).unsqueeze(0)
            z = z.view(-1, CROP, CROP).unsqueeze(0)
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
        )

    log_artifacts(ARTIFACT_DIR)