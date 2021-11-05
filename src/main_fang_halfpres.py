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
from tqdm import tqdm


try:
    experiment_id = mlflow.create_experiment("baseline_unet_w_coarse_model")
    experiment = mlflow.get_experiment(experiment_id)
except:
    experiment = mlflow.get_experiment_by_name("baseline_unet_w_coarse_model")
    experiment_id = experiment.experiment_id


torch.manual_seed(0)
CUDA = torch.cuda.is_available()

"""
args and constants
"""
BATCH_SIZE = 1
LR = 0.0002
N_EPOCHS = 100
SEQ_LENGTH = 600


"""
dataset and dataloader
"""
dataset = FingerprintDataset(
    "/media/DataSSD/datasets/MRI_Issie/train/",
    seq_length=600
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
#model = UNet()
model = Unet_3ds_rcab(64, 2).float()
coarse_model = Coarse_net(input_channels=SEQ_LENGTH).float()

if CUDA:
    model = model.to(torch.device("cuda:0"))
    coarse_model = coarse_model.to(torch.device("cuda:0"))



"""
optimiser and loss
"""
criterion = nn.MSELoss()
optim_model = optim.Adam(
    chain(
        model.parameters(),
        coarse_model.parameters()
    ),
    lr=LR
)





"""
training loop
"""
scaler = torch.cuda.amp.GradScaler()

with mlflow.start_run(experiment_id=experiment_id) as run:
    run_id = run.info.run_id
    # make required directories
    ARTIFACT_DIR = f"./artifacts/{experiment_id}/{run_id}/"
    T1_GT_DIR = Path(ARTIFACT_DIR) / "T1_GT"
    T1_PRED_DIR = Path(ARTIFACT_DIR) / "T1_PRED"
    T2_GT_DIR = Path(ARTIFACT_DIR) / "T2_GT"
    T2_PRED_DIR = Path(ARTIFACT_DIR) / "T2_PRED"
    STATE_DIR = Path(ARTIFACT_DIR) / "States"

    paths = [T1_GT_DIR, T1_PRED_DIR, T2_GT_DIR, T2_PRED_DIR, STATE_DIR]
    for p in paths:
        p.mkdir(exist_ok=True, parents=True)


    log_param("batch_size", BATCH_SIZE)
    log_param("learning_rate", LR)
    log_param("n_epochs", N_EPOCHS)
    log_param("seq_length", SEQ_LENGTH)
    for epoch in range(N_EPOCHS):
        for i, (batch_of_data, ground_truth) in tqdm(enumerate(dataloader), total=len(dataloader)):
            with torch.cuda.amp.autocast():
                if CUDA:
                    batch_of_data = batch_of_data.to(torch.device("cuda:0"))
                    ground_truth = ground_truth.to(torch.device("cuda:0"))
                optim_model.zero_grad()
                
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
            #print("loss", loss_value)
            log_metric("MSE_loss", loss_value.item())
            scaler.scale(loss_value).backward() # backwards pass

            scaler.step(optim_model) # update weights
            scaler.update()

            # print("z2 shape:", z2.shape)
            #break

        # epoch end
        z = z.detach().cpu().float()
        ground_truth = ground_truth.detach().cpu().float()

        utils.save_image(
            z[:, 0, :, :].unsqueeze(1),
            T1_PRED_DIR / f"{epoch}.png",
            normalize=True
        )   
        utils.save_image(
            z[:, 1, :, :].unsqueeze(1),
            T2_PRED_DIR / f"{epoch}.png",
            normalize=True
        )   
        utils.save_image(
            ground_truth[:, 0, :, :].unsqueeze(1),
            T1_GT_DIR / f"{epoch}.png",
            normalize=True
        ) 
        utils.save_image(
            ground_truth[:, 1, :, :].unsqueeze(1),
            T2_GT_DIR / f"{epoch}.png",
            normalize=True
        )

        if epoch % 11 == 0:
            torch.save(coarse_model.state_dict(), STATE_DIR / f"coarse_model-{epoch}.state")
            torch.save(model.state_dict(), STATE_DIR / f"model-{epoch}.state")
        log_artifacts(ARTIFACT_DIR)