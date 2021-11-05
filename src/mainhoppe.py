"""
imports
"""
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from data.RNNdataset import RNNDataset, make_dataset_and_sampler
from model.hoppeRNN import Hoppernn
import matplotlib
from matplotlib import pyplot as plt
from torchvision import utils
from itertools import chain
import mlflow
from mlflow import log_metric, log_param, log_artifacts, log_image
from pathlib import Path
from tqdm import tqdm
matplotlib.use("Agg")

experiment_name = "Hoppe_RNN"
try:
    experiment_id = mlflow.create_experiment(experiment_name)
    experiment = mlflow.get_experiment(experiment_id)
except:
    experiment = mlflow.get_experiment_by_name(experiment_name)
    experiment_id = experiment.experiment_id


torch.manual_seed(0)
CUDA = torch.cuda.is_available()

"""
args and constants
"""
BATCH_SIZE = 16
LR = 0.0004
N_EPOCHS = 100



"""
paths
"""



"""
dataset and dataloader
"""
dataset, sampler = make_dataset_and_sampler(
    "clustered_paths.txt"
)

dataloader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    num_workers=8,
    shuffle=False,
    sampler=sampler
)




"""
model
"""
model = Hoppernn(CUDA)

if CUDA:
    model = model.cuda()


"""
optimiser and loss
"""
optim_model = optim.Adam(model.parameters())  #, lr=0.0002, betas=(0.5, 0.999))
criterion = nn.MSELoss()



"""
training loop
"""
with mlflow.start_run(experiment_id=experiment_id) as run:
    run_id = run.info.run_id
    ARTIFACT_DIR = f"./artifacts/{experiment_id}/{run_id}/"
    STATE_DIR = Path(ARTIFACT_DIR) / "States"

    paths = [FINGERPRINTS, STATE_DIR]
    for p in paths:
        p.mkdir(exist_ok=True, parents=True)

    log_param("batch_size", BATCH_SIZE)
    log_param("learning_rate", LR)
    log_param("n_epochs", N_EPOCHS)

    for epoch in range(N_EPOCHS):
        print("epoch=", epoch)
        for i, (x, gt, c) in tqdm(enumerate(dataloader), total=len(dataloader)):

            if CUDA:
                x = x.cuda()
                gt = gt.cuda()

            batch_size = x.shape[0]
            #print("batch size", batch_size)
            #print("x in main shape", x.shape)

            """
            TRAINING
            """
            optim_model.zero_grad()
            
            #with torch.no_grad():
            y = model(x)
            #print("y shape after model", y.shape)
            #print("gt shape", gt.shape)
            loss_value = criterion(y, gt)
            log_metric("train_loss", loss_value.item())
            #print("loss", loss_value)

            loss_value.backward()
            optim_model.step()



            """
            END OF TRAINING
            """
            


            if i % 10 == 0:
                print("loss", loss_value)
            #     print("gen loss", loss_generator.item())
            #     print("dis loss", loss_discriminator.item())



        if epoch % 10 == 0 and epoch > 0:
            torch.save(model.state_dict(), STATE_DIR / f"model-{epoch}.state")

        log_artifacts(ARTIFACT_DIR)
