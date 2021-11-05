"""
imports
"""
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from data.RNNdataset import RNNDataset, make_dataset_and_sampler
from model.rnn import RNNGenerator, RNNDiscriminator
import matplotlib
from matplotlib import pyplot as plt
from torchvision import utils
from itertools import chain
import mlflow
from mlflow import log_metric, log_param, log_artifacts, log_image
from pathlib import Path
from tqdm import tqdm
matplotlib.use("Agg")

# GAN training

# MLFlow for logging metrics and results
experiment_name = "RNN_pretraining"
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
DIS_MSELOSS_FREQ = 16
SEQ_LENGTH = 600

# Discriminator parameters
CONV_INPUT_SIZE = 1
INPUT_SIZE = 32
HIDDEN_SIZE = 64
NUM_LAYERS = 1

"""
paths
"""



"""
dataset and dataloader
"""
dataset, sampler = make_dataset_and_sampler(
    "clustered_paths.txt",
    seq_length=SEQ_LENGTH
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
generator = RNNGenerator(CUDA, seq_gen_length=int(SEQ_LENGTH/8))
discriminator = RNNDiscriminator(CONV_INPUT_SIZE, INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS)


if CUDA:
    generator = generator.cuda()
    discriminator = discriminator.cuda()

"""
optimiser and loss
"""
optim_generator = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
optim_discriminator = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
criterion = nn.BCELoss()
#real_label = 0.
#fake_label = 1.
other_criterion = nn.MSELoss()

"""
training loop
"""
with mlflow.start_run(experiment_id=experiment_id) as run:
    # make required directories to save states and outputs
    run_id = run.info.run_id
    ARTIFACT_DIR = f"./artifacts/{experiment_id}/{run_id}/"
    FINGERPRINTS = Path(ARTIFACT_DIR) / "FINGERPRINTS"
    STATE_DIR = Path(ARTIFACT_DIR) / "States"

    paths = [FINGERPRINTS, STATE_DIR]
    for p in paths:
        p.mkdir(exist_ok=True, parents=True)

    log_param("batch_size", BATCH_SIZE)
    log_param("learning_rate", LR)
    log_param("n_epochs", N_EPOCHS)
    log_param("conv_input_size", CONV_INPUT_SIZE)
    log_param("dis_input_size", INPUT_SIZE)
    log_param("dis_hidden_size", HIDDEN_SIZE)
    log_param("dis_layers", NUM_LAYERS)
    log_param("MSE_loss_usage_freq", DIS_MSELOSS_FREQ)
    log_param("sequence_length", SEQ_LENGTH)
    for epoch in range(N_EPOCHS):
        print("epoch=", epoch)
        for i, (x_real, _, c) in tqdm(enumerate(dataloader), total=len(dataloader)):

            if CUDA:
                x_real = x_real.cuda()

            batch_size = x_real.shape[0]

            """
            DISCRIMINATOR TRAINING
            """
            optim_discriminator.zero_grad()
            # Generate fake fingerprint signal
            x_fake = generator(batch_size)
            # Label smoothing
            label = torch.normal(0.9, 0.15, (batch_size,), device=x_real.device).clamp(min=0.7, max=1.0)

            # Push real fingerprint through discriminator
            y_real = discriminator(x_real).view(-1)
            # Calculate discriminator loss for real fingerprint
            loss_discriminator_real = criterion(y_real, label)
            loss_discriminator_real.backward() # backwards pass

            # Label smoothing
            label = torch.normal(0.1, 0.15, (batch_size,), device=x_real.device).clamp(min=0.0, max=0.3)
            # Push generated fingerprint through discriminator
            y_fake = discriminator(x_fake.detach()).view(-1)
            # Calculate discriminator loss for generated fingerprint
            loss_discriminator_fake = criterion(y_fake, label)
            loss_discriminator_fake.backward() # backwards pass

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(discriminator.parameters(), 1.0)

            # Discriminator loss
            loss_discriminator = loss_discriminator_real + loss_discriminator_fake

            optim_discriminator.step() # Update weights
            

            """
            END OF DISCRIMINATOR TRAINING
            """


            """
            GENERATOR TRAINING
            """
            optim_generator.zero_grad()
            label.fill_(1.0)

            # Push generated fingerprint through discriminator and calculate generator loss
            y_fake = discriminator(x_fake).view(-1)
            loss_generator = criterion(y_fake, label)

            # Reconstruction loss added to generator loss ever 50 iterations
            if i % 50 == 0:
                MSE_loss = other_criterion(x_fake, x_real)
                total_loss = loss_generator + MSE_loss
                total_loss.backward()
                log_metric("MSE_generator", MSE_loss.item())
            else:
                loss_generator.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(generator.parameters(), 1.0)

            optim_generator.step() # UPdate weights
            
            
            
            """
            END OF GENERATOR TRAINING
            """

            if i % 200 == 0:
                log_metric("loss_generator", loss_generator.item())
                log_metric("loss_discriminator", loss_discriminator.item())
                x_real = x_real.detach().cpu()
                x_fake = x_fake.detach().cpu()
                fig,ax = plt.subplots(8, 2, figsize=(8, 8))
                rand_idx = torch.randint(0, len(x_real), size=(8,))
                ax[0,0].set_title("Real")
                ax[0,1].set_title("Fake")
                for j in range(8):
                    ax[j,0].plot(x_real[rand_idx[j]])
                    ax[j,0].set_ylim([-1.0, 1.0])
                    ax[j,1].plot(x_fake[rand_idx[j]])
                    ax[j,1].set_ylim([-1.0, 1.0])
                plt.savefig(FINGERPRINTS / f"{epoch}.png")
                plt.close(fig)

                c_flat = c.view(-1)
                fig, ax = plt.subplots()
                ax.hist(c_flat, bins=10, range=[0, 10])
                plt.savefig(FINGERPRINTS / "class_balance.png")
                plt.close(fig)

        if epoch % 2 == 0 and epoch > 0:
            torch.save(discriminator.state_dict(), STATE_DIR / f"discriminator-{epoch}.state")
            torch.save(generator.state_dict(), STATE_DIR / f"generator-{epoch}.state")
        log_artifacts(ARTIFACT_DIR)
