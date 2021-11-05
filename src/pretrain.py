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

experiment_name = "GAN_pretraining"

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
BATCH_SIZE = 4
LR = 0.0002
N_EPOCHS = 100


"""
dataset and dataloader
"""
print("building dataset")
dataset = FingerprintDataset(
    "/pytorch_data/train/"
)

print("building dataloader")
dataloader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    num_workers=4,
    shuffle=True
)




"""
model
"""
#model = UNet()
print("building models")
generator = Unet_3ds_rcab(10, 1000).float()
coarse_model = Coarse_net().float()
discriminator = Unet_3ds_rcab(64, 2).float()

if CUDA:
    generator = generator.to(torch.device("cuda:0"))
    coarse_model = coarse_model.to(torch.device("cuda:0"))
    discriminator = discriminator.to(torch.device("cuda:0"))



"""
optimiser and loss
"""
print("building optimisers and loss functions")
criterion = nn.MSELoss()
optim_discriminator = optim.Adam(
    chain(
        discriminator.parameters(),
        coarse_model.parameters()
    ),
    lr=LR
)
optim_generator = optim.Adam(generator.parameters(), lr=LR)





"""
training loop
"""
print("beginning training loop")
with mlflow.start_run(experiment_id=experiment_id) as run:
    run_id = run.info.run_id
    # make required directories
    ARTIFACT_DIR = f"./artifacts/{experiment_id}/{run_id}/"
    FINGERPRINTS = Path(ARTIFACT_DIR) / "FINGERPRINTS"
    FAKE = Path(ARTIFACT_DIR) / "FAKE"
    STATE_DIR = Path(ARTIFACT_DIR) / "States"

    paths = [FINGERPRINTS, FAKE, STATE_DIR]
    for p in paths:
        p.mkdir(exist_ok=True, parents=True)


    log_param("batch_size", BATCH_SIZE)
    log_param("learning_rate", LR)
    log_param("n_epochs", N_EPOCHS)
    for epoch in tqdm(range(N_EPOCHS)):
        print("epoch", epoch)
        for batch_id, (batch_of_data, ground_truth) in enumerate(dataloader):
            print("batch", batch_id)
            #print("batch_of_data", batch_of_data.shape)

            z = torch.randn(BATCH_SIZE, 10, 256, 256)

            if CUDA:
                batch_of_data = batch_of_data.to(torch.device("cuda:0"))
                ground_truth = ground_truth.to(torch.device("cuda:0"))
                z = z.to(torch.device("cuda:0"))

            optim_discriminator.zero_grad()
            optim_generator.zero_grad()

            """
            GENERATOR TRAINING
            """

            x_fake = generator(z)
            print("x_fake:", x_fake.shape)
            y_fake_map = discriminator(coarse_model(x_fake))
            #print("y_fake_map:", y_fake_map.shape)
            y_fake = y_fake_map.view(-1, 2, 256*256).mean(dim=2)
            #print("y_fake shape:", y_fake.shape)

            loss_generator = 0.5 * torch.pow(y_fake[:, 0] - 1, 2).mean()
            loss_generator.backward()
            optim_generator.step()
            log_metric("loss_generator", loss_generator.item())
            

            """
            END OF GENERATOR TRAINING
            """

            """
            DISCRIMINATOR TRAINING
            """
            optim_discriminator.zero_grad()
            
            with torch.no_grad():
                z = torch.randn(1, 10, 256, 256, device=z.device)
                x_fake = generator(z)

            y_real_map = discriminator(coarse_model(batch_of_data))
            y_real = y_real_map.view(-1, 2, 256*256).mean(dim=2)[:, 0]

            y_fake_map = discriminator(coarse_model(x_fake))
            y_fake = y_fake_map.view(-1, 2, 256*256).mean(dim=2)[:, 0]

            loss_discriminator = 0.5 * (torch.pow(y_real - 1, 2) + torch.pow(y_fake, 2)).mean()
            loss_discriminator.backward()
            optim_discriminator.step()
            log_metric("loss_discriminator", loss_discriminator.item())
            



            """
            END OF DISCRIMINATOR TRAINING
            """
            
            #print(batch_of_data.shape)

            # utils.save_image(
            #     batch_of_data[:, :3, :, :],
            #     "./data.png",
            #     normalize=True
            # )

            #print(batch_of_data[0,100,100,:])

            #print("z shape:", z.shape, z.min().item(), z.max().item())
            #print("ground truth shiz: ", ground_truth.shape, ground_truth.min().item(), ground_truth.max().item())


            # print("z2 shape:", z2.shape)
            #break

        # epoch end

        utils.save_image(
            x_fake[:, 5, :, :].unsqueeze(1),
            FAKE / f"{epoch}.png",
            normalize=True
        )   
        fig,ax = plt.subplots(1, 2)
        ax[0].plot(x_fake[0, :, 125, 125])
        ax[1].plot(batch_of_data[0, :, 125, 125])
        plt.savefig(FINGERPRINTS / f"{epoch}.png")
        plt.close(fig)
    
        if epoch % 10 == 0 and epoch > 0:
            torch.save(coarse_model.state_dict(), STATE_DIR / f"coarse_model-{epoch}.state")
            torch.save(discriminator.state_dict(), STATE_DIR / f"discriminator-{epoch}.state")
            torch.save(generator.state_dict(), STATE_DIR / f"generator-{epoch}.state")
        log_artifacts(ARTIFACT_DIR)
