import torch
from torch import nn
from torch.functional import norm
from torchvision.models.resnet import BasicBlock


class AutoEncoder(nn.Module):

    def __init__(self, input_channels, output_channels):
        super().__init__()
        # The mean and standard deviation for T1 and T2 values in training data
        mean = torch.FloatTensor([0.5113866338173979, 0.06698109627281001]).view(1, 2, 1, 1)
        self.register_buffer("mean", mean)
        std = torch.FloatTensor([0.7184752571113431, 0.12903516669471898]).view(1, 2, 1, 1)
        self.register_buffer("std", std)

        # Use make_down_layer function to compose encoder
        self.encoder = nn.Sequential(
            self.make_down_layer(2, 64),
            self.make_down_layer(64, 128),
            self.make_down_layer(128, 256),
            self.make_down_layer(256, 512)
        )
        # Use UpLayer class below to compose decoder
        self.decoder = nn.Sequential(
            UpLayer(512, 256),
            UpLayer(256, 128),
            UpLayer(128, 64),
            UpLayer(64, 2, activation=None, norm=False)
        )
    
    @staticmethod
    def make_down_layer(in_channels, out_channels, kernel_size=4, stride=2, padding=1, activation=nn.ReLU, norm=True):
        layer = []

        layer.append(BasicBlock(in_channels, in_channels))
        layer.append(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding))
        
        if norm:
            layer.append(nn.BatchNorm2d(out_channels))

        if activation is not None:
            layer.append(activation())

        return nn.Sequential(*layer)

    def forward(self, x):
        # Normalise input data 
        x = (x - self.mean) / self.std.clamp(min=1e-6)
        x = torch.tanh(x)
        z = self.encoder(x) # encoded image
        x_hat = self.decoder(z) # reconstructed image
        # Returns the reconstructed image as well as the final feature layer of the encoder
        return x_hat, z

       
class UpLayer(nn.Module):

    def __init__(self, in_channels, out_channels, activation=nn.ReLU, norm=True):
        super().__init__()

        self.residual = BasicBlock(in_channels, in_channels)

        layers = []
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1))

        if norm:
            layers.append(nn.BatchNorm2d(out_channels))

        if activation is not None:
            layers.append(activation())

        
        self.main = nn.Sequential(*layers)


    def forward(self, x):
        x = self.residual(x)
        # upsample x
        x_big = nn.functional.interpolate(x, scale_factor=2)
        z = self.main(x_big)
        # Return the upsampled input
        return z
