import torch
from torch import nn
import functools
from .rcab import RCAB

class Unet_3ds_rcab(nn.Module):
    #def __init__(self, opt, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, gpu_ids=[]):
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, gpu_ids=[]):
        super(Unet_3ds_rcab, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        n_rcab = 1
        
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        
        model = {}

        model['down1'] = [
            nn.Conv2d(self.input_nc, self.ngf, kernel_size=3, padding=1, bias=use_bias),
            norm_layer(self.ngf),
            self.make_rcab(ngf, n_rcab),
            nn.ReLU(True)
            
            ]
        self.down1 = nn.Sequential(*model['down1'])

        model['down2'] = [
            nn.MaxPool2d(2), 
            nn.Conv2d(self.ngf, self.ngf*2, kernel_size=3, padding=1, bias=use_bias),
            norm_layer(self.ngf*2),
            nn.ReLU(True),
            self.make_rcab(ngf*2, n_rcab)
            
            ]
        self.down2 = nn.Sequential(*model['down2'])

        model['down3'] = [
            nn.MaxPool2d(2), 
            nn.Conv2d(self.ngf*2, self.ngf*4, kernel_size=3, padding=1, bias=use_bias),
            norm_layer(self.ngf*4),
            nn.ReLU(True),
            self.make_rcab(ngf*4, n_rcab)
            ]
        self.down3 = nn.Sequential(*model['down3'])


        model['b'] = [
            nn.MaxPool2d(2), 
            nn.Conv2d(self.ngf*4, self.ngf*8, kernel_size=3, padding=1, bias=use_bias),
            norm_layer(self.ngf*8),
            nn.ReLU(True),
            self.make_rcab(ngf*8, n_rcab),
            nn.ConvTranspose2d(self.ngf*8, self.ngf*4, kernel_size=2, stride=2)
            # norm_layer(self.ngf*2),
            # nn.ReLU(True)
            ]
        self.b = nn.Sequential(*model['b'])

        model['up3'] = [
            nn.Conv2d(self.ngf*8, self.ngf*4, kernel_size=3, padding=1, bias=use_bias),
            norm_layer(self.ngf*4),
            nn.ReLU(True),
            self.make_rcab(ngf*4, n_rcab),
            nn.ConvTranspose2d(self.ngf*4, self.ngf*2, kernel_size=2, stride=2)
            # norm_layer(self.ngf),
            # nn.ReLU(True)
            ]
        self.up3 = nn.Sequential(*model['up3'])
        

        model['up2'] = [
            nn.Conv2d(self.ngf*4, self.ngf*2, kernel_size=3, padding=1, bias=use_bias),
            norm_layer(self.ngf*2),
            nn.ReLU(True),
            self.make_rcab(ngf*2, n_rcab),
            nn.ConvTranspose2d(self.ngf*2, self.ngf, kernel_size=2, stride=2)
            # norm_layer(self.ngf),
            # nn.ReLU(True)
            ]
        self.up2 = nn.Sequential(*model['up2'])


        model['up1'] = [
            nn.Conv2d(self.ngf*2, self.ngf, kernel_size=3, padding=1, bias=use_bias),
            norm_layer(self.ngf),
            nn.ReLU(True),
            self.make_rcab(ngf, n_rcab),
            nn.Conv2d(self.ngf, self.output_nc, kernel_size=3, padding=1)
            # norm_layer(self.output_nc),
            # nn.ReLU(True)
            ]
        self.up1 = nn.Sequential(*model['up1'])
        
        # self.model = model

    def make_rcab(self, n_feat, n_block):
        if n_block==1:
            return RCAB(n_feat=n_feat)
        rcab_chain = []
        for i in range(n_block):
            rcab_chain.append(RCAB(n_feat=n_feat))
        return nn.Sequential(*rcab_chain)

    def forward(self, input):
        fd1 = self.down1(input)
        fd2 = self.down2(fd1)
        fd3 = self.down3(fd2)
        fu3 = torch.cat([fd3, self.b(fd3)], 1)
        fu2 = torch.cat([fd2, self.up3(fu3)], 1)
        fu1 = torch.cat([fd1, self.up2(fu2)], 1)
        return self.up1(fu1)