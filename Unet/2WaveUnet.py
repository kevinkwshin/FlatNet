import torch
import torch.nn as nn
from DWT_IDWT_Layer import *

def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.BatchNorm2d(num_features=out_channels),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.BatchNorm2d(num_features=out_channels)
    )

class maxpool(nn.Module):
    def __init__(self):
        super(maxpool, self).__init__()
        self.pool = nn.MaxPool2d(2)
    def forward(self,x):
        x = self.pool(x)
        return (x,None,None,None)

class upsample(nn.Module):
    def __init__(self):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
    def forward(self,x ,lh,hl,hh):
        x = self.upsample(x)
        return x

class Wave_UNet(nn.Module):
    def __init__(self, n_channels, n_classes, pool='maxpool'):
        super().__init__()            
        self.encoder1 = double_conv(n_channels,64)
        self.encoder2 = double_conv(64,128)
        self.encoder3 = double_conv(128,256)
        self.encoder4 = double_conv(256,512)
        self.encoder5 = double_conv(512,1024)
        
        self.maxpool = maxpool()
        self.upsample = upsample()

        if pool=='wavelet':
            self.maxpool = DWT_2D('haar')
            self.upsample = IDWT_2D('haar')

        self.decoder1 = double_conv(1024,512)
        self.decoder2 = double_conv(1024,512)
        self.decoder3 = double_conv(512,256)
        self.decoder4 = double_conv(256,128)
        self.decoder5 = double_conv(128,64)
        self.out = nn.Conv2d(64, n_classes, 3, padding=1)
        
    def forward(self, x):
        # 1 -> 64
        conv1 = self.encoder1(x)
        x1 = self.maxpool(conv1)
        # 62 -> 128
        conv2 = self.encoder2(x1[0])
        x2 = self.maxpool(conv2)
        # 128 -> 256
        conv3 = self.encoder3(x2[0])
        x3 = self.maxpool(conv3)
        # 256->512
        conv4 = self.encoder4(x3[0])
        x4 = self.maxpool(conv4)
        # 512->1024
        x = self.encoder5(x4[0])
        
        print(x4[1].shape)
        print(x3[1].shape)
        print(x2[1].shape)
        print(x1[1].shape)
        
        # 1024 -> 512
        x = self.decoder1(x)
        x = self.upsample(x,x4[1],x4[2],x4[3])
        x = torch.cat([x, conv4], dim=1)
        #1024 -> 512
        x = self.decoder2(x)
        x = self.upsample(x,x3[1],x3[2],x3[3])
        x = torch.cat([x, conv3], dim=1)
        # 512 -> 256
        x = self.decoder3(x)
        x = self.upsample(x,x3[1],x3[2],x3[3])
        x = torch.cat([x, conv2], dim=1)
        # 256 -> 128
        x = self.decoder4(x)
        x = self.upsample(x,x2[1],x2[2],x2[3])
        x = torch.cat([x, conv1], dim=1)
        # 128 -> 64
        x = self.decoder5(x)
        # 64 -> 1
        x = self.out(x)
        return x