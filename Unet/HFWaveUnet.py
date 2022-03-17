import torch.nn as nn
import torch
from DWT_IDWT_Layer import *
from HFM import *

class CBR2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True):
        super(CBR2d, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                kernel_size=kernel_size, stride=stride, padding=padding,
                                bias=bias)
        self.batch_norm = nn.BatchNorm2d(num_features=out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.batch_norm(x)
        x = self.relu(x)
        return x

class Encoder_block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Encoder_block, self).__init__()
        self.pool_layer = DWT_2D('haar')
        self.enc1 = CBR2d(in_channels=in_channels[0], out_channels=out_channels[0])
        self.enc2 = CBR2d(in_channels=in_channels[1], out_channels=out_channels[1])

    def forward(self, x):
        x1 = self.enc1(x)
        x2 = self.enc2(x1)
        LL, LH, HL, HH = self.pool_layer(x2)
        return x2, LL, LH, HL, HH

class Decoder_block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Decoder_block, self).__init__()

        # Pooling Layer
        self.unpool = IDWT_2D('haar')

        # Attention Mopdule
        self.hfm = HFM(in_channels[1])
            
        self.dec1 = CBR2d(in_channels=in_channels[0], out_channels=out_channels[0])
        self.dec2 = CBR2d(in_channels=in_channels[1], out_channels=out_channels[1])
        self.conv = nn.Conv2d(in_channels=out_channels[1], out_channels=out_channels[1]//2, kernel_size=1, bias=True)

    def forward(self, x, LH, HL, HH, cat):
        hf1, hf2, hf3 = self.hfm(LH, HL, HH)
        pool_x = self.unpool(x, LH, HL, HH)
        cat_x = torch.cat((pool_x, cat), dim=1)
        x1 = self.dec1(cat_x)
        x2 = self.dec2(x1)
        x3 = self.conv(x2)
        return x3



class HFWaveUNet(nn.Module):
    def __init__(self, encoder_in_channels, encoder_out_channels, decoder_in_channels, decoder_out_channels):
        super(HFWaveUNet, self).__init__()
        # Contracting path
        self.encoder_1 = Encoder_block(in_channels=encoder_in_channels[0], out_channels=encoder_out_channels[0])
        self.encoder_2 = Encoder_block(in_channels=encoder_in_channels[1], out_channels=encoder_out_channels[1])
        self.encoder_3 = Encoder_block(in_channels=encoder_in_channels[2], out_channels=encoder_out_channels[2])
        self.encoder_4 = Encoder_block(in_channels=encoder_in_channels[3], out_channels=encoder_out_channels[3])
        self.encoder_5 = Encoder_block(in_channels=encoder_in_channels[4], out_channels=encoder_out_channels[4])

        self.encoder_6 = CBR2d(in_channels=512, out_channels=1024)
        
        # Expansive path
        self.decoder_6 = CBR2d(in_channels=1024, out_channels=1024)
        self.conv = nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=1, bias=True)

        self.decoder_5 = Decoder_block(in_channels=decoder_in_channels[0], out_channels=decoder_out_channels[0])
        self.decoder_4 = Decoder_block(in_channels=decoder_in_channels[1], out_channels=decoder_out_channels[1])
        self.decoder_3 = Decoder_block(in_channels=decoder_in_channels[2], out_channels=decoder_out_channels[2])
        self.decoder_2 = Decoder_block(in_channels=decoder_in_channels[3], out_channels=decoder_out_channels[3])
        self.decoder_1 = Decoder_block(in_channels=decoder_in_channels[4], out_channels=decoder_out_channels[4])

        self.fc = nn.Conv2d(in_channels=16, out_channels=1, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x):
        enc1_1, enc1_LL, enc1_LH, enc1_HL, enc1_HH = self.encoder_1(x)
        enc2_1, enc2_LL, enc2_LH, enc2_HL, enc2_HH = self.encoder_2(enc1_LL)
        enc3_1, enc3_LL, enc3_LH, enc3_HL, enc3_HH = self.encoder_3(enc2_LL)
        enc4_1, enc4_LL, enc4_LH, enc4_HL, enc4_HH = self.encoder_4(enc3_LL)
        enc5_1, enc5_LL, enc5_LH, enc5_HL, enc5_HH = self.encoder_5(enc4_LL)

        enc6 = self.encoder_6(enc5_LL)
        dec6 = self.decoder_6(enc6)
        dec6 = self.conv(dec6)

        dec5 = self.decoder_5(dec6, enc5_LH, enc5_HL, enc5_HH, enc5_1)
        dec4 = self.decoder_4(dec5, enc4_LH, enc4_HL, enc4_HH, enc4_1)
        dec3 = self.decoder_3(dec4, enc3_LH, enc3_HL, enc3_HH, enc3_1)
        dec2 = self.decoder_2(dec3, enc2_LH, enc2_HL, enc2_HH, enc2_1)
        dec1 = self.decoder_1(dec2, enc1_LH, enc1_HL, enc1_HH, enc1_1)
        out = self.fc(dec1)

        return out

def get_hfwaveunet():
    en_in_channels = [[1, 32], [32, 64], [64, 128], [128, 256], [256, 512]]
    en_out_channels = [[32, 32], [64, 64], [128, 128], [256, 256], [512, 512]]

    de_in_channels = [[2*512,512], [2*256,256], [2*128,128], [2*64,64], [2*32, 32]]
    de_out_channels = [[512,512], [256,256], [128,128], [64,64], [32, 32]]
    return HFWaveUNet(encoder_in_channels=en_in_channels, encoder_out_channels=en_out_channels, decoder_in_channels=de_in_channels, decoder_out_channels=de_out_channels)