import torch
import torch.nn as nn
import torch.nn.functional as F

##########################################################################################################   
def dice(inputs, target):
    smooth = 1.
    iflat = inputs.view(-1)
    tflat = target.view(-1)
    intersection = (iflat*tflat).sum()
    return (2.*intersection + smooth) / (iflat.sum() + tflat.sum() + smooth)
    
def dice_loss(inputs,target):
    return 1-dice(inputs, target)
##########################################################################################################   
groupnorm_parameter = 8

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
#             nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
#             nn.BatchNorm2d(mid_channels),           
#             nn.ReLU(inplace=True),
            Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.GroupNorm(int(mid_channels/groupnorm_parameter),mid_channels),
            nn.GELU(),
#             nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(inplace=True),
            Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(int(out_channels/groupnorm_parameter),out_channels),
            nn.GELU()
        )

    def forward(self, x):
        return self.double_conv(x)
    

class MCDropout(nn.Dropout):
    
    def forward(self, input):
        return F.dropout(input, self.p, True, self.inplace)
    
def bayesian_inference_seg(net, x, activate, iteration=20):
    yhat_stack = []
    net.eval()

    with torch.no_grad():
        for idx in range(iteration):
            yhat = net(x)        
            yhat = activate(yhat)
            yhat_stack.append(yhat)

    yhat_stack = torch.stack(yhat_stack)
    pred = torch.mean(yhat_stack,0)
    
    uncert = torch.var(yhat_stack,0)
    uncert = torch.mean(uncert,1)  
    
    return pred, uncert

class Conv2d(nn.Conv2d):
    
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(Conv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        
    def forward(self, x):
        weight = self.weight
        weight_mean = weight.mean(dim=1, keepdim=True).mean(dim=2, keepdim=True).mean(dim=3, keepdim=True)
        weight = weight - weight_mean
        std = weight.view(weight.size(0), -1).std(dim=1).view(-1, 1, 1, 1) + 1e-5
        weight = weight / std.expand_as(weight)
        return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

##########################################################################################################       
class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)
##########################################################################################################    
class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            # 변경함!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            #self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
            self.conv = DoubleConv(in_channels, out_channels)
        else:
            self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)


    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)
##########################################################################################################   

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
#with_logistic_weight    
    
class UNet_Logistic(nn.Module):
    def __init__(self, n_channels = 1, n_classes = 1, bilinear=True):
        super(UNet_Logistic, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 32)
        self.down1 = Down(32, 64)
        self.down2 = Down(64, 128)
        self.down3 = Down(128, 256)
        self.down4 = Down(256, 512)
        self.down5 = Down(512, 1024)
        factor = 2 if bilinear else 1
        self.up1 = Up(1536, 512, bilinear)
        self.up2 = Up(768, 256, bilinear)
        self.up3 = Up(384, 128, bilinear)
        self.up4 = Up(192, 64, bilinear)
        self.up5 = Up(96, 32, bilinear)
        self.outc = OutConv(32, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x6 = self.down5(x5)
        x = self.up1(x6, x5)
        x = self.up2(x, x4)
        x = self.up3(x, x3)
        x = self.up4(x, x2)
        x = self.up5(x, x1)
        logits = self.outc(x)
        logits= torch.sigmoid(logits)
        return logits

    
class UNet(nn.Module):
    def __init__(self, n_channels = 1, n_classes = 1, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 32)
        self.down1 = Down(32, 64)
        self.down2 = Down(64, 128)
        self.down3 = Down(128, 256)
        self.down4 = Down(256, 512)
        self.down5 = Down(512, 1024)
        self.MCDropout = MCDropout(0.5)
        factor = 2 if bilinear else 1
        self.up1 = Up(1536, 512, bilinear)
        self.up2 = Up(768, 256, bilinear)
        self.up3 = Up(384, 128, bilinear)
        self.up4 = Up(192, 64, bilinear)
        self.up5 = Up(96, 32, bilinear)
        self.outc = OutConv(32, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x6 = self.down5(x5)
#         x6 = self.MCDropout(x6)
        x = self.up1(x6, x5)
        x = self.up2(x, x4)
        x = self.up3(x, x3)
        x = self.up4(x, x2)
        x = self.up5(x, x1)
        logits = self.outc(x)
        return logits
    
class UNet_4(nn.Module):
    def __init__(self, n_channels = 1, n_classes = 1, bilinear=True):
        super(UNet_4, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 32)
        self.down1 = Down(32, 64)
        self.down2 = Down(64, 128)
        self.down3 = Down(128, 256)
        self.down4 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.up1 = Up(768, 256, bilinear)
        self.up2 = Up(384, 128, bilinear)
        self.up3 = Up(192, 64, bilinear)
        self.up4 = Up(96, 32, bilinear)
        self.outc = OutConv(32, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits