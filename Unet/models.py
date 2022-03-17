import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

groupnorm_parameter = 16

def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)

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

class conv_block(nn.Module):
    def __init__(self,ch_in,ch_out,norm='batch'):
        super(conv_block,self).__init__()
        inplace = True
#         inplace = False
        
        if norm=='batch':
            self.conv = nn.Sequential(
                nn.Conv2d(ch_in, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
                nn.BatchNorm2d(ch_out),
#                 nn.ReLU(inplace=inplace),
                nn.GELU(),
                nn.Conv2d(ch_out, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
                nn.BatchNorm2d(ch_out),
#                 nn.ReLU(inplace=inplace)
                nn.GELU(),

            )
        elif norm=='instance':
            self.conv = nn.Sequential(
                nn.Conv2d(ch_in, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
                nn.InstanceNorm2d(ch_out),
#                 nn.ReLU(inplace=inplace),
                nn.GELU(),
                nn.Conv2d(ch_out, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
                nn.InstanceNorm2d(ch_out),
                nn.ReLU(inplace=inplace)
            )
        elif norm=='group':
            self.conv = nn.Sequential(
                nn.Conv2d(ch_in, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
                nn.GroupNorm(int(ch_out/groupnorm_parameter),ch_out),
                nn.ReLU(inplace=inplace),
                nn.Conv2d(ch_out, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
                nn.GroupNorm(int(ch_out/groupnorm_parameter),ch_out),
#                 nn.ReLU(inplace=inplace)
                nn.GELU(),
            )
        elif norm=='ws':
            self.conv = nn.Sequential(
                Conv2d(ch_in, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
                nn.GroupNorm(int(ch_out/groupnorm_parameter),ch_out),
#                 nn.ReLU(inplace=inplace),
                nn.GELU(),
                Conv2d(ch_out, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
                nn.GroupNorm(int(ch_out/groupnorm_parameter),ch_out),
#                 nn.ReLU(inplace=inplace)
                nn.GELU(),
            )

    def forward(self,x):
        x = self.conv(x)
        return x

class up_conv(nn.Module):
    def __init__(self,ch_in,ch_out,norm='batch'):
        super(up_conv,self).__init__()
        
        inplace = True
#         inplace = False
    
        if norm=='batch':
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=2),
                nn.Conv2d(ch_in,ch_out,kernel_size=3,stride=1,padding=1,bias=True),
                nn.BatchNorm2d(ch_out),
#                 nn.ReLU(inplace=inplace)
                nn.GELU(),

            )
        elif norm=='instance':
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=2),
                nn.Conv2d(ch_in,ch_out,kernel_size=3,stride=1,padding=1,bias=True),
                nn.InstanceNorm2d(ch_out),
#                 nn.ReLU(inplace=inplace)
                nn.GELU(),

            )
        elif norm=='group':
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=2),
                nn.Conv2d(ch_in,ch_out,kernel_size=3,stride=1,padding=1,bias=True),
                nn.GroupNorm(int(ch_out/groupnorm_parameter),ch_out),
#                 nn.ReLU(inplace=inplace)
                nn.GELU(),
            )
        elif norm=='ws':
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=2),
                Conv2d(ch_in,ch_out,kernel_size=3,stride=1,padding=1,bias=True),
                nn.GroupNorm(int(ch_out/groupnorm_parameter),ch_out),
#                 nn.ReLU(inplace=inplace)
                nn.GELU(),
            )
    def forward(self,x):
        x = self.up(x)
        return x

class Recurrent_block(nn.Module):
    def __init__(self,ch_out,t=2,norm='batch'):
        super(Recurrent_block,self).__init__()
        self.t = t
        self.ch_out = ch_out
        inplace = True
#         inplace = False
        
        if norm=='batch':
            self.conv = nn.Sequential(
                nn.Conv2d(ch_out,ch_out,kernel_size=3,stride=1,padding=1,bias=True),
                nn.BatchNorm2d(ch_out),
#                 nn.ReLU(inplace=inplace)
                nn.GELU(),

            )
        elif norm=='instance':
            self.conv = nn.Sequential(
                nn.Conv2d(ch_out,ch_out,kernel_size=3,stride=1,padding=1,bias=True),
                nn.InstanceNorm2d(ch_out),
#                 nn.ReLU(inplace=inplace)
                nn.GELU(),
            )
        elif norm=='group':
            self.conv = nn.Sequential(
                nn.Conv2d(ch_out,ch_out,kernel_size=3,stride=1,padding=1,bias=True),
                nn.GroupNorm(int(ch_out/groupnorm_parameter),ch_out),
#                 nn.ReLU(inplace=inplace)
                nn.GELU(),
            )
        elif norm=='ws':
            self.conv = nn.Sequential(
                Conv2d(ch_out,ch_out,kernel_size=3,stride=1,padding=1,bias=True),
                nn.GroupNorm(int(ch_out/groupnorm_parameter),ch_out),
#                 nn.ReLU(inplace=inplace)
                nn.GELU(),

            )

    def forward(self,x):
        for i in range(self.t):

            if i==0:
                x1 = self.conv(x)
            
            x1 = self.conv(x+x1)
        return x1
        
class RRCNN_block(nn.Module):
    def __init__(self,ch_in,ch_out,t=2,norm='batch'):
        super(RRCNN_block,self).__init__()
        self.RCNN = nn.Sequential(
            Recurrent_block(ch_out,t=t,norm=norm),
            Recurrent_block(ch_out,t=t,norm=norm)
        )
        self.Conv_1x1 = nn.Conv2d(ch_in,ch_out,kernel_size=1,stride=1,padding=0)
        
    def forward(self,x):
        x = self.Conv_1x1(x)
        x1 = self.RCNN(x)
        return x+x1


class single_conv(nn.Module):
    def __init__(self,ch_in,ch_out,norm='batch'):
        super(single_conv,self).__init__()
        inplace = True
#         inplace = False
        if norm=='batch':
            self.conv = nn.Sequential(
                nn.Conv2d(ch_in, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
                nn.BatchNorm2d(ch_out),
#                 nn.ReLU(inplace=inplace)
                nn.GELU(),
            )
        elif norm=='instance':
            self.conv = nn.Sequential(
                nn.Conv2d(ch_in, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
                nn.InstanceNorm2d(ch_out),
#                 nn.ReLU(inplace=inplace)
                nn.GELU(),

            )
        elif norm=='group':
            self.conv = nn.Sequential(
                nn.Conv2d(ch_in, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
                nn.GroupNorm(int(ch_out/groupnorm_parameter),ch_out),
#                 nn.ReLU(inplace=inplace)
                nn.GELU(),

            )
        elif norm=='ws':
            self.conv = nn.Sequential(
                Conv2d(ch_in, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
                nn.GroupNorm(int(ch_out/groupnorm_parameter),ch_out),
#                 nn.ReLU(inplace=inplace)
                nn.GELU(),

            )

    def forward(self,x):
        x = self.conv(x)
        return x

class Attention_block(nn.Module):
    def __init__(self,F_g,F_l,F_int,norm='batch'):
        super(Attention_block,self).__init__()
        inplace= True
#         inplace= False
        
        if norm=='batch':
            self.W_g = nn.Sequential(
                nn.Conv2d(F_g, F_int, kernel_size=1,stride=1,padding=0,bias=True),
                nn.BatchNorm2d(F_int)
                )

            self.W_x = nn.Sequential(
                nn.Conv2d(F_l, F_int, kernel_size=1,stride=1,padding=0,bias=True),
                nn.BatchNorm2d(F_int)
            )

            self.psi = nn.Sequential(
                nn.Conv2d(F_int, 1, kernel_size=1,stride=1,padding=0,bias=True),
                nn.BatchNorm2d(1),
                nn.Sigmoid()
            )
        elif norm=='instance':
            self.W_g = nn.Sequential(
                nn.Conv2d(F_g, F_int, kernel_size=1,stride=1,padding=0,bias=True),
                nn.InstanceNorm2d(F_int)
                )

            self.W_x = nn.Sequential(
                nn.Conv2d(F_l, F_int, kernel_size=1,stride=1,padding=0,bias=True),
                nn.InstanceNorm2d(F_int)
            )

            self.psi = nn.Sequential(
                nn.Conv2d(F_int, 1, kernel_size=1,stride=1,padding=0,bias=True),
                nn.InstanceNorm2d(1),
                nn.Sigmoid()
            )
        elif norm=='group':
            self.W_g = nn.Sequential(
                nn.Conv2d(F_g, F_int, kernel_size=1,stride=1,padding=0,bias=True),
                nn.GroupNorm(int(F_int/groupnorm_parameter),F_int),
                )

            self.W_x = nn.Sequential(
                nn.Conv2d(F_l, F_int, kernel_size=1,stride=1,padding=0,bias=True),
                nn.GroupNorm(int(F_int/groupnorm_parameter),F_int),
            )

            self.psi = nn.Sequential(
                nn.Conv2d(F_int, 1, kernel_size=1,stride=1,padding=0,bias=True),
                nn.GroupNorm(1,1),
                nn.Sigmoid()
            )
        elif norm=='ws':
            self.W_g = nn.Sequential(
                Conv2d(F_g, F_int, kernel_size=1,stride=1,padding=0,bias=True),
                nn.GroupNorm(int(F_int/groupnorm_parameter),F_int),
                )

            self.W_x = nn.Sequential(
                Conv2d(F_l, F_int, kernel_size=1,stride=1,padding=0,bias=True),
                nn.GroupNorm(int(F_int/groupnorm_parameter),F_int),
            )

            self.psi = nn.Sequential(
                Conv2d(F_int, 1, kernel_size=1,stride=1,padding=0,bias=True),
                nn.GroupNorm(1,1),
                nn.Sigmoid()
            )
        self.relu = nn.ReLU(inplace=inplace)

        
    def forward(self,g,x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1+x1)
        psi = self.psi(psi)

        return x*psi


class MCDropout(nn.Dropout):
    """
    Drops elements of input variable randomly.
    This module drops input elements randomly with probability ``p`` and
    scales the remaining elements by factor ``1 / (1 - p)``.
    Args:
        p: probability of an element to be zeroed. Default: 0.5
        inplace: If set to ``True``, will do this operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(*)`. Input can be of any shape
        - Output: :math:`(*)`. Output is of the same shape as input

    Examples::

        >>> m = MCDropout(p=0.2)
        >>> input = torch.randn(20, 16)
        >>> output = m(input)

    See the paper by Y. Gal, and G. Zoubin: `Dropout as a bayesian approximation: \
    Representing model uncertainty in deep learning .\
    <https://arxiv.org/abs/1506.02142>`

    See also: A. Kendall: `Bayesian SegNet: Model Uncertainty \
    in Deep Convolutional Encoder-Decoder Architectures for Scene Understanding \
    <https://arxiv.org/abs/1511.02680>`_.
    """

    def forward(self, input):
        self.inplace
        return F.dropout(input, self.p, True, self.inplace)


class U_Net(nn.Module):
    def __init__(self,img_ch=3,output_ch=1):
        super(U_Net,self).__init__()
        
        self.Maxpool = nn.MaxPool2d(kernel_size=2,stride=2)

        self.Conv1 = conv_block(ch_in=img_ch,ch_out=64)
        self.Conv2 = conv_block(ch_in=64,ch_out=128)
        self.Conv3 = conv_block(ch_in=128,ch_out=256)
        self.Conv4 = conv_block(ch_in=256,ch_out=512)
        self.Conv5 = conv_block(ch_in=512,ch_out=1024)

        self.Up5 = up_conv(ch_in=1024,ch_out=512)
        self.Up_conv5 = conv_block(ch_in=1024, ch_out=512)

        self.Up4 = up_conv(ch_in=512,ch_out=256)
        self.Up_conv4 = conv_block(ch_in=512, ch_out=256)
        
        self.Up3 = up_conv(ch_in=256,ch_out=128)
        self.Up_conv3 = conv_block(ch_in=256, ch_out=128)
        
        self.Up2 = up_conv(ch_in=128,ch_out=64)
        self.Up_conv2 = conv_block(ch_in=128, ch_out=64)

        self.Conv_1x1 = nn.Conv2d(64,output_ch,kernel_size=1,stride=1,padding=0)


    def forward(self,x):
        # encoding path
        x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)
        
        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        d5 = torch.cat((x4,d5),dim=1)
        
        d5 = self.Up_conv5(d5)
        
        d4 = self.Up4(d5)
        d4 = torch.cat((x3,d4),dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2,d3),dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1,d2),dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)

        return d1


class R2U_Net(nn.Module):
    def __init__(self,img_ch=3,output_ch=2,t=2,norm='batch'):
        super(R2U_Net,self).__init__()
        
        self.Maxpool = nn.MaxPool2d(kernel_size=2,stride=2)
        self.Upsample = nn.Upsample(scale_factor=2)

        self.RRCNN1 = RRCNN_block(ch_in=img_ch,ch_out=64,t=t,norm=norm)
        self.RRCNN2 = RRCNN_block(ch_in=64,ch_out=128,t=t,norm=norm)
        self.RRCNN3 = RRCNN_block(ch_in=128,ch_out=256,t=t,norm=norm)        
        self.RRCNN4 = RRCNN_block(ch_in=256,ch_out=512,t=t,norm=norm)        
        self.RRCNN5 = RRCNN_block(ch_in=512,ch_out=1024,t=t,norm=norm)
        
        self.Up5 = up_conv(ch_in=1024,ch_out=512,norm=norm)
        self.Up_RRCNN5 = RRCNN_block(ch_in=1024, ch_out=512,t=t,norm=norm)
        
        self.Up4 = up_conv(ch_in=512,ch_out=256,norm=norm)
        self.Up_RRCNN4 = RRCNN_block(ch_in=512, ch_out=256,t=t,norm=norm)
        
        self.Up3 = up_conv(ch_in=256,ch_out=128,norm=norm)
        self.Up_RRCNN3 = RRCNN_block(ch_in=256, ch_out=128,t=t,norm=norm)
        
        self.Up2 = up_conv(ch_in=128,ch_out=64,norm=norm)
        self.Up_RRCNN2 = RRCNN_block(ch_in=128, ch_out=64,t=t,norm=norm)
        
        self.Conv_1x1 = nn.Conv2d(64,output_ch,kernel_size=1,stride=1,padding=0)


    def forward(self,x):
        # encoding path
        x1 = self.RRCNN1(x)

        x2 = self.Maxpool(x1)
        x2 = self.RRCNN2(x2)
        
        x3 = self.Maxpool(x2)
        x3 = self.RRCNN3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.RRCNN4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.RRCNN5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        d5 = torch.cat((x4,d5),dim=1)
        d5 = self.Up_RRCNN5(d5)
        
        d4 = self.Up4(d5)
        d4 = torch.cat((x3,d4),dim=1)
        d4 = self.Up_RRCNN4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2,d3),dim=1)
        d3 = self.Up_RRCNN3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1,d2),dim=1)
        d2 = self.Up_RRCNN2(d2)

        d1 = self.Conv_1x1(d2)

        return d1

class AttU_Net(nn.Module):
    def __init__(self,img_ch=3,output_ch=2,norm='batch',mc_dropout=0.0):
        super(AttU_Net,self).__init__()
        
        self.Maxpool = nn.MaxPool2d(kernel_size=2,stride=2)

        self.Conv1 = conv_block(ch_in=img_ch,ch_out=64,norm=norm)
        self.Conv2 = conv_block(ch_in=64,ch_out=128,norm=norm)
        self.Conv3 = conv_block(ch_in=128,ch_out=256,norm=norm)
        self.Conv4 = conv_block(ch_in=256,ch_out=512,norm=norm)
        self.Conv5 = conv_block(ch_in=512,ch_out=1024,norm=norm)

        self.Up5 = up_conv(ch_in=1024,ch_out=512,norm=norm)
        self.Att5 = Attention_block(F_g=512,F_l=512,F_int=256,norm=norm)
        self.Up_conv5 = conv_block(ch_in=1024, ch_out=512,norm=norm)

        self.Up4 = up_conv(ch_in=512,ch_out=256,norm=norm)
        self.Att4 = Attention_block(F_g=256,F_l=256,F_int=128,norm=norm)
        self.Up_conv4 = conv_block(ch_in=512, ch_out=256,norm=norm)
        
        self.Up3 = up_conv(ch_in=256,ch_out=128,norm=norm)
        self.Att3 = Attention_block(F_g=128,F_l=128,F_int=64,norm=norm)
        self.Up_conv3 = conv_block(ch_in=256, ch_out=128,norm=norm)
        
        self.Up2 = up_conv(ch_in=128,ch_out=64,norm=norm)
        self.Att2 = Attention_block(F_g=64,F_l=64,F_int=32,norm=norm)
        self.Up_conv2 = conv_block(ch_in=128, ch_out=64,norm=norm)

        self.Conv_1x1 = nn.Conv2d(64,output_ch,kernel_size=1,stride=1,padding=0)
#         self.MCDropout = MCDropout(p=mc_dropout)
        
#         self.nnblock512 = NONLocalBlock2D(512)
#         self.nnblock1024 = NONLocalBlock2D(1024)
        
    def forward(self,x):
        # encoding path
        x1 = self.Conv1(x)
#         x1 = self.MCDropout(x1)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)
#         x2 = self.MCDropout(x2)
        
        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)
#         x3 = self.MCDropout(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)
#         x4 = self.nnblock512(x4) 
#         x4 = self.MCDropout(x4)

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)
#         x5 = self.nnblock1024(x5) 
#         x5 = self.MCDropout(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        x4 = self.Att5(g=d5,x=x4)
        d5 = torch.cat((x4,d5),dim=1)        
        d5 = self.Up_conv5(d5)
#         d5 = self.nnblock512(d5) 
#         d5 = self.MCDropout(d5)
        
        d4 = self.Up4(d5)
        x3 = self.Att4(g=d4,x=x3)
        d4 = torch.cat((x3,d4),dim=1)
        d4 = self.Up_conv4(d4)
#         d4 = self.MCDropout(d4)
        
        d3 = self.Up3(d4)
        x2 = self.Att3(g=d3,x=x2)
        d3 = torch.cat((x2,d3),dim=1)
        d3 = self.Up_conv3(d3)
#         d3 = self.MCDropout(d3)
        
        d2 = self.Up2(d3)
        x1 = self.Att2(g=d2,x=x1)
        d2 = torch.cat((x1,d2),dim=1)
        d2 = self.Up_conv2(d2)
#         d2 = self.MCDropout(d2)

        d1 = self.Conv_1x1(d2)

        return d1

class SoftMTL_AttUNet(nn.Module):
    def __init__(self,img_ch=3,output_ch=2,recon_ch=3,norm = 'batch',mc_dropout=0.0):
        super(SoftMTL_AttUNet,self).__init__()
        
        self.Maxpool = nn.MaxPool2d(kernel_size=2,stride=2)

        self.Conv1 = conv_block(ch_in=img_ch,ch_out=64,norm=norm)
        self.Conv2 = conv_block(ch_in=64,ch_out=128,norm=norm)
        self.Conv3 = conv_block(ch_in=128,ch_out=256,norm=norm)
        self.Conv4 = conv_block(ch_in=256,ch_out=512,norm=norm)
        self.Conv5 = conv_block(ch_in=512,ch_out=1024,norm=norm)

        self.Up5 = up_conv(ch_in=1024,ch_out=512,norm=norm)
        self.Att5 = Attention_block(F_g=512,F_l=512,F_int=256,norm=norm)
        self.Up_conv5 = conv_block(ch_in=1024, ch_out=512,norm=norm)

        self.Up4 = up_conv(ch_in=512,ch_out=256,norm=norm)
        self.Att4 = Attention_block(F_g=256,F_l=256,F_int=128,norm=norm)
        self.Up_conv4 = conv_block(ch_in=512, ch_out=256,norm=norm)
        
        self.Up3 = up_conv(ch_in=256,ch_out=128,norm=norm)
        self.Att3 = Attention_block(F_g=128,F_l=128,F_int=64,norm=norm)
        self.Up_conv3 = conv_block(ch_in=256, ch_out=128,norm=norm)
        
        self.Up2 = up_conv(ch_in=128,ch_out=64,norm=norm)
        self.Att2 = Attention_block(F_g=64,F_l=64,F_int=32,norm=norm)
        self.Up_conv2 = conv_block(ch_in=128, ch_out=64,norm=norm)

        self.Conv_1x1 = nn.Conv2d(64,output_ch,kernel_size=1,stride=1,padding=0)
        self.Conv_1x1_r = nn.Conv2d(64,recon_ch,kernel_size=1,stride=1,padding=0)
        
        self.MCDropout = MCDropout(p=mc_dropout)
        
        self.nnblock512 = NONLocalBlock2D(512)
        self.nnblock1024 = NONLocalBlock2D(1024)
        
    def forward(self,x):
        # encoding path
        x1 = self.Conv1(x)
#         x1 = self.MCDropout(x1)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)
#         x2 = self.MCDropout(x2)
        
        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)
#         x3 = self.MCDropout(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)
#         x4 = self.nnblock512(x4) 
#         x4 = self.MCDropout(x4)

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)
#         x5 = self.nnblock1024(x5) 
        x5 = self.MCDropout(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        x4 = self.Att5(g=d5,x=x4)
        d5 = torch.cat((x4,d5),dim=1)        
        d5 = self.Up_conv5(d5)
#         d5 = self.nnblock512(d5) 
#         d5 = self.MCDropout(d5)
        
        d4 = self.Up4(d5)
        x3 = self.Att4(g=d4,x=x3)
        d4 = torch.cat((x3,d4),dim=1)
        d4 = self.Up_conv4(d4)
#         d4 = self.MCDropout(d4)
        
        d3 = self.Up3(d4)
        x2 = self.Att3(g=d3,x=x2)
        d3 = torch.cat((x2,d3),dim=1)
        d3 = self.Up_conv3(d3)
#         d3 = self.MCDropout(d3)
        
        d2 = self.Up2(d3)
        x1 = self.Att2(g=d2,x=x1)
        d2 = torch.cat((x1,d2),dim=1)
        d2 = self.Up_conv2(d2)
#         d2 = self.MCDropout(d2)

        d1 = self.Conv_1x1(d2)
    
        # decoding + concat path
        r5 = self.Up5(x5)
        x4 = self.Att5(g=r5,x=x4)
        r5 = torch.cat((x4,r5),dim=1)        
        r5 = self.Up_conv5(r5)
        r5 = self.MCDropout(r5)
        
        r4 = self.Up4(r5)
        x3 = self.Att4(g=r4,x=x3)
        r4 = torch.cat((x3,r4),dim=1)
        r4 = self.Up_conv4(r4)
#         r4 = self.MCDropout(r4)
        
        r3 = self.Up3(r4)
        x2 = self.Att3(g=r3,x=x2)
        r3 = torch.cat((x2,r3),dim=1)
        r3 = self.Up_conv3(r3)
#         r3 = self.MCDropout(r3)
        
        r2 = self.Up2(r3)
        x1 = self.Att2(g=r2,x=x1)
        r2 = torch.cat((x1,r2),dim=1)
        r2 = self.Up_conv2(r2)
#         r2 = self.MCDropout(r2)

        r1 = self.Conv_1x1_r(r2)

        return d1,r1

class HardMTL_AttUNet(nn.Module):
    def __init__(self,img_ch=3,output_ch=2,recon_ch=3,norm = 'batch'):
        super(HardMTL_AttUNet,self).__init__()
        
        self.Maxpool = nn.MaxPool2d(kernel_size=2,stride=2)
        
        self.Conv1 = conv_block(ch_in=img_ch,ch_out=64,norm=norm)
        self.Conv2 = conv_block(ch_in=64,ch_out=128,norm=norm)
        self.Conv3 = conv_block(ch_in=128,ch_out=256,norm=norm)
        self.Conv4 = conv_block(ch_in=256,ch_out=512,norm=norm)
        self.Conv5 = conv_block(ch_in=512,ch_out=1024,norm=norm)

        self.Up5 = up_conv(ch_in=1024,ch_out=512,norm=norm)
        self.Att5 = Attention_block(F_g=512,F_l=512,F_int=256,norm=norm)
        self.Up_conv5 = conv_block(ch_in=1024, ch_out=512,norm=norm)

        self.Up4 = up_conv(ch_in=512,ch_out=256,norm=norm)
        self.Att4 = Attention_block(F_g=256,F_l=256,F_int=128,norm=norm)
        self.Up_conv4 = conv_block(ch_in=512, ch_out=256,norm=norm)
        
        self.Up3 = up_conv(ch_in=256,ch_out=128,norm=norm)
        self.Att3 = Attention_block(F_g=128,F_l=128,F_int=64,norm=norm)
        self.Up_conv3 = conv_block(ch_in=256, ch_out=128,norm=norm)
        
        self.Up2 = up_conv(ch_in=128,ch_out=64,norm=norm)
        self.Att2 = Attention_block(F_g=64,F_l=64,F_int=32,norm=norm)
        self.Up_conv2 = conv_block(ch_in=128, ch_out=64,norm=norm)

        self.Conv_1x1 = nn.Conv2d(64,output_ch,kernel_size=1,stride=1,padding=0)
        self.Conv_1x1_r = nn.Conv2d(64,recon_ch,kernel_size=1,stride=1,padding=0)
        
        self.MCDropout = MCDropout(p=0.5)
        
        self.nnblock512 = NONLocalBlock2D(512)
        self.nnblock1024 = NONLocalBlock2D(1024)
        
    def forward(self,x):
        # encoding path
        x1 = self.Conv1(x)
        x1 = self.MCDropout(x1)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)
        x2 = self.MCDropout(x2)
        
        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)
        x3 = self.MCDropout(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)
#         x4 = self.nnblock512(x4) 
        x4 = self.MCDropout(x4)

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)
#         x5 = self.nnblock1024(x5) 
        x5 = self.MCDropout(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        x4 = self.Att5(g=d5,x=x4)
        d5 = torch.cat((x4,d5),dim=1)        
        d5 = self.Up_conv5(d5)
#         d5 = self.nnblock512(d5) 
        d5 = self.MCDropout(d5)
        
        d4 = self.Up4(d5)
        x3 = self.Att4(g=d4,x=x3)
        d4 = torch.cat((x3,d4),dim=1)
        d4 = self.Up_conv4(d4)
        d4 = self.MCDropout(d4)
        
        d3 = self.Up3(d4)
        x2 = self.Att3(g=d3,x=x2)
        d3 = torch.cat((x2,d3),dim=1)
        d3 = self.Up_conv3(d3)
        d3 = self.MCDropout(d3)
        
        d2 = self.Up2(d3)
        x1 = self.Att2(g=d2,x=x1)
        d2 = torch.cat((x1,d2),dim=1)
        d2 = self.Up_conv2(d2)
        d2 = self.MCDropout(d2)

        d1 = self.Conv_1x1(d2)
        
        r2 = self.Up2(d3)
        x1 = self.Att2(g=r2,x=x1)
        r2 = torch.cat((x1,r2),dim=1)
        r2 = self.Up_conv2(r2)
        r2 = self.MCDropout(r2)

        r1 = self.Conv_1x1_r(r2)

        return d1,r1

class R2AttU_Net(nn.Module):
    def __init__(self,img_ch=3,output_ch=2,t=2,norm='batch',mc_dropout=0.0,nnblock=True):
        super(R2AttU_Net,self).__init__()
        
        self.Maxpool = nn.MaxPool2d(kernel_size=2,stride=2)
        self.Upsample = nn.Upsample(scale_factor=2)

        self.RRCNN1 = RRCNN_block(ch_in=img_ch,ch_out=64,t=t,norm=norm)
        self.RRCNN2 = RRCNN_block(ch_in=64,ch_out=128,t=t,norm=norm)        
        self.RRCNN3 = RRCNN_block(ch_in=128,ch_out=256,t=t,norm=norm)        
        self.RRCNN4 = RRCNN_block(ch_in=256,ch_out=512,t=t,norm=norm)        
        self.RRCNN5 = RRCNN_block(ch_in=512,ch_out=1024,t=t,norm=norm)        

        self.Up5 = up_conv(ch_in=1024,ch_out=512,norm=norm)
        self.Att5 = Attention_block(F_g=512,F_l=512,F_int=256,norm=norm)
        self.Up_RRCNN5 = RRCNN_block(ch_in=1024, ch_out=512,t=t,norm=norm)
        
        self.Up4 = up_conv(ch_in=512,ch_out=256,norm=norm)
        self.Att4 = Attention_block(F_g=256,F_l=256,F_int=128,norm=norm)
        self.Up_RRCNN4 = RRCNN_block(ch_in=512, ch_out=256,t=t,norm=norm)
        
        self.Up3 = up_conv(ch_in=256,ch_out=128,norm=norm)
        self.Att3 = Attention_block(F_g=128,F_l=128,F_int=64,norm=norm)
        self.Up_RRCNN3 = RRCNN_block(ch_in=256, ch_out=128,t=t,norm=norm)
        
        self.Up2 = up_conv(ch_in=128,ch_out=64,norm=norm)
        self.Att2 = Attention_block(F_g=64,F_l=64,F_int=32,norm=norm)
        self.Up_RRCNN2 = RRCNN_block(ch_in=128, ch_out=64,t=t,norm=norm)
        
        if norm=='ws':
            self.Conv_1x1 = Conv2d(64,output_ch,kernel_size=1,stride=1,padding=0)         
        else:
            self.Conv_1x1 = nn.Conv2d(64,output_ch,kernel_size=1,stride=1,padding=0)  
        self.MCDropout = MCDropout(p=mc_dropout)     
        self.nnblock = nnblock
        self.nnblock256 = NONLocalBlock2D(256)
        self.nnblock512 = NONLocalBlock2D(512)
        self.nnblock1024 = NONLocalBlock2D(1024)

    def forward(self,x):
        # encoding path
        x1 = self.RRCNN1(x)
#         x1 = self.MCDropout(x1)

        x2 = self.Maxpool(x1)
        x2 = self.RRCNN2(x2)
#         x2 = self.MCDropout(x2)
        
        x3 = self.Maxpool(x2)
        x3 = self.RRCNN3(x3)
#         if self.nnblock == True:
#             x3 = self.nnblock256(x3) 
#         x3 = self.MCDropout(x3)
        
        x4 = self.Maxpool(x3)
        x4 = self.RRCNN4(x4)
        if self.nnblock == True:
            x4 = self.nnblock512(x4) 
#         x4 = self.MCDropout(x4)

        x5 = self.Maxpool(x4)
        x5 = self.RRCNN5(x5)
        if self.nnblock == True:
            x5 = self.nnblock1024(x5) 
        x5 = self.MCDropout(x5)
        
        # decoding + concat path
        d5 = self.Up5(x5)
        x4 = self.Att5(g=d5,x=x4)
        d5 = torch.cat((x4,d5),dim=1)
        d5 = self.Up_RRCNN5(d5)      
        if self.nnblock == True:
            d5 = self.nnblock512(d5) 
#         d5 = self.MCDropout(d5)
        
        d4 = self.Up4(d5)
        x3 = self.Att4(g=d4,x=x3)
        d4 = torch.cat((x3,d4),dim=1)
        d4 = self.Up_RRCNN4(d4)        
#         if self.nnblock == True:
#             d4 = self.nnblock256(d4)
#         d4 = self.MCDropout(d4)

        d3 = self.Up3(d4)
        x2 = self.Att3(g=d3,x=x2)
        d3 = torch.cat((x2,d3),dim=1)
        d3 = self.Up_RRCNN3(d3)
#         d3 = self.MCDropout(d3)

        d2 = self.Up2(d3)
        x1 = self.Att2(g=d2,x=x1)
        d2 = torch.cat((x1,d2),dim=1)
        d2 = self.Up_RRCNN2(d2)
#         d2 = self.MCDropout(d2)

        d1 = self.Conv_1x1(d2)

        return d1

# NNBlock
import torch
from torch import nn
from torch.nn import functional as F

class _NonLocalBlockND(nn.Module):
    def __init__(self, in_channels, inter_channels=None, dimension=3, sub_sample=True, bn_layer=True):
        super(_NonLocalBlockND, self).__init__()

        assert dimension in [1, 2, 3]

        self.dimension = dimension
        self.sub_sample = sub_sample

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        if dimension == 3:
            conv_nd = nn.Conv3d
            max_pool_layer = nn.MaxPool3d(kernel_size=(1, 2, 2))
#             bn = nn.BatchNorm3d
            bn = nn.GroupNorm
        elif dimension == 2:
#             conv_nd = nn.Conv2d
            conv_nd = Conv2d
            max_pool_layer = nn.MaxPool2d(kernel_size=(2, 2))
#             bn = nn.BatchNorm2d
            bn = nn.GroupNorm
        else:
            conv_nd = nn.Conv1d
            max_pool_layer = nn.MaxPool1d(kernel_size=(2))
#             bn = nn.BatchNorm1d
            bn = nn.GroupNorm

        self.g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                         kernel_size=1, stride=1, padding=0)

        if bn_layer:
            self.W = nn.Sequential(
                conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                        kernel_size=1, stride=1, padding=0),
#                 bn(self.in_channels)
                bn(int(self.in_channels/16),self.in_channels)
            )
            nn.init.constant_(self.W[1].weight, 0)
            nn.init.constant_(self.W[1].bias, 0)
        else:
            self.W = conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                             kernel_size=1, stride=1, padding=0)
            nn.init.constant_(self.W.weight, 0)
            nn.init.constant_(self.W.bias, 0)

        self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0)

        self.phi = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0)

        if sub_sample:
            self.g = nn.Sequential(self.g, max_pool_layer)
            self.phi = nn.Sequential(self.phi, max_pool_layer)

    def forward(self, x):
        '''
        :param x: (b, c, t, h, w)
        :return:
        '''

        batch_size = x.size(0)

        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
        f = torch.matmul(theta_x, phi_x)
        N = f.size(-1)
        f_div_C = f / N

        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x

        return z

class NONLocalBlock1D(_NonLocalBlockND):
    def __init__(self, in_channels, inter_channels=None, sub_sample=True, bn_layer=True):
        super(NONLocalBlock1D, self).__init__(in_channels,
                                              inter_channels=inter_channels,
                                              dimension=1, sub_sample=sub_sample,
                                              bn_layer=bn_layer)


class NONLocalBlock2D(_NonLocalBlockND):
    def __init__(self, in_channels, inter_channels=None, sub_sample=True, bn_layer=True):
        super(NONLocalBlock2D, self).__init__(in_channels,
                                              inter_channels=inter_channels,
                                              dimension=2, sub_sample=sub_sample,
                                              bn_layer=bn_layer)


class NONLocalBlock3D(_NonLocalBlockND):
    def __init__(self, in_channels, inter_channels=None, sub_sample=True, bn_layer=True):
        super(NONLocalBlock3D, self).__init__(in_channels,
                                              inter_channels=inter_channels,
                                              dimension=3, sub_sample=sub_sample,
                                              bn_layer=bn_layer)

class DiscriminateNet(nn.Module):
    def __init__(self, n_class=2):
        super(StanfordBNet, self).__init__()

        self.conv1_1 = nn.Conv2d(n_class, 16, 5, stride=1, padding=1)

        self.conv2_1 = nn.Conv2d(3, 8, 5, stride=1, padding=1)
        self.conv2_2 = nn.Conv2d(8, 16, 5, stride=1, padding=1)

        self.conv3_1 = nn.Conv2d(32, 64, 3, stride=1, padding=1)
        self.conv3_2 = nn.Conv2d(64, 128, 3, stride=1, padding=1)
        self.conv3_3 = nn.Conv2d(128, 256, 3, stride=1, padding=1)
        self.conv3_4 = nn.Conv2d(256, 2, 3, stride=1, padding=1)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, segmented_input, original_input):
        res1 = self.relu(self.conv1_1(segmented_input))

        res2 = self.relu(self.conv2_1(original_input))
        res2 = F.max_pool2d(res2, 2, stride=1, padding=1)
        res2 = self.relu(self.conv2_2(res2))
        res2 = F.max_pool2d(res2, 2, stride=1, padding=1)

        res3 = torch.cat((res1, res2), 1)

        res3 = self.relu(self.conv3_1(res3))
        res3 = F.max_pool2d(res3, 2, stride=1)
        res3 = self.relu(self.conv3_2(res3))
        res3 = F.max_pool2d(res3, 2, stride=1)
        res3 = self.relu(self.conv3_3(res3))
        res3 = self.conv3_4(res3)
        
        # return res
        out = F.avg_pool2d(res3, (res3.shape[2],res3.shape[3]))
        out = F.softmax(out)
        n , _ , _ ,_  = segmented_input.size()
        return out.view(n,-1).transpose(0,1)[0]
    

    
import pywt
import torch
from torch.autograd import Variable

w=pywt.Wavelet('db1')

dec_hi = torch.Tensor(w.dec_hi[::-1]) 
dec_lo = torch.Tensor(w.dec_lo[::-1])
rec_hi = torch.Tensor(w.rec_hi)
rec_lo = torch.Tensor(w.rec_lo)

filters = torch.stack([dec_lo.unsqueeze(0)*dec_lo.unsqueeze(1)/2.0,
                       dec_lo.unsqueeze(0)*dec_hi.unsqueeze(1),
                       dec_hi.unsqueeze(0)*dec_lo.unsqueeze(1),
                       dec_hi.unsqueeze(0)*dec_hi.unsqueeze(1)], dim=0)

inv_filters = torch.stack([rec_lo.unsqueeze(0)*rec_lo.unsqueeze(1)*2.0,
                           rec_lo.unsqueeze(0)*rec_hi.unsqueeze(1),
                           rec_hi.unsqueeze(0)*rec_lo.unsqueeze(1),
                           rec_hi.unsqueeze(0)*rec_hi.unsqueeze(1)], dim=0)

def wt(vimg):
    padded = vimg
    res = torch.zeros(vimg.shape[0],4*vimg.shape[1],int(vimg.shape[2]/2),int(vimg.shape[3]/2))
    res = res.cuda()
    for i in range(padded.shape[1]):
        res[:,4*i:4*i+4] = torch.nn.functional.conv2d(padded[:,i:i+1], Variable(filters[:,None].cuda(),requires_grad=True),stride=2)
        res[:,4*i+1:4*i+4] = (res[:,4*i+1:4*i+4]+1)/2.0

    return res

def iwt(vres):
    res = torch.zeros(vres.shape[0],int(vres.shape[1]/4),int(vres.shape[2]*2),int(vres.shape[3]*2))
    res = res.cuda()
    for i in range(res.shape[1]):
        vres[:,4*i+1:4*i+4]=2*vres[:,4*i+1:4*i+4]-1
        temp = torch.nn.functional.conv_transpose2d(vres[:,4*i:4*i+4], Variable(inv_filters[:,None].cuda(),requires_grad=True),stride=2)
        res[:,i:i+1,:,:] = temp
    return res

    
# from wavelet import wt,iwt
class Waveletnet(nn.Module):
    def __init__(self,in_channel=3,out_channel=2, c=16,ws=True,):
        super(Waveletnet, self).__init__()
        self.num=1
        self.gn1 = nn.GroupNorm(int(c/groupnorm_parameter),c)
        self.gn2 = nn.GroupNorm(int(4*c/groupnorm_parameter),4*c)
        self.gn3 = nn.GroupNorm(int(16*c/groupnorm_parameter),16*c)
        self.gn4 = nn.GroupNorm(int(64*c/groupnorm_parameter),64*c)

        if ws == False:
            self.conv1 = nn.Conv2d(4*in_channel,c,3, 1,padding=1)
            self.conv2 = nn.Conv2d(4*c,4*c,3, 1,padding=1)        
            self.conv3 = nn.Conv2d(16*c,16*c,3, 1,padding=1)  
            self.conv4 = nn.Conv2d(64*c,64*c,3, 1,padding=1)
            
            self.convd1 = nn.Conv2d(c,12,3, 1,padding=1)
            self.convd2 = nn.Conv2d(2*c,c,3, 1,padding=1) 
            self.convd3 = nn.Conv2d(8*c,4*c,3, 1,padding=1)        
            self.convd4 = nn.Conv2d(32*c,16*c,3, 1,padding=1)  
        else:
            self.conv1 = Conv2d(4*in_channel,c,3, 1,padding=1)
            self.conv2 = Conv2d(4*c,4*c,3, 1,padding=1)        
            self.conv3 = Conv2d(16*c,16*c,3, 1,padding=1)  
            self.conv4 = Conv2d(64*c,64*c,3, 1,padding=1)
            
            self.convd1 = Conv2d(c,4*in_channel,3, 1,padding=1)
            self.convd2 = Conv2d(2*c,c,3, 1,padding=1) 
            self.convd3 = Conv2d(8*c,4*c,3, 1,padding=1)        
            self.convd4 = Conv2d(32*c,16*c,3, 1,padding=1)  
            
        self.relu = nn.LeakyReLU(0.2)
        self.final = Conv2d(in_channel,out_channel,1,padding=0, bias=False)

    def forward(self, x):
        
        w1 = wt(x)
        c1 = self.relu(self.conv1(w1))
        c1 = self.gn1(c1)
                
        w2 = wt(c1)
        c2 = self.relu(self.conv2(w2))
        c2 = self.gn2(c2)
        
        w3 = wt(c2)
        c3 = self.relu(self.conv3(w3))
        c3 = self.gn3(c3)
        
        w4 = wt(c3)
        c4 = self.relu(self.conv4(w4))
        c4 = self.gn4(c4)
        c5 = self.relu(self.conv4(c4))
        c5 = self.gn4(c5)
        c6 = (self.conv4(c5))
        ic4 = self.relu(c6+w4)

        iw4 = iwt(ic4)
        iw4 = torch.cat([c3,iw4],1)
        ic3 = self.relu(self.convd4(iw4))
        ic3 = self.gn3(ic3)
        
        iw3 = iwt(ic3)
        iw3 = torch.cat([c2,iw3],1)
        ic2 = self.relu(self.convd3(iw3))
        ic2 = self.gn2(ic2)
        
        iw2 = iwt(ic2)
        iw2 = torch.cat([c1,iw2],1)
        ic1 = self.relu(self.convd2(iw2))
        ic1 = self.gn1(ic1)

        iw1 = self.relu(self.convd1(ic1))
        
        y = iwt(iw1)
        y = self.final(y)
        return y

groupnorm_parameter = 8

class TightWaveletnet(nn.Module):
    def __init__(self,in_channel=3,out_channel=2, c=32, norm='ws',mc_dropout=0.0, nnblock=False):
        super(TightWaveletnet, self).__init__()
        self.num=1
        
#         def split_2x2(inputs):
#             inputs_shape = inputs.shape
#             inputs_ul = inputs[:,:,int(inputs_shape[2]/2),int(inputs_shape[3]/2)]
#             inputs_ur = inputs[:,:,int(inputs_shape[2]/2),:int(inputs_shape[3]/2)]
#             inputs_ll = inputs[:,:,:int(inputs_shape[2]/2),int(inputs_shape[3]/2)]
#             inputs_lr = inputs[:,:,:int(inputs_shape[2]/2),:int(inputs_shape[3]/2)]
#             return inputs_ul,inputs_ur,inputs_ll,inputs_lr

        self.gn1 = nn.GroupNorm(int(c/groupnorm_parameter),c)
        self.gn2 = nn.GroupNorm(int(4*c/groupnorm_parameter),4*c)
        self.gn3 = nn.GroupNorm(int(16*c/groupnorm_parameter),16*c)
        self.gn4 = nn.GroupNorm(int(64*c/groupnorm_parameter),64*c)
              
#         self.Conv1 = conv_block(ch_in=img_ch,ch_out=64,norm=norm)
#         self.Conv2 = conv_block(ch_in=64,ch_out=128,norm=norm)
#         self.Conv3 = conv_block(ch_in=128,ch_out=256,norm=norm)
#         self.Conv4 = conv_block(ch_in=256,ch_out=512,norm=norm)
#         self.Conv5 = conv_block(ch_in=512,ch_out=1024,norm=norm)

#         self.Up5 = up_conv(ch_in=1024,ch_out=512,norm=norm)
#         self.Att5 = Attention_block(F_g=512,F_l=512,F_int=256,norm=norm)
#         self.Up_conv5 = conv_block(ch_in=1024, ch_out=512,norm=norm)

#         self.Up4 = up_conv(ch_in=512,ch_out=256,norm=norm)
#         self.Att4 = Attention_block(F_g=256,F_l=256,F_int=128,norm=norm)
#         self.Up_conv4 = conv_block(ch_in=512, ch_out=256,norm=norm)
        
#         self.Up3 = up_conv(ch_in=256,ch_out=128,norm=norm)
#         self.Att3 = Attention_block(F_g=128,F_l=128,F_int=64,norm=norm)
#         self.Up_conv3 = conv_block(ch_in=256, ch_out=128,norm=norm)
        
#         self.Up2 = up_conv(ch_in=128,ch_out=64,norm=norm)
#         self.Att2 = Attention_block(F_g=64,F_l=64,F_int=32,norm=norm)
#         self.Up_conv2 = conv_block(ch_in=128, ch_out=64,norm=norm)

        self.conv0 = conv_block(ch_in=in_channel,ch_out=int(c/4),norm=norm) # x,c
        self.conv1 = conv_block(ch_in=c,ch_out=c,norm=norm)                 # x/2, c
        self.conv2 = conv_block(ch_in=4*c,ch_out=4*c,norm=norm)             # x/4, 4*c
        self.conv3 = conv_block(ch_in=16*c,ch_out=16*c,norm=norm)           # x/8, 16*c
        self.conv4 = conv_block(ch_in=64*c,ch_out=64*c,norm=norm)           # x/16,64*c

        self.Att4 = Attention_block(F_g=16*c,F_l=16*c,F_int=4*c,norm=norm)
        self.convd4 = conv_block(ch_in=32*c,ch_out=16*c,norm=norm)          # x/16,c
        self.Att3 = Attention_block(F_g=4*c,F_l=4*c,F_int=c,norm=norm)
        self.convd3 = conv_block(ch_in=8*c,ch_out=4*c,norm=norm)            # x/8, c
        self.Att2 = Attention_block(F_g=c,F_l=c,F_int=c,norm=norm)
        self.convd2 = conv_block(ch_in=2*c,ch_out=c,norm=norm)              # x/4, c
        self.convd1 = conv_block(ch_in=c,ch_out=4*in_channel,norm=norm)     # x/2, c
        
        self.MCDropout = MCDropout(p=mc_dropout)
        self.nnblock = nnblock
        if nnblock == True:
            self.nnblock_bottleneck = NONLocalBlock2D(64*c)

        self.relu = nn.LeakyReLU(0.2)
        self.final = Conv2d(in_channel,out_channel,1,padding=0, bias=False)

    def forward(self, x):
        # x 
        x = self.conv0(x)
        w1 = wt(x) # x/2
#         w1_ul,w1_ur,w1_ll,w1_lr = split_2x2(w1)
        
        c1 = self.relu(self.conv1(w1))
        c1 = self.gn1(c1)
                
        w2 = wt(c1) # x/4
#         print('w2',w2.shape)
        c2 = self.relu(self.conv2(w2))
        c2 = self.gn2(c2)
        
        w3 = wt(c2) # x/8
#         print('w3',w3.shape)
        c3 = self.relu(self.conv3(w3))
        c3 = self.gn3(c3)
        
        w4 = wt(c3) # x/16
        if self.nnblock == True:
            w4 = self.nnblock_bottleneck(w4)
        w4 = self.MCDropout(w4)
        c4 = self.relu(self.conv4(w4))
        c4 = self.gn4(c4)
        c5 = self.relu(self.conv4(c4))
        c5 = self.gn4(c5)
        c6 = (self.conv4(c5))
        ic4 = self.relu(c6+w4)
                
#         d5 = self.Up5(x5)
#         x4 = self.Att5(g=d5,x=x4)
#         d5 = torch.cat((x4,d5),dim=1)        
#         d5 = self.Up_conv5(d5)

        iw4 = iwt(ic4) # x/8
        c3 = self.Att4(iw4,c3)
        iw4 = torch.cat([c3,iw4],1)
        ic3 = self.relu(self.convd4(iw4))
        ic3 = self.gn3(ic3)
        
        iw3 = iwt(ic3) # x/4
        c2 = self.Att3(iw3,c2)
        iw3 = torch.cat([c2,iw3],1)
        ic2 = self.relu(self.convd3(iw3))
        ic2 = self.gn2(ic2)
        
        iw2 = iwt(ic2) # x/2
        c1 = self.Att2(iw2,c1)
        iw2 = torch.cat([c1,iw2],1)
        ic1 = self.relu(self.convd2(iw2))
        ic1 = self.gn1(ic1)

        iw1 = self.relu(self.convd1(ic1))

        y = iwt(iw1) # x
        y = self.final(y)
        return y


    
    
# import math
# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# __all__ = ['axial26s', 'axial50s', 'axial50m', 'axial50l']


# def conv1x1(in_planes, out_planes, stride=1):
#     """1x1 convolution"""
#     return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


# class AxialAttention(nn.Module):
#     def __init__(self, in_planes, out_planes, groups=8, kernel_size=56,
#                  stride=1, bias=False, width=False):
#         assert (in_planes % groups == 0) and (out_planes % groups == 0)
#         super(AxialAttention, self).__init__()
#         self.in_planes = in_planes
#         self.out_planes = out_planes
#         self.groups = groups
#         self.group_planes = out_planes // groups
#         self.kernel_size = kernel_size
#         self.stride = stride
#         self.bias = bias
#         self.width = width

#         # Multi-head self attention
#         self.qkv_transform = qkv_transform(in_planes, out_planes * 2, kernel_size=1, stride=1,
#                                            padding=0, bias=False)
#         self.bn_qkv = nn.BatchNorm1d(out_planes * 2)
#         self.bn_similarity = nn.BatchNorm2d(groups * 3)
#         #self.bn_qk = nn.BatchNorm2d(groups)
#         #self.bn_qr = nn.BatchNorm2d(groups)
#         #self.bn_kr = nn.BatchNorm2d(groups)
#         self.bn_output = nn.BatchNorm1d(out_planes * 2)

#         # Position embedding
#         self.relative = nn.Parameter(torch.randn(self.group_planes * 2, kernel_size * 2 - 1), requires_grad=True)
#         query_index = torch.arange(kernel_size).unsqueeze(0)
#         key_index = torch.arange(kernel_size).unsqueeze(1)
#         relative_index = key_index - query_index + kernel_size - 1
#         self.register_buffer('flatten_index', relative_index.view(-1))
#         if stride > 1:
#             self.pooling = nn.AvgPool2d(stride, stride=stride)

#         self.reset_parameters()

#     def forward(self, x):
#         if self.width:
#             x = x.permute(0, 2, 1, 3)
#         else:
#             x = x.permute(0, 3, 1, 2)  # N, W, C, H
#         N, W, C, H = x.shape
#         x = x.contiguous().view(N * W, C, H)

#         # Transformations
#         qkv = self.bn_qkv(self.qkv_transform(x))
#         q, k, v = torch.split(qkv.reshape(N * W, self.groups, self.group_planes * 2, H), [self.group_planes // 2, self.group_planes // 2, self.group_planes], dim=2)

#         # Calculate position embedding
#         all_embeddings = torch.index_select(self.relative, 1, self.flatten_index).view(self.group_planes * 2, self.kernel_size, self.kernel_size)
#         q_embedding, k_embedding, v_embedding = torch.split(all_embeddings, [self.group_planes // 2, self.group_planes // 2, self.group_planes], dim=0)
#         qr = torch.einsum('bgci,cij->bgij', q, q_embedding)
#         kr = torch.einsum('bgci,cij->bgij', k, k_embedding).transpose(2, 3)
#         qk = torch.einsum('bgci, bgcj->bgij', q, k)
#         stacked_similarity = torch.cat([qk, qr, kr], dim=1)
#         stacked_similarity = self.bn_similarity(stacked_similarity).view(N * W, 3, self.groups, H, H).sum(dim=1)
#         #stacked_similarity = self.bn_qr(qr) + self.bn_kr(kr) + self.bn_qk(qk)
#         # (N, groups, H, H, W)
#         similarity = F.softmax(stacked_similarity, dim=3)
#         sv = torch.einsum('bgij,bgcj->bgci', similarity, v)
#         sve = torch.einsum('bgij,cij->bgci', similarity, v_embedding)
#         stacked_output = torch.cat([sv, sve], dim=-1).view(N * W, self.out_planes * 2, H)
#         output = self.bn_output(stacked_output).view(N, W, self.out_planes, 2, H).sum(dim=-2)

#         if self.width:
#             output = output.permute(0, 2, 1, 3)
#         else:
#             output = output.permute(0, 2, 3, 1)

#         if self.stride > 1:
#             output = self.pooling(output)

#         return output

#     def reset_parameters(self):
#         self.qkv_transform.weight.data.normal_(0, math.sqrt(1. / self.in_planes))
#         #nn.init.uniform_(self.relative, -0.1, 0.1)
#         nn.init.normal_(self.relative, 0., math.sqrt(1. / self.group_planes))


# class AxialBlock(nn.Module):
#     expansion = 2

#     def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
#                  base_width=64, dilation=1, norm_layer=None, kernel_size=56):
#         super(AxialBlock, self).__init__()
#         if norm_layer is None:
#             norm_layer = nn.BatchNorm2d
#         width = int(planes * (base_width / 64.))
#         # Both self.conv2 and self.downsample layers downsample the input when stride != 1
#         self.conv_down = conv1x1(inplanes, width)
#         self.bn1 = norm_layer(width)
#         self.hight_block = AxialAttention(width, width, groups=groups, kernel_size=kernel_size)
#         self.width_block = AxialAttention(width, width, groups=groups, kernel_size=kernel_size, stride=stride, width=True)
#         self.conv_up = conv1x1(width, planes * self.expansion)
#         self.bn2 = norm_layer(planes * self.expansion)
#         self.relu = nn.ReLU(inplace=True)
#         self.downsample = downsample
#         self.stride = stride

#     def forward(self, x):
#         identity = x

#         out = self.conv_down(x)
#         out = self.bn1(out)
#         out = self.relu(out)

#         out = self.hight_block(out)
#         out = self.width_block(out)
#         out = self.relu(out)

#         out = self.conv_up(out)
#         out = self.bn2(out)

#         if self.downsample is not None:
#             identity = self.downsample(x)

#         out += identity
#         out = self.relu(out)

#         return out


# class AxialAttentionNet(nn.Module):

#     def __init__(self, block, layers, in_channel=1, num_classes=2, zero_init_residual=True,
#                  groups=8, width_per_group=64, replace_stride_with_dilation=None,
#                  norm_layer=None, s=0.5):
#         super(AxialAttentionNet, self).__init__()
#         if norm_layer is None:
#             norm_layer = nn.BatchNorm2d
#         self._norm_layer = norm_layer

#         self.inplanes = int(64 * s)
#         self.dilation = 1
#         if replace_stride_with_dilation is None:
#             # each element in the tuple indicates if we should replace
#             # the 2x2 stride with a dilated convolution instead
#             replace_stride_with_dilation = [False, False, False]
#         if len(replace_stride_with_dilation) != 3:
#             raise ValueError("replace_stride_with_dilation should be None "
#                              "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
#         self.groups = groups
#         self.base_width = width_per_group
#         self.conv1 = nn.Conv2d(in_channel, self.inplanes, kernel_size=7, stride=2, padding=3,
#                                bias=False)
#         self.bn1 = norm_layer(self.inplanes)
#         self.relu = nn.ReLU(inplace=True)
#         self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
#         self.layer1 = self._make_layer(block, int(128 * s), layers[0], kernel_size=56)
#         self.layer2 = self._make_layer(block, int(256 * s), layers[1], stride=2, kernel_size=56,
#                                        dilate=replace_stride_with_dilation[0])
#         self.layer3 = self._make_layer(block, int(512 * s), layers[2], stride=2, kernel_size=28,
#                                        dilate=replace_stride_with_dilation[1])
#         self.layer4 = self._make_layer(block, int(1024 * s), layers[3], stride=2, kernel_size=14,
#                                        dilate=replace_stride_with_dilation[2])
#         self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
#         self.fc = nn.Linear(int(1024 * block.expansion * s), num_classes)

#         for m in self.modules():
#             if isinstance(m, (nn.Conv2d, nn.Conv1d)):
#                 if isinstance(m, qkv_transform):
#                     pass
#                 else:
#                     nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#             elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d, nn.GroupNorm)):
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0)

#         # Zero-initialize the last BN in each residual branch,
#         # so that the residual branch starts with zeros, and each residual block behaves like an identity.
#         # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
#         if zero_init_residual:
#             for m in self.modules():
#                 if isinstance(m, AxialBlock):
#                     nn.init.constant_(m.bn2.weight, 0)

#     def _make_layer(self, block, planes, blocks, kernel_size=56, stride=1, dilate=False):
#         norm_layer = self._norm_layer
#         downsample = None
#         previous_dilation = self.dilation
#         if dilate:
#             self.dilation *= stride
#             stride = 1
#         if stride != 1 or self.inplanes != planes * block.expansion:
#             downsample = nn.Sequential(
#                 conv1x1(self.inplanes, planes * block.expansion, stride),
#                 norm_layer(planes * block.expansion),
#             )

#         layers = []
#         layers.append(block(self.inplanes, planes, stride, downsample, groups=self.groups,
#                             base_width=self.base_width, dilation=previous_dilation, 
#                             norm_layer=norm_layer, kernel_size=kernel_size))
#         self.inplanes = planes * block.expansion
#         if stride != 1:
#             kernel_size = kernel_size // 2

#         for _ in range(1, blocks):
#             layers.append(block(self.inplanes, planes, groups=self.groups,
#                                 base_width=self.base_width, dilation=self.dilation,
#                                 norm_layer=norm_layer, kernel_size=kernel_size))

#         return nn.Sequential(*layers)

#     def _forward_impl(self, x):
#         # See note [TorchScript super()]
#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = self.relu(x)
#         x = self.maxpool(x)

#         x = self.layer1(x)
#         x = self.layer2(x)
#         x = self.layer3(x)
#         x = self.layer4(x)

#         x = self.avgpool(x)
#         x = torch.flatten(x, 1)
#         x = self.fc(x)

#         return x

#     def forward(self, x):
#         return self._forward_impl(x)
    
# import torch.nn as nn

# class qkv_transform(nn.Conv1d):
#     """Conv1d for qkv_transform"""

# def axial26s(pretrained=False, in_channel=1, **kwargs):
#     model = AxialAttentionNet(AxialBlock, [1, 2, 4, 1],in_channel=in_channel,s=0.5, **kwargs)
#     return model


# def axial50s(pretrained=False, in_channel=1,**kwargs):
#     model = AxialAttentionNet(AxialBlock, [3, 4, 6, 3],in_channel=in_channel,s=0.5, **kwargs)
#     return model


# def axial50m(pretrained=False, in_channel=1,**kwargs):
#     model = AxialAttentionNet(AxialBlock, [3, 4, 6, 3],in_channel=in_channel,s=0.75, **kwargs)
#     return model


# def axial50l(pretrained=False, in_channel=1,**kwargs):
#     model = AxialAttentionNet(AxialBlock, [3, 4, 6, 3],in_channel=in_channel, s=1, **kwargs)
#     return model