import argparse
import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

import torch
import torch.nn as nn




class bsbias(nn.Module):
    def __init__(self, dim, scale=1.0):
        super().__init__()
        assert dim % 2 == 0
        self.dim = dim
        self.scale = scale

    def forward(self, x): 
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / half_dim
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = torch.outer(x * self.scale, emb)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)

        return emb


class butterfly_conv(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, padding=1, dilation=1, groups=1, bias=False, b_type='ud'):
        super(butterfly_conv, self).__init__()
        assert b_type in ['ud', 'lr'], 'b_type should be in ud or lr'
        assert dilation in [1, 2], 'dilation for ad_conv should be in 1 or 2'
        assert padding == dilation, 'padding for ad_conv set wrong'
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.b_type = b_type
        self.weight = nn.Parameter(torch.Tensor(self.out_channels, self.in_channels // self.groups, 3, 3))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        self.mask1 = torch.cuda.FloatTensor([[1, 0, 1], [1, 1, 1], [1, 0, 1]])
        self.mask2 = torch.cuda.FloatTensor([[1, 1, 1], [0, 1, 0], [1, 1, 1]])

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        if self.b_type == 'lr':
            weight = torch.mul(self.weight, self.mask1)
        elif self.b_type == 'ud':
            weight = torch.mul(self.weight, self.mask2)
        o = F.conv2d(input, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

        return o
    



class layer_b(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(layer_b, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, padding=1, groups=in_channels),
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=1))
        self.conv33_ud = nn.Sequential(
            butterfly_conv(in_channels=in_channels, out_channels=in_channels, padding=1, dilation=1,
                           groups=in_channels, bias=True, b_type='ud'),
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, bias=False))
        self.conv33_lr = nn.Sequential(
            butterfly_conv(in_channels=in_channels, out_channels=in_channels, padding=1, dilation=1,
                           groups=in_channels, bias=True, b_type='lr'),
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, bias=False))
        self.conv33d_ud = nn.Sequential(
            butterfly_conv(in_channels=in_channels, out_channels=in_channels, padding=2, dilation=2,
                           groups=in_channels, bias=True, b_type='ud'),
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, bias=False))
        self.conv33d_lr = nn.Sequential(
            butterfly_conv(in_channels=in_channels, out_channels=in_channels, padding=2, dilation=2,
                           groups=in_channels, bias=True, b_type='lr'),
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, bias=False))

        self.down = nn.Sequential(
            nn.Conv2d(in_channels=out_channels * 4, out_channels=out_channels * 4, kernel_size=3, padding=1,
                      groups=out_channels),
            nn.InstanceNorm2d(out_channels * 4),
            nn.GELU(),
            nn.Conv2d(in_channels=out_channels * 4, out_channels=out_channels, kernel_size=1, bias=False))
        self.shortcut = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, bias=False)


    def forward(self, x):
        x = self.conv1(x)
        conv33_ud = self.conv33_ud(x)
        conv33_lr = self.conv33_lr(x)
        conv33d_ud = self.conv33d_ud(x)
        conv33d_lr = self.conv33d_lr(x)
        o = torch.cat([conv33_ud, conv33_lr, conv33d_ud, conv33d_lr], dim=1)
        o = self.down(o)
        o = self.shortcut(x) + o
        return o


class CPX(nn.Module):
    def __init__(self, channels, scale=1.5, args=None):
        super(CPX, self).__init__()
        self.channels = channels
        self.time_emb = nn.Sequential(bsbias(channels),
                                      nn.Linear(channels, channels * 2),
                                      nn.SiLU(),
                                      nn.Linear(channels * 2, channels))
        self.step = torch.tensor([[i + 1] for i in range(args.range)]).to(args.device_num)

    def forward(self, x):
        time_emb = self.time_emb(self.step[0])[:, :, None, None]  
        x = x + time_emb  
        return x

        


class encoder(nn.Module):
    def __init__(self,args):
        super(encoder, self).__init__()
        basic_c = args.basic_c
        channels = [basic_c, basic_c*2, basic_c*3, basic_c*3]
        self.input = nn.Conv2d(in_channels=channels[0], out_channels=channels[0], kernel_size=3, padding=1)
        self.stage1 = nn.Sequential(
            layer_b(in_channels=channels[0], out_channels=channels[0]),
            nn.InstanceNorm2d(channels[0]), nn.GELU(),
            nn.Dropout(p=0.3),
            layer_b(in_channels=channels[0], out_channels=channels[0]))

        self.stage2 = nn.Sequential(
            layer_b(in_channels=channels[1], out_channels=channels[1]),
            nn.InstanceNorm2d(channels[1]), nn.GELU(),
            nn.Dropout(p=0.3),
            layer_b(in_channels=channels[1], out_channels=channels[1]))

        self.stage3 = nn.Sequential(
            layer_b(in_channels=channels[2], out_channels=channels[2]),
            nn.InstanceNorm2d(channels[2]), nn.GELU(),
            nn.Dropout(p=0.3),
            layer_b(in_channels=channels[2], out_channels=channels[2]))

        self.stage4 = nn.Sequential(
            layer_b(in_channels=channels[3], out_channels=channels[3]),
            nn.InstanceNorm2d(channels[3]), nn.GELU(),
            nn.Dropout(p=0.3),
            layer_b(in_channels=channels[3], out_channels=channels[3]))
        self.up1 = nn.Conv2d(in_channels=channels[0], out_channels=channels[1], kernel_size=1)
        self.up2 = nn.Conv2d(in_channels=channels[1], out_channels=channels[2], kernel_size=1)

        self.pool1 = nn.MaxPool2d(2)
        self.pool2 = nn.MaxPool2d(2)
        self.pool3 = nn.MaxPool2d(2)

        self.avgpool1 = nn.AvgPool2d(2)
        self.avgpool2 = nn.AvgPool2d(2)
        self.avgpool3 = nn.AvgPool2d(2)

        self.conv1 = nn.Conv2d(in_channels=channels[0],out_channels=channels[1],kernel_size=1, padding=0)
        self.conv2 = nn.Conv2d(in_channels=channels[1],out_channels=channels[2],kernel_size=1, padding=0)
        self.conv3 = nn.Conv2d(in_channels=channels[2],out_channels=channels[3],kernel_size=1, padding=0)

    def forward(self, x):
        x = self.input(x)
        o1 = self.stage1(x)
        o1a = self.pool1(o1)
        o2 = self.stage2(self.up1(self.avgpool1(o1)))
        o2a = self.pool2(o2)
        o3 = self.stage3(self.up2(self.avgpool2(o2)))
        o3a = self.pool3(o3)
        o4 = self.stage4(self.avgpool3(o3))

        o1a = self.conv1(o1a)
        o2a = self.conv2(o2a)
        o3a = self.conv3(o3a)

        return [o1, o2, o3, o4], [o1a, o2a, o3a]


class promptconv(nn.Module):
    def __init__(self, channels, ratio=2):
        super(promptconv, self).__init__()
        self.channels_expan = nn.Conv2d(channels, channels * ratio ** 2 - channels, kernel_size=1, padding=0)

        self.SubConv = nn.PixelShuffle(ratio)
        self.conv = nn.Sequential(nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, padding=1,groups=channels),
                                  nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=1, padding=0))

    def forward(self, x, xa):
        x = self.channels_expan(x)
        x = torch.cat([x, xa], dim=1)
        x = self.SubConv(x)
        x = self.conv(x)

        return x
    

class decoder(nn.Module):
    def __init__(self,args):
        super(decoder, self).__init__()
        basic_c = args.basic_c
        channels = [basic_c, basic_c*2, basic_c*3, basic_c*3]
        self.prompt43 = promptconv(channels[3])
        self.prompt32 = promptconv(channels[2])
        self.prompt21 = promptconv(channels[1])
        self.frm3 = nn.Sequential(nn.Conv2d(in_channels=channels[3]+channels[2], out_channels=channels[3]+channels[2], kernel_size=3, padding=1,groups=channels[3]+channels[2]),
                                     nn.InstanceNorm2d(channels[3]+channels[2]),nn.GELU(),
                                     nn.Conv2d(in_channels=channels[3]+channels[2], out_channels=channels[2], kernel_size=1, padding=0))
        self.frm2 = nn.Sequential(nn.Conv2d(in_channels=channels[2]+channels[1], out_channels=channels[2]+channels[1], kernel_size=3, padding=1,groups=channels[2]+channels[1]),
                                     nn.InstanceNorm2d(channels[2]+channels[1]),nn.GELU(),
                                     nn.Conv2d(in_channels=channels[2]+channels[1], out_channels=channels[1], kernel_size=1, padding=0))
        self.frm1 = nn.Sequential(nn.Conv2d(in_channels=channels[1]+channels[0], out_channels=channels[1]+channels[0], kernel_size=3, padding=1,groups=channels[1]+channels[0]),
                                     nn.InstanceNorm2d(channels[1]+channels[0]),nn.GELU(),
                                     nn.Conv2d(in_channels=channels[1]+channels[0], out_channels=channels[1]+channels[0], kernel_size=1, padding=0))
        self.head = nn.Conv2d(in_channels=channels[1]+channels[0], out_channels=1, kernel_size=3, padding=1)

    def forward(self, x, diff):
        _, _, h1, w1 = x[0].shape
        _, _, h2, w2 = x[1].shape
        _, _, h3, w3 = x[2].shape
        x3 = self.prompt43(x[3], diff[2])
        if x3.shape[-2:] != [h3,w3]:
            x3 = F.interpolate(x3, [h3,w3], mode="bilinear", align_corners=True)
        x3 = torch.cat([x3,x[2]],dim=1)
        x3 = self.frm3(x3)
            
        x2 = self.prompt32(x3, diff[1])
        if x2.shape[-2:] != [h2,w2]:
            x2 = F.interpolate(x2, [h2,w2], mode="bilinear", align_corners=True)
        x2 = torch.cat([x2,x[1]],dim=1)
        x2 = self.frm2(x2)

        x1 = self.prompt21(x2, diff[0])
        if x1.shape[-2:] != [h1,w1]:
            x1 = F.interpolate(x1, [h1,w1], mode="bilinear", align_corners=True)
        x1 = torch.cat([x1,x[0]],dim=1)
        x1 = self.frm1(x1)

        out =  self.head(x1)
        

        return out.sigmoid()#, x1.mean(), x2.mean(), x3.mean()


class bcnet(nn.Module):
    def __init__(self, args=None):
        super(bcnet, self).__init__()
        basic_c = args.basic_c
        channels = [basic_c, basic_c*2, basic_c*3, basic_c*3]
        self.input = nn.Conv2d(in_channels=3, out_channels=channels[0], kernel_size=3, padding=1)
        self.args = args
        self.cpx = CPX(args=args, channels=channels[0])
        self.encoder = encoder(args)
        self.decoder = decoder(args)


    def forward(self, x):
        feaure = self.input(x)
        feaure = self.cpx(x=feaure)  # 加入BS bias
        feaure, diff = self.encoder(feaure)
        # edge, x1, x2, x3 = self.decoder(feaure, diff)
        edge = self.decoder(feaure, diff)
        return edge# , x1, x2, x3
