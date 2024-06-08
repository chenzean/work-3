import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import numpy as np


class SAS_Conv2D(nn.Module):
    def __init__(self, channels, args):
        super(SAS_Conv2D, self).__init__()
        self.chan_num = channels
        self.an = args.angRes
        self.spaconv = nn.Conv2d(in_channels=self.chan_num, out_channels=self.chan_num, kernel_size=3, stride=1, padding=1, dilation=1, bias=True)   # spatial conv
        self.angconv = nn.Conv2d(in_channels=self.chan_num, out_channels=self.chan_num, kernel_size=3, stride=1, padding=1, dilation=1, bias=True)   # angular conv
        self.sas_relu = nn.ReLU(inplace=True)

    def forward(self, x):
        N, c, h, w = x.shape        # [N*u*v,c,h,w]
        N = N // (self.an * self.an)

        out = self.spaconv(x)       # [N*u*v,c,h,w]  spatial conv
        out = self.sas_relu(out)
        out = out.reshape(N, self.an * self.an, c, h * w)    # [N,u*v,c,h*w]
        out = torch.transpose(out, 1, 3)                  # [N,h*w,c,u*v]
        out = out.reshape(N * h * w, c, self.an, self.an)    # [N*h*w,c,u,v]

        out = self.angconv(out)     # [N*h*w,c,u,v]   angular conv
        out = out.reshape(N, h * w, c, self.an * self.an)    # [N,h*w,c,u*v]
        out = torch.transpose(out, 1, 3)                  # [N,u*v,c,h*w]
        out = out.reshape(N * self.an * self.an, c, h, w)    # [N*u*v,c,h,w]
        return out


