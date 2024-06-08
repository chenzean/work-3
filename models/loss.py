# -*- coding: utf-8 -*-
# @Time    : 2024/5/12 18:36
# @Author  : Chen Zean
# @Site    : Ningbo University
# @File    : loss.py
# @Software: PyCharm

import torch
import torch.nn as nn
import math
from torch.autograd import Variable
import torch.nn.functional as F
import random

def L1_Charbonnier_loss(X, Y):
    eps = 1e-6
    diff = torch.add(X, -Y)
    error = torch.sqrt(diff * diff + eps)
    Charbonnier_loss = torch.sum(error) / torch.numel(error)
    return Charbonnier_loss


class VGGInfoNCE(nn.Module):
    def __init__(self, args):
        super(VGGInfoNCE, self).__init__()
        self.l1 = nn.L1Loss()
        self.args = args
        self.cl_layer = 4
        self.sr_factor = args.SR_factor

    def infer(self, x):
        return self.vgg(x)

    def forward(self, mdoel, input, t, sr, hr, lr, ref_coord):
        # sr_features = self.vgg(sr)
        # if not isinstance(hr, list):
        #     hr = [hr, ]
        # if not isinstance(lr, list):
        #     lr = [lr, ]
        #
        # if self.args.pos_id != -1:
        #     hr = [hr[self.pos_id], ]
        hr_list =[]
        hr_list.append(hr)
        lr_list = []
        lr_list.append(lr)
        b,c,h,w = hr.shape

        # 通过网络
        with torch.no_grad():
            neg_image = mdoel(input,t, ref_coord)
            lr_list.append(neg_image)
        for i in range(self.cl_layer):
            # 插值
            factor = torch.round(torch.rand(1) * self.sr_factor * 10) / 10
            factor = factor.item()
            if factor == 0:
                factor = 1
            hr_down = F.interpolate(hr, scale_factor=(1/factor), mode='bicubic')
            neg_image = F.interpolate(hr_down,size=(h,w), mode='bicubic')
            lr_list.append(neg_image)




        loss = self.infoNCE(sr, hr_list, lr_list)
        return loss

    def infoNCE(self, sr, hr, lr):
        # b, c, h, w = hr.shape

        infoNCE_loss = 0

        if self.args.cl_loss_type == 'InfoNCE_L1':
            nce_loss = self.l1_nce(sr, hr, lr)
        elif self.args.cl_loss_type == 'InfoNCE':
            nce_loss = self.nce(sr, hr, lr)
        else:
            raise TypeError(f'{self.args.cl_loss_type} is not found')

        infoNCE_loss += nce_loss

        return infoNCE_loss / self.cl_layer

    def l1_nce(self, sr_layer, hr_layers, lr_layers):

        loss = 0
        b, c, h, w = sr_layer.shape

        neg_logits = []
        for f_lr in lr_layers:
            neg_diff = torch.abs(sr_layer-f_lr).mean(dim=[-3, -2, -1]).unsqueeze(1)
            neg_logits.append(neg_diff)

        pos_logits = []
        for f_hr in hr_layers:
            pos_diff = torch.abs(sr_layer-f_hr).mean(dim=[-3, -2, -1]).unsqueeze(1)
            pos_logits.append(pos_diff)

            if self.args.cl_loss_type == 'InfoNCE':
                logits = torch.cat(pos_logits + neg_logits, dim=1)
                cl_loss = F.cross_entropy(logits, torch.zeros(b, device=logits.device, dtype=torch.long)) # self.ce_loss(logits)

            elif self.args.cl_loss_type == 'InfoNCE_L1':
                neg_logits = torch.cat(neg_logits, dim=1).mean(dim=1, keepdim=True)
                cl_loss = torch.mean(pos_logits[0] / neg_logits)

            elif self.args.cl_loss_type == 'LMCL':
                cl_loss = self.lmcl_loss(pos_logits + neg_logits)

            else:
                raise TypeError(f'{self.args.cl_loss_type} is not found in loss/cl.py')

            loss += cl_loss
        return loss / len(lr_layers)


    def nce(self, sr_layer, hr_layers, lr_layers):

        loss = 0
        b, c, h, w = sr_layer.shape

        neg_logits = []

        for f_lr in lr_layers:
            neg_diff = torch.sum(
                F.normalize(sr_layer, dim=1) * F.normalize(f_lr, dim=1), dim=1).mean(dim=[-1, -2]).unsqueeze(1)
            neg_logits.append(neg_diff)

        if self.args.shuffle_neg:
            batch_list = list(range(b))

            for f_lr in lr_layers:
                random.shuffle(batch_list)
                neg_diff = torch.sum(
                    F.normalize(sr_layer, dim=1) * F.normalize(f_lr[batch_list, :, :, :], dim=1), dim=1).mean(
                    dim=[-1, -2]).unsqueeze(1)
                neg_logits.append(neg_diff)

        for f_hr in hr_layers:
            pos_logits = []
            pos_diff = torch.sum(
                F.normalize(sr_layer, dim=1) * F.normalize(f_hr, dim=1), dim=1).mean(dim=[-1, -2]).unsqueeze(1)
            pos_logits.append(pos_diff)

            if self.args.cl_loss_type == 'InfoNCE':
                logits = torch.cat(pos_logits + neg_logits, dim=1)
                cl_loss = F.cross_entropy(logits, torch.zeros(b, device=logits.device, dtype=torch.long)) # self.ce_loss(logits)
            elif self.args.cl_loss_type == 'LMCL':
                cl_loss = self.lmcl_loss(pos_logits + neg_logits)
            else:
                raise TypeError(f'{self.args.cl_loss_type} is not found in loss/cl.py')
            loss += cl_loss
        return loss / len(hr_layers)

import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from math import exp


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


class SSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)

            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)

            self.window = window
            self.channel = channel

        return _ssim(img1, img2, window, self.window_size, channel, self.size_average)


def ssim(img1, img2, window_size=11, size_average=True):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)




def ssim_loss(img1, img2):
    return 1 - ssim(img1, img2)