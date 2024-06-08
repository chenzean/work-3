# -*- coding: utf-8 -*-
# @Time    : 2024/5/12 18:33
# @Author  : Chen Zean
# @Site    : Ningbo University
# @File    : untils_models.py
# @Software: PyCharm

import numpy as np
import math
import os
from skimage import metrics
import torch
import torch.nn as nn
from pathlib import Path
import matplotlib.pyplot as plt
import logging
# from config import args
from einops import rearrange
import xlwt
import torch.nn.functional as F
from matplotlib import pyplot as plt



def data_transform(X):
    return 2 * X - 1.0


def inverse_data_transform(X):
    return torch.clamp((X + 1.0) / 2.0, 0.0, 1.0)


# 用于实现指数移动平均 (Exponential Moving Average, EMA)。EMA 是一种平滑时间序列数据的方法，它可以用来平滑模型的参数，以减少训练过程中的参数抖动和不稳定性。
class EMAHelper(object):
    def __init__(self, mu=0.9999):
        self.mu = mu
        self.shadow = {}

    def register(self, module1):
        if isinstance(module1, nn.DataParallel):
            module1 = module1.module
        for name, param in module1.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, module1):
        if isinstance(module1, nn.DataParallel):
            module1 = module1.module
        for name, param in module1.named_parameters():
            if param.requires_grad:
                self.shadow[name].data = (1. - self.mu) * param.data + self.mu * self.shadow[name].data

    def ema(self, module):
        if isinstance(module, nn.DataParallel):
            module = module.module
        for name, param in module.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.shadow[name].data)

    def ema_copy(self, module):
        if isinstance(module, nn.DataParallel):
            inner_module = module.module
            module_copy = type(inner_module)(inner_module.config).to(inner_module.config.device)
            module_copy.load_state_dict(inner_module.state_dict())
            module_copy = nn.DataParallel(module_copy)
        else:
            module_copy = type(module)(module.config).to(module.config.device)
            module_copy.load_state_dict(module.state_dict())
        self.ema(module_copy)
        return module_copy

    def state_dict(self):
        return self.shadow

    def load_state_dict(self, state_dict):
        self.shadow = state_dict


# 通常用于配置扩散模型中的扩散过程，其中 beta 值用于控制数据逐渐淡化和变化的速度
def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    def sigmoid(x):
        return 1 / (np.exp(-x) + 1)

    if beta_schedule == "quad":
        betas = (np.linspace(beta_start ** 0.5, beta_end ** 0.5, num_diffusion_timesteps, dtype=np.float64) ** 2)
    elif beta_schedule == "linear":
        betas = np.linspace(beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "const":
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "jsd":  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1.0 / np.linspace(num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "sigmoid":
        betas = np.linspace(-6, 6, num_diffusion_timesteps)
        betas = sigmoid(betas) * (beta_end - beta_start) + beta_start
    elif beta_schedule == "cosine":
        pass
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas





class ExcelFile():
    def __init__(self):
        self.xlsx_file = xlwt.Workbook()
        self.worksheet = self.xlsx_file.add_sheet(r'sheet1', cell_overwrite_ok=True)
        self.worksheet.write(0, 0, 'Datasets')
        self.worksheet.write(0, 1, 'Scenes')
        self.worksheet.write(0, 2, 'PSNR')
        self.worksheet.write(0, 3, 'SSIM')
        self.worksheet.col(0).width = 256 * 16
        self.worksheet.col(1).width = 256 * 22
        self.worksheet.col(2).width = 256 * 10
        self.worksheet.col(3).width = 256 * 10
        self.sum = 1

    def write_sheet(self, test_name, LF_name, psnr_iter_test, ssim_iter_test):
        ''' Save PSNR & SSIM '''
        for i in range(len(psnr_iter_test)):
            self.add_sheet(test_name, LF_name[i], psnr_iter_test[i], ssim_iter_test[i])

        psnr_epoch_test = float(np.array(psnr_iter_test).mean())
        ssim_epoch_test = float(np.array(ssim_iter_test).mean())
        self.add_sheet(test_name, 'average', psnr_epoch_test, ssim_epoch_test)
        self.sum = self.sum + 1

    def add_sheet(self, test_name, LF_name, psnr_iter_test, ssim_iter_test):
        ''' Save PSNR & SSIM '''
        self.worksheet.write(self.sum, 0, test_name)
        self.worksheet.write(self.sum, 1, LF_name)
        self.worksheet.write(self.sum, 2, '%.6f' % psnr_iter_test)
        self.worksheet.write(self.sum, 3, '%.6f' % ssim_iter_test)
        self.sum = self.sum + 1


class ExcelFile_allviews():
    def __init__(self):
        self.xlsx_file = xlwt.Workbook()
        self.worksheet = self.xlsx_file.add_sheet(r'sheet1', cell_overwrite_ok=True)
        self.worksheet.write(0, 0, 'Datasets')
        self.worksheet.write(0, 1, 'Scenes')
        self.worksheet.write(0, 2, 'PSNR')
        self.worksheet.write(0, 3, 'SSIM')
        self.worksheet.col(0).width = 256 * 16
        self.worksheet.col(1).width = 256 * 22
        self.worksheet.col(2).width = 256 * 10
        self.worksheet.col(3).width = 256 * 10
        self.sum = 1

    def write_sheet(self, test_name, LF_name, psnr_iter_test, ssim_iter_test, psnr_iter_test_allviews,
                    ssim_iter_test_allviews):
        ''' Save PSNR & SSIM '''
        for i in range(len(psnr_iter_test)):
            self.add_sheet(test_name, LF_name[i], psnr_iter_test[i], ssim_iter_test[i])
            self.sum = self.sum - 1
            temp_views = np.array(psnr_iter_test_allviews[i]).reshape(-1)
            self.worksheet.write(self.sum, 4, '%.6f' % math.sqrt(np.var(temp_views)))
            self.sum = self.sum + 1
            b, u, v = psnr_iter_test_allviews[i].shape
            for x in range(u):
                for y in range(v):
                    self.add_sheet("view", str(x + 1) + "_" + str(y + 1), psnr_iter_test_allviews[i][:, x, y],
                                   ssim_iter_test_allviews[i][:, x, y])  # 写入该视角下每个数据

            self.sum = self.sum + 1

        psnr_epoch_test = float(np.array(psnr_iter_test).mean())
        ssim_epoch_test = float(np.array(ssim_iter_test).mean())
        self.add_sheet(test_name, 'average', psnr_epoch_test, ssim_epoch_test)
        self.sum = self.sum + 1

    def add_sheet(self, test_name, LF_name, psnr_iter_test, ssim_iter_test):
        ''' Save PSNR & SSIM '''
        self.worksheet.write(self.sum, 0, test_name)
        self.worksheet.write(self.sum, 1, LF_name)
        self.worksheet.write(self.sum, 2, '%.6f' % psnr_iter_test)
        self.worksheet.write(self.sum, 3, '%.6f' % ssim_iter_test)
        self.sum = self.sum + 1


def get_logger(log_dir, args):
    '''LOG '''
    logger = logging.getLogger(args.model_name)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/%s.txt' % (log_dir, args.model_name))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def create_dir(path):
    log_dir_log = Path(path)
    log_dir_log.mkdir(exist_ok=True)
    task_path = 'SR_' + str(5) + 'x' + str(5) + '_' + str(4) + 'x'
    log_dir = log_dir_log.joinpath(task_path)
    log_dir.mkdir(exist_ok=True)
    log_dir = log_dir.joinpath('HILFSSR_Diffusion')
    log_dir.mkdir(exist_ok=True)
    # log_dir = log_dir.joinpath('DDPM_LFSR')
    # log_dir.mkdir(exist_ok=True)

    # checkpoints_dir = log_dir.joinpath('checkpoints/')
    # checkpoints_dir.mkdir(exist_ok=True)

    results_dir = log_dir.joinpath('results/')
    results_dir.mkdir(exist_ok=True)
    CenterView_dir = log_dir.joinpath('CenterView/')
    CenterView_dir.mkdir(exist_ok=True)
    mat_file = log_dir.joinpath('mat_file/')
    mat_file.mkdir(exist_ok=True)

    return log_dir_log, results_dir, CenterView_dir, mat_file


class Logger():
    def __init__(self, log_dir, args):
        self.logger = get_logger(log_dir, args)

    def log_string(self, str):
        self.logger.info(str)
        print(str)


def cal_metrics(label, out, ):
    # print('label', label)     # [3, 1, 1280, 1280]float32
    # print('out', out)         # [3, 1, 1280, 1280]float32
    if len(label.size()) == 4:
        label = rearrange(label, 'b c (a1 h) (a2 w) -> b c a1 h a2 w', a1=5, a2=5)
        out = rearrange(out, 'b c (a1 h) (a2 w) -> b c a1 h a2 w', a1=5, a2=5)

    if len(label.size()) == 5:
        label = label.permute((0, 1, 3, 2, 4)).unsqueeze(0)
        out = out.permute((0, 1, 3, 2, 4)).unsqueeze(0)

    B, C, U, h, V, w = label.size()
    label_y = label[:, 0, :, :, :, :].data.cpu()
    out_y = out[:, 0, :, :, :, :].data.cpu()

    PSNR = np.zeros(shape=(B, U, V), dtype='float32')
    SSIM = np.zeros(shape=(B, U, V), dtype='float32')
    for b in range(B):
        for u in range(U):
            for v in range(V):
                PSNR[b, u, v] = metrics.peak_signal_noise_ratio(label_y[b, u, :, v, :].numpy(),
                                                                out_y[b, u, :, v, :].numpy(), data_range=1.0)
                SSIM[b, u, v] = metrics.structural_similarity(label_y[b, u, :, v, :].numpy(),
                                                              out_y[b, u, :, v, :].numpy(),
                                                              gaussian_weights=True, data_range=1.0)

    return PSNR, SSIM


def ImageExtend(Im, bdr):
    '''
    Im 的维度是25, 1, 108, 156
    bdr 的维度是[8, 23, 8, 23 ]
    '''
    [_, _, h, w] = Im.size()
    # 左右翻转 Im_lr
    Im_lr = torch.flip(Im, dims=[-1])
    # 上下翻转
    Im_ud = torch.flip(Im, dims=[-2])
    # 对角线翻转
    Im_diag = torch.flip(Im, dims=[-1, -2])

    Im_up = torch.cat((Im_diag, Im_ud, Im_diag), dim=-1)
    Im_mid = torch.cat((Im_lr, Im, Im_lr), dim=-1)
    Im_down = torch.cat((Im_diag, Im_ud, Im_diag), dim=-1)

    Im_Ext = torch.cat((Im_up, Im_mid, Im_down), dim=-2)
    Im_out = Im_Ext[:, :, h - bdr[0]: 2 * h + bdr[1], w - bdr[2]: 2 * w + bdr[3]]

    return Im_out


def LFdivide(data, angRes, patch_size, stride):
    data = rearrange(data, '(a1 h) (a2 w) -> (a1 a2) 1 h w', a1=angRes, a2=angRes)  # 25, 1, 108, 156
    # data_show = data.cpu().numpy()
    # plt.imshow(data_show[1,0,:,:])
    # plt.show()
    [_, _, h0, w0] = data.size()
    bdr = (patch_size - stride) // 2
    numU = (h0 + bdr * 2 - 1) // stride
    numV = (w0 + bdr * 2 - 1) // stride
    data_pad = ImageExtend(data, [bdr, bdr + stride - 1, bdr, bdr + stride - 1])  # 25,1,239,187
    subLF = F.unfold(data_pad, kernel_size=patch_size, stride=stride)  # 25, 1024,70
    subLF = rearrange(subLF, '(a1 a2) (h w) (n1 n2) -> n1 n2 (a1 h) (a2 w)',
                      a1=angRes, a2=angRes, h=patch_size, w=patch_size, n1=numU, n2=numV)
    return subLF


def Refdivide(data, angRes, patch_size, stride, scale_factor):
    [_, _, h0, w0] = data.shape
    patch_size = patch_size * scale_factor
    stride = int(patch_size / 2)
    bdr = int((patch_size - stride) // 2)
    numU = int((h0 + bdr * 2 - 1) // stride)
    numV = int((w0 + bdr * 2 - 1) // stride)
    data_pad = ImageExtend(data, [bdr, int(bdr + stride - 1), bdr, int(bdr + stride - 1)])
    subRef = F.unfold(data_pad, kernel_size=patch_size, stride=stride)
    subRef = subRef.squeeze()
    subRef = rearrange(subRef, '(h w) (n1 n2) -> n1 n2 h w', h=patch_size, w=patch_size, n1=numU, n2=numV)

    return subRef


def LFintegrate(subLF, angRes, pz, stride, h, w):
    if subLF.dim() == 4:
        subLF = rearrange(subLF, 'n1 n2 (a1 h) (a2 w) -> n1 n2 a1 a2 h w', a1=angRes, a2=angRes)
        pass
    bdr = (pz - stride) // 2
    outLF = subLF[:, :, :, :, bdr:bdr + stride, bdr:bdr + stride]
    outLF = rearrange(outLF, 'n1 n2 a1 a2 h w -> a1 a2 (n1 h) (n2 w)')
    outLF = outLF[:, :, 0:h, 0:w]

    return outLF


def rgb2ycbcr(x):
    y = np.zeros(x.shape, dtype='double')
    y[:, :, 0] = 65.481 * x[:, :, 0] + 128.553 * x[:, :, 1] + 24.966 * x[:, :, 2] + 16.0
    y[:, :, 1] = -37.797 * x[:, :, 0] - 74.203 * x[:, :, 1] + 112.000 * x[:, :, 2] + 128.0
    y[:, :, 2] = 112.000 * x[:, :, 0] - 93.786 * x[:, :, 1] - 18.214 * x[:, :, 2] + 128.0

    y = y / 255.0
    return y


def ycbcr2rgb(in_ycbcr):
    # YCbCr to RGB
    in_y = np.expand_dims(in_ycbcr[:, :, 0], axis=-1)
    in_cb = np.expand_dims(in_ycbcr[:, :, 1], axis=-1)
    in_cr = np.expand_dims(in_ycbcr[:, :, 2], axis=-1)

    in_y = in_y * 255.0 - 16.
    in_cb = in_cb * 255.0 - 128.
    in_cr = in_cr * 255.0 - 128.

    tran_m = np.array([[65.481, 128.553, 24.966],
                       [-37.797, -74.203, 112],
                       [112, -93.786, -18.214]])

    tran_l = np.linalg.inv(tran_m.transpose()) * 255.

    rec_r = tran_l[0, 0] * in_y + tran_l[1, 0] * in_cb + tran_l[2, 0] * in_cr
    rec_g = tran_l[0, 1] * in_y + tran_l[1, 1] * in_cb + tran_l[2, 1] * in_cr
    rec_b = tran_l[0, 2] * in_y + tran_l[1, 2] * in_cb + tran_l[2, 2] * in_cr
    out_rgb_img = np.concatenate((rec_r, rec_g, rec_b), axis=-1)
    return out_rgb_img / 255.
