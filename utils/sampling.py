import torch
import os
import torchvision
from torchvision.transforms.functional import crop


# This script is adapted from the following repository: https://github.com/ermongroup/ddim

def compute_alpha(beta, t):
    beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
    a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1, 1)
    return a


def data_transform(X):
    return 2 * X - 1.0


def inverse_data_transform(X):
    return torch.clamp((X + 1.0) / 2.0, 0.0, 1.0)


def generalized_steps(x, x_cond, seq, model, b, eta=0.):
    '''
    Args:
        x: 噪声
        x_cond: 条件
        seq: 采样的序列
        model: 扩散模型
        b: beta
        eta: 0等于使用DDIM(快速采样),1使用DDPM

    Returns:

    '''
    with torch.no_grad():
        n = x.size(0)
        seq_next = [-1] + list(seq[:-1])
        x0_preds = []
        xs = [x]
        for i, j in zip(reversed(seq), reversed(seq_next)):
            t = (torch.ones(n) * i).to(x.device)
            next_t = (torch.ones(n) * j).to(x.device)
            at = compute_alpha(b, t.long())
            at_next = compute_alpha(b, next_t.long())
            xt = xs[-1].to('cuda')

            xo_t  = model(torch.cat([xt, x_cond], dim=1), t)
            et = (xt - xo_t * at.sqrt()) / (1 - at).sqrt()
            # 预测噪声的推导公式
            # x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()
            x0_preds.append(xo_t.to('cpu'))
            # x0_preds.append(x0_t.to('cpu'))

            c1 = eta * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
            c2 = ((1 - at_next) - c1 ** 2).sqrt()
            # !!!!!!!重点
            # xt_next = at_next.sqrt() * x0_t + c1 * torch.randn_like(x) + c2 * et
            xt_next = at_next.sqrt() * xo_t + c1 * torch.randn_like(x) + c2 * et
            # ##################################################################
            # t时刻
            xs.append(xt_next.to('cpu'))
    return xs, x0_preds