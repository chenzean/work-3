import argparse
import os
import random
import socket
import yaml
import torch
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
import models
import datasets
import utils
from models import DenoisingDiffusion, DiffusiveRestoration

from config import parse_args_and_config


def main():
    args = parse_args_and_config()

    # setup device to run
    device = args.device
    print("Using device: {}".format(device))

    if torch.cuda.is_available():
        print('Note: Currently supports evaluations (restoration) when run only on a single GPU!')

    # set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = True

    # data loading
    print("=> using dataset '{}'".format(args.dataset))
    DATASET = datasets.__dict__[args.dataset](args)
    _, test_loader = DATASET.get_loaders(parse_patches=False, validation=args.dataset)

    # create model
    print("=> creating denoising-diffusion model with wrapper...")
    diffusion = DenoisingDiffusion(args)
    model = DiffusiveRestoration(diffusion, args)
    model.restore(test_loader, validation=args.dataset,)


if __name__ == '__main__':
    main()
