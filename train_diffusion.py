import argparse
import os
import random
import socket
import yaml
import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
import numpy as np
import torchvision
import models
import datasets
import utils
from models import DenoisingDiffusion


from config import parse_args_and_config





def main():
    args = parse_args_and_config()

    # setup device to run
    device = args.device
    print("Using device: {}".format(device))

    # set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = True
    print('The random seed is set : {}'.format(args.seed))

    # data loading
    print("=> using dataset '{}'".format(args.dataset))
    DATASET = datasets.__dict__[args.dataset](args)

    # create model
    print("=> creating denoising-diffusion model...")
    diffusion = DenoisingDiffusion(args)
    diffusion.train(DATASET)


if __name__ == "__main__":
    main()
