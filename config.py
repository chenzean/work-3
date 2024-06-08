import argparse


def parse_args_and_config():
    parser = argparse.ArgumentParser(description='Training Patch-Based Denoising Diffusion Models LFSSR-HI')
    parser.add_argument('--model_name', default='LFSSR-HI-Diffusion-Mamba', type=str)

    # data相关参数设置
    parser.add_argument('--image_size', default=96, type=int)
    parser.add_argument('--conditional', default=True, type=bool)
    parser.add_argument('--num_workers', type=int, default=0, help='num workers of the Data Loader')
    parser.add_argument('--path_for_train', type=str, default='E:\dataset(fixed point)/data_for_training/')
    parser.add_argument('--path_for_test', type=str, default='E:\dataset(fixed point)/data_for_test/')
    parser.add_argument("--channels", type=int, default=1, help="input images channels number")
    parser.add_argument("--dataset", type=str, default='LFSSR_HI')

    # SR参数
    parser.add_argument('--SR_factor', default=4, type=int, help='4 or 8')
    parser.add_argument('--angRes', default=5, type=int, help='angle number')
    parser.add_argument("--patch_size", type=int, default=24, help="patch size for train")
    parser.add_argument("--stide_size", type=int, default=12, help="patch size for train")

    # other 参数
    parser.add_argument("--sampling_timesteps", type=int, default=5,
                        help="Number of implicit sampling steps for validation image patches")
    parser.add_argument("--image_folder", default='results/images/', type=str,
                        help="Location to save restored validation image patches")
    parser.add_argument('--seed', default=66, type=int, metavar='N',
                        help='Seed for initializing training (default: 61)')
    parser.add_argument('--device', type=str, default='cuda:0')

    #
    parser.add_argument('--resume', default='./results/ckpts/LFSSR_HI_ddpm_120.pth.tar', type=str,
                        help='Path for checkpoint to load and resume')
    parser.add_argument('--n_steps', type=int, default=25, help='number of epochs to update learning rate')
    parser.add_argument('--gamma', type=float, default=0.5, help='gamma')
    parser.add_argument('--val', type=bool, default=False)

    # 对比损失
    # parser.add_argument('--use_neg', type=bool, default=True)
    # parser.add_argument('--mcl_neg', type=int, default=4)
    # parser.add_argument('--neg_sr', type=bool, default=True)
    # parser.add_argument('--use_ema', type=bool, default=True)
    # parser.add_argument('--cl_layer', type=str, default='0,2,4')
    # parser.add_argument('--cl_loss_type', type=str, default='InfoNCE_L1')
    # parser.add_argument('--pos_id', type=int, default=-1)
    # parser.add_argument('--shuffle_neg', default='true',
    #                     help='Used in CLD in loss/discriminator.py, shuffle idx in batch')


    # model参数
    parser.add_argument('--in_channels', type=int, default=1)
    parser.add_argument('--out_ch', type=int, default=1)
    parser.add_argument('--ch', type=int, default=24)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--ema_rate',type=float , default=0.999)
    parser.add_argument('--ema', type=bool, default=True)
    parser.add_argument('--resamp_with_conv', type=bool, default=True)

    # training
    parser.add_argument('--n_epochs', type=int, default=200)
    parser.add_argument('--save_freq', type=int, default=1)
    parser.add_argument('--training_batch_size', type=int, default=1)

    # sampling
    parser.add_argument('--sampling_batch_size', type=int, default=1)
    parser.add_argument('--sampling_last_only', type=bool, default=True)

    # optim
    parser.add_argument('--weight_decay', type=float, default=0.000)
    parser.add_argument('--optimizer', type=str, default='Adam')
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--eps', type=float, default=0.00000001)
    parser.add_argument('--amsgrad', type=bool, default=False)

    # diffusion参数
    parser.add_argument('--beta_schedule', type=str, default='linear')
    parser.add_argument('--beta_start',type=float , default=0.0001)
    parser.add_argument('--beta_end', type=float, default= 0.02)
    parser.add_argument('--num_diffusion_timesteps', type=int, default=1000)

    # 保存路径
    parser.add_argument('--save_dir', type=str, default='./results/')


    args = parser.parse_args()

    return args