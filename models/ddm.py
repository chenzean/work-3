import time
#
# import fastjsonschema.indent
import tqdm
import torch.nn as nn
import torch.backends.cudnn as cudnn
import utils
from models.LF_UNet import DiffusionUNet
from models.restoration import DiffusiveRestoration
from tqdm import tqdm
import random
from loss.lpips.loss import LPIPS
from torchvision import models
from models.untils_models import *
from models.loss import *
from utils.logging import *



# This script is adapted from the following repositories
# https://github.com/ermongroup/ddim
# https://github.com/bahjat-kawar/ddrm



def noise_estimation_loss(model, residual_labels, label, ref, lr_lf, t, noise, b):
    '''
    Args:
        model: 该模型用于预测噪声
        x0: 表示输入数据，通常是图像数据
        t: 采样数
        e: 表示真实的噪声数据，与输入数据 x0 具有相同的形状，通常是模型训练的目标。
        b: 表示 beta 值，用于控制数据逐渐淡化和变化的速度。

    Returns:
        rec_loss        L1重建损失
        detail_loss     SSIM损失

    '''
    '''
     # 对比损失
     MCL = VGGInfoNCE(args)
    a = (1-b).cumprod(dim=0).index_select(0, t).view(-1, 1, 1, 1)       # 根据时间步长张量 t 从上一步计算的累积乘积张量中选择特定的元素
    # 加噪过程
    x = x0 * a.sqrt() + e * (1.0 - a).sqrt()   # 加噪后的图像
    # plt.imshow(x[0,0,:,:].cpu().detach().numpy())
    # plt.show()
    input = torch.cat([x, lr, ref], dim=1)
    # 预测噪声
    output = model(input, t.float(), ref_coord)
    # noise_loss = (e - output).square().sum(dim=(1, 2, 3)).mean(dim=0)


    noise_loss = L1_Charbonnier_loss(e, output)
    x0_t = (x - output * (1 - a).sqrt()) / a.sqrt()
    # x0_t_cpu = x0_t.cpu()
    # gt_cpu = gt.cpu()
    # lr_cpu = lr.cpu()

    # 预测的图像去计算对比损失
    contrastiveloss = MCL(model, input, t.float(), x0_t, gt, lr, ref_coord)
    recloss = L1_Charbonnier_loss(x0_t, gt)

    print(noise_loss)
    print(10 *contrastiveloss)
    print(0.1 * recloss)
    total_loss = noise_loss + 10 * contrastiveloss + 0.1 * recloss

    return total_loss
    '''
    a = (1-b).cumprod(dim=0).index_select(0, t).view(-1, 1, 1, 1)       # 根据时间步长张量 t 从上一步计算的累积乘积张量中选择特定的元素
    # 加噪过程
    residual_labels_add_noise = residual_labels * a.sqrt() + noise * (1.0 - a).sqrt()   # 加噪后的图像

    # 预测图像
    output = model(torch.cat([residual_labels_add_noise, lr_lf, ref], dim=1), t.float())

    rec_loss = L1_Charbonnier_loss(residual_labels, output)

    lr_output = output + lr_lf
    detail_loss = ssim_loss(label, lr_output)

    return rec_loss, detail_loss



class DenoisingDiffusion(object):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.device = args.device

        self.model = DiffusionUNet(args)

        self.model.to(self.device)

        self.ema_helper = EMAHelper()
        self.ema_helper.register(self.model)
        self.optimizer = utils.optimize.get_optimizer(self.args, self.model.parameters())
        self.start_epoch, self.step = 1, 0

        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=args.n_steps, gamma=args.gamma)



        # set beta
        betas = get_beta_schedule(
            beta_schedule=args.beta_schedule,
            beta_start=args.beta_start,
            beta_end=args.beta_end,
            num_diffusion_timesteps=args.num_diffusion_timesteps,
        )

        betas = self.betas = torch.from_numpy(betas).float().to(self.device)
        self.num_timesteps = betas.shape[0]



    def load_ddm_ckpt(self, load_path, ema=False):
        checkpoint = load_checkpoint(load_path, None)
        self.start_epoch = checkpoint['epoch'] + 1
        self.model.load_state_dict(checkpoint['state_dict'], strict=True)
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.scheduler.load_state_dict(checkpoint['scheduler'])
        self.ema_helper.load_state_dict(checkpoint['ema_helper'])
        if ema:
            self.ema_helper.ema(self.model)
        print("=> loaded checkpoint '{}' (epoch {})".format(load_path, checkpoint['epoch']))

    def train(self, DATASET):
        cudnn.benchmark = True
        train_loader, test_loader = DATASET.get_loaders()
        save_dir = self.args.save_dir
        log_dir, checkpoints_dir, val_dir,mat_file = create_dir(save_dir)

        ''' Logger '''
        logger = Logger(log_dir, self.args)

        ' load ckpt'
        if os.path.isfile(self.args.resume):
            self.load_ddm_ckpt(self.args.resume)

        'Number of parameter'
        total = sum([param.nelement() for param in self.model.parameters()])
        print("Number of parameter: %.2fM" % (total / 1e6))

        '''start training'''
        for epoch in range(self.start_epoch, self.args.n_epochs):
            logger.log_string('\n epoch is %d'% epoch)
            data_start = time.time()
            data_time = 0
            ave_loss = 0.
            for i, (x, y, ref, data_info) in tqdm(enumerate(train_loader), total=len(train_loader), ncols=70):
                '''
                input: 
                x       [1,1,120,120]
                y       [1,1,480,480]
                ref     [1,1,96,96]
                '''
                # 可视化
                # plt.imshow(x[0,0,:,:].cpu().detach().numpy(), cmap='gray')
                # plt.show()
                # plt.imshow(y[0,0,:,:].cpu().detach().numpy(), cmap='gray')
                # plt.show()
                # plt.imshow(ref[0,0,:,:].cpu().detach().numpy(), cmap='gray')
                # plt.show()
                # ###########################################################
                self.step = self.step + 1
                self.model.train()
                data_time += time.time() - data_start

                b,C,H,W = y.shape
                h = H//self.args.angRes
                w = W // self.args.angRes

                lr_lf = rearrange(x, 'b c (an1 h) (an2 w) ->(b an1 an2) c h w',an1=self.args.angRes,an2=self.args.angRes,h=h//self.args.SR_factor,w=w//self.args.SR_factor)
                label = rearrange(y, 'b c (an1 h) (an2 w) ->(b an1 an2) c h w',an1=self.args.angRes,an2=self.args.angRes,h=h,w=w)

                # 上采样
                lr_lf = F.interpolate(lr_lf, scale_factor=self.args.SR_factor, mode='bicubic')

                # 获取残差标签
                residual_labels = label - lr_lf
                n = residual_labels.size(0)

                ref_num = n//self.args.training_batch_size
                ref = ref.repeat(ref_num,1,1,1).to(self.device)

                '''move to cuda'''
                # ref
                ref = ref.to(self.device)
                # GT
                label = label.to(self.device)
                # lr
                lr_lf = lr_lf.to(self.device)
                # label res
                residual_labels = residual_labels.to(self.device)

                # noise
                noise = torch.randn_like(residual_labels[:, :, :, :])        # [25,1,96,96]

                # 设置的betas值
                b = self.betas

                # antithetic sampling
                t = torch.randint(low=0, high=self.num_timesteps, size=(n // 2 + 1,)).to(self.device)
                t = torch.cat([t, self.num_timesteps - t - 1], dim=0)[:n]


                l1_loss, ssim_loss = noise_estimation_loss(self.model, residual_labels, label, ref, lr_lf, t, noise, b)

                total_loss = ssim_loss * 0.1 + l1_loss

                ave_loss += total_loss.item()
                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()
                self.ema_helper.update(self.model)

            ''' scheduler '''
            self.scheduler.step()

            '''save checkpoints'''
            if epoch % self.args.save_freq == 0:
                save_checkpoint({
                    'epoch': epoch,
                    'state_dict': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'ema_helper': self.ema_helper.state_dict(),
                    'scheduler': self.scheduler.state_dict(),
                    'params': self.args,
                }, filename=os.path.join(self.args.save_dir, 'ckpts', self.args.dataset + '_ddpm_' + str(epoch)))

                logger.log_string(
                    "epoch: %.3d, loss: %.5f,  learning_rate: %e" % (
                        epoch, ave_loss/len(train_loader), self.optimizer.state_dict()['param_groups'][0]['lr']))



    # 用于在扩散模型中生成图像数据的样本
    def sample_image(self, x_cond, noise, last=True, patch_locs=None, patch_size=None):
        '''
        self.config.diffusion.num_diffusion_timesteps   表示总的扩散时间步数
        self.args.sampling_timesteps 表示采样的时间步数
        skip 的计算用于确定在哪些时间步上进行采样。
        '''
        skip = self.args.num_diffusion_timesteps // self.args.sampling_timesteps
        # 创建了一个时间步序列 seq，其中包含了需要进行采样的时间步。采样的时间步是根据 skip 计算的，从0开始，以skip为间隔，直到达到总的扩散时间步数
        seq = range(0, self.args.num_diffusion_timesteps, skip)
        xs = utils.sampling.generalized_steps(noise, x_cond, seq, self.model, self.betas, eta=0.)
        if last:
            xs = xs[0][-1]
        return xs
    
    def sample_validation_patches(self, val_loader, step):
        image_folder = os.path.join(self.args.image_folder, self.args.dataset + str(self.args.image_size))
        with torch.no_grad():
            print(f"Processing a single batch of validation images at step: {step}")
            for i, (Lr_SAI_y, Hr_SAI_y, Sr_SAI_cbcr, data_info, LF_name) in enumerate(
                    val_loader):  # Lr_SAI_y, Hr_SAI_y, Sr_SAI_cbcr

                Lr_SAI_y = Lr_SAI_y.squeeze()
                ''' Crop LFs into Patches '''
                subLFin = LFdivide(Lr_SAI_y, 5, 32, 16)
                numU, numV, H, W = subLFin.size()
                subLFin = rearrange(subLFin, 'n1 n2 a1h a2w -> (n1 n2) 1 a1h a2w')
                subLFout = torch.zeros(numU * numV, 1, 640, 640)
                ''' SR the Patches '''

                for i in range(0, numU * numV, 1):
                    tmp = subLFin[i:min(i + 1, numU * numV), :, :, :]  # [2,1,160,160]

                    n = tmp.size(0)
                    # x的条件
                    x_cond = tmp[:, :1, :, :].to(self.device)
                    x_cond = data_transform(x_cond)
                    x = torch.randn(n, 1, self.args.image_size, self.args.image_size, device=self.device)
                    x = self.sample_image(x_cond, x)

                    x = inverse_data_transform(x)
                    x_cond = inverse_data_transform(x_cond)

                    for i in range(n):
                        save_image(x_cond[i], os.path.join(image_folder, str(step), f"{i}_cond.png"))
                        save_image(x[i], os.path.join(image_folder, str(step), f"{i}.png"))


