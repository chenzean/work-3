import torch
import torch.nn as nn
import utils
import torchvision
import os
from models.untils_models import *
import imageio
from scipy.io import savemat
from models.untils_models import *
from tqdm import tqdm



class DiffusiveRestoration:
    def __init__(self, diffusion, args):
        super(DiffusiveRestoration, self).__init__()
        self.args = args
        self.val_flag = self.args.val
        self.diffusion = diffusion
        if self.val_flag == False:
            if os.path.isfile(args.resume):
                self.diffusion.load_ddm_ckpt(args.resume, ema=True)
                self.diffusion.model.eval()
            else:
                print('Pre-trained diffusion model path is missing!')

    def restore(self, test_loader, validation='LFSSR_HI', epoch_folder=None, logger=None):

        LF_iter_test = []
        psnr_iter_test = []
        ssim_iter_test = []

        psnr_iter_test_allviews=[]
        ssim_iter_test_allviews=[]


        with torch.no_grad():
            for i, (Lr_SAI_y, Hr_SAI_y, ref, Sr_SAI_cbcr, data_info, LF_name) in tqdm(enumerate(test_loader), total=len(test_loader), ncols=70):     # Lr_SAI_y, Hr_SAI_y, Sr_SAI_cbcr
                # print(f"starting processing from image {LF_name}")
                Hr_y = Hr_SAI_y.squeeze()
                Lr_SAI_y = Lr_SAI_y.squeeze()
                ref = ref.squeeze()

                ''' Crop LFs into Patches '''
                subLFin = LFdivide(Lr_SAI_y, self.args.angRes, self.args.patch_size, self.args.stide_size)
                subLFlabel = LFdivide(Hr_y, self.args.angRes, self.args.patch_size * self.args.SR_factor, self.args.stide_size * self.args.SR_factor)
                ref_in = LFdivide(ref, 1, self.args.patch_size * self.args.SR_factor, self.args.stide_size * self.args.SR_factor)
                numU, numV, H, W = subLFin.size()

                h = H//self.args.angRes
                w = W//self.args.angRes
                subLFin = rearrange(subLFin, 'n1 n2 a1h a2w -> (n1 n2) 1 a1h a2w')
                subLFlabel = rearrange(subLFlabel, 'n1 n2 a1h a2w -> (n1 n2) 1 a1h a2w')
                ref_in = rearrange(ref_in, 'n1 n2 a1h a2w -> (n1 n2) 1 a1h a2w')
                subLFout = torch.zeros(numU * numV, 1, H * self.args.SR_factor, W * self.args.SR_factor)

                ''' SR the Patches '''
                for i in range(0, numU * numV, 1):
                    lr = subLFin[i:min(i + 1, numU * numV), :, :, :]       # [1,1,160,160]
                    gt = subLFlabel[i:min(i + 1, numU * numV), :, :, :]
                    ref = ref_in[i:min(i + 1, numU * numV), :, :, :]
                    # tmp是子孔径阵列
                    lr = rearrange(lr,'b c (an1 h) (an2 w)->(b an1 an2) c h w', an1=self.args.angRes, an2=self.args.angRes)

                    # bicubic-up
                    lr_up = F.interpolate(lr, scale_factor=self.args.SR_factor, mode='bicubic')
                    gt = rearrange(gt,'b c (an1 h) (an2 w)->(b an1 an2) c h w', an1=self.args.angRes, an2=self.args.angRes)

                    b,_,_,_ = gt.shape
                    ref_num = b // self.args.sampling_batch_size
                    ref = ref.repeat(ref_num, 1, 1, 1)

                    # LR image
                    lr_up_skip = lr_up
                    lr_up = lr_up[:, :, :, :].to(self.args.device)
                    ref = ref[:, :, :, :].to(self.args.device)

                    x_output = self.diffusive_restoration(ref, lr_up)

                    output = lr_up_skip + x_output

                    x_output = rearrange(output,'(b an1 an2) c h w->b c (an1 h) (an2 w)', an1=self.args.angRes, an2=self.args.angRes)
                    subLFout[i:min(i + 2, numU * numV), :, :, :] = x_output

                subLFout = rearrange(subLFout, '(n1 n2) 1 a1h a2w -> n1 n2 a1h a2w', n1=numU, n2=numV)

                ''' Restore the Patches to LFs '''
                Sr_4D_y = LFintegrate(subLFout, self.args.angRes, self.args.patch_size *self.args.SR_factor, self.args.stide_size *self.args.SR_factor,
                                      Hr_SAI_y.size(-2) // self.args.angRes,
                                      Hr_SAI_y.size(-1) // self.args.angRes)

                Sr_SAI_y = rearrange(Sr_4D_y, 'a1 a2 h w -> 1 1 (a1 h) (a2 w)')

                if epoch_folder != None:
                    save_dir_ = epoch_folder
                else:
                    save_dir_ = os.path.join(self.args.save_dir, 'Test')

                log_dir, results_dir, CenterView_dir, mat_file = create_dir(save_dir_)

                ''' Calculate the PSNR & SSIM '''
                psnr, ssim = cal_metrics(Hr_SAI_y, Sr_SAI_y)

                psnr_mean = psnr.sum() / np.sum(psnr > 0)
                ssim_mean = ssim.sum() / np.sum(ssim > 0)

                psnr_iter_test.append(psnr_mean)
                ssim_iter_test.append(ssim_mean)

                LF_iter_test.append(LF_name[0])

                psnr_iter_test_allviews.append(psnr)
                ssim_iter_test_allviews.append(ssim)

                ''' Save RGB png and mat file'''
                if results_dir is not None:
                    save_dir_ = results_dir.joinpath(LF_name[0])
                    save_dir_.mkdir(exist_ok=True)

                    Sr_SAI_ycbcr = torch.cat((Sr_SAI_y, Sr_SAI_cbcr), dim=1)
                    Sr_SAI_rgb = (ycbcr2rgb(Sr_SAI_ycbcr.squeeze().permute(1, 2, 0).numpy()).clip(0, 1) * 255).astype(
                        'uint8')
                    Sr_4D_rgb = rearrange(Sr_SAI_rgb, '(a1 h) (a2 w) c -> a1 a2 h w c', a1=5, a2=5)

                    # save the center view and .mat file
                    img = Sr_4D_rgb[5 // 2, 5 // 2, :, :, :]
                    path = str(CenterView_dir) + '/' + LF_name[0] + '_' + 'CenterView.png'
                    path_mat = str(mat_file) + '/' + LF_name[0] + '.mat'
                    Sr_SAI_y_numpy = Sr_SAI_y.numpy()
                    savemat(path_mat, {'SR': Sr_SAI_y_numpy})
                    imageio.imwrite(path, img)

                    # save all views
                    for i in range(5):
                        for j in range(5):
                            img = Sr_4D_rgb[i, j, :, :, :]
                            path = str(save_dir_) + '/' + LF_name[0] + '_' + str(i) + '_' + str(j) + '.png'
                            imageio.imwrite(path, img)
                            pass
                        pass
                    pass
                pass
            if logger != None:
               logger.log_string('ave PNSR: %.5f, ave SSIM: %.5f ' % (np.mean(psnr_iter_test),np.mean(ssim_iter_test)))

    def diffusive_restoration(self, ref, lr, r=None):
        '''
        Args:
            ref: 参考图像
            lr:  低分辨率图像
        Returns:
            输出图像
        '''

        assert len(ref.shape) == 4 and len(lr.shape) == 4, "Both tensors must be four-dimensional."
        assert ref.shape[0] == lr.shape[0], "Mismatch in dimension 0 (batch size and angular dim)."
        assert ref.shape[1] == lr.shape[1], "Mismatch in dimension 1 (channels)."
        assert ref.shape[2] == lr.shape[2], "Mismatch in dimension 2 (height)."
        assert ref.shape[3] == lr.shape[3], "Mismatch in dimension 3 (width)."

        x = torch.randn(ref.size(), device=ref.device)
        x_cond = torch.cat([lr, ref], dim=1)
        x_output = self.diffusion.sample_image(x_cond, x, patch_locs=None, patch_size=None)
        return x_output


