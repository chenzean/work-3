import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
# from models.VSSM import VSSBlock
from models.SAS_conv import SAS_Conv2D


def get_timestep_embedding(timesteps, embedding_dim):
    """
    This matches the implementation in Denoising Diffusion Probabilistic Models:
    From Fairseq.
    Build sinusoidal embeddings.
    This matches the implementation in tensor2tensor, but differs slightly
    from the description in Section 3.5 of "Attention Is All You Need".
    """
    assert len(timesteps.shape) == 1

    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
    emb = emb.to(device=timesteps.device)
    emb = timesteps.float()[:, None] * emb[None, :]     # [50, 32]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)        # [50, 64]
    if embedding_dim % 2 == 1:  # zero pad
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    return emb


def Normalize(in_channels):
    return torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)


class AttnBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.k = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.v = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.proj_out = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b, c, h, w = q.shape
        q = q.reshape(b, c, h*w)
        q = q.permute(0, 2, 1)   # b,hw,c
        k = k.reshape(b, c, h*w)  # b,c,hw
        w_ = torch.bmm(q, k)     # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        w_ = w_ * (int(c)**(-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = v.reshape(b, c, h*w)
        w_ = w_.permute(0, 2, 1)   # b,hw,hw (first hw of k, second of q)
        # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
        h_ = torch.bmm(v, w_)
        h_ = h_.reshape(b, c, h, w)

        h_ = self.proj_out(h_)

        return x+h_


class ResnetBlock(nn.Module):
    def __init__(self, args, *,
                 in_channels,
                 out_channels=None,
                 conv_shortcut=False,
                 dropout=0,
                 temb_channels=96):
        super().__init__()
        self.args = args
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        # self.norm1 = Normalize(in_channels)
        self.conv1 = torch.nn.Conv2d(in_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)

        self.temb_proj = torch.nn.Linear(temb_channels,
                                         in_channels)


        # self.norm2 = Normalize(out_channels)
        self.dropout = torch.nn.Dropout(dropout)


        # self.conv2_spa = VSSBlock(out_channels, out_channels)
        # self.conv2_ang = VSSBlock(out_channels, out_channels)
        # self.conv2_spa = AttnBlock(out_channels)
        # self.conv2_spa = AttnBlock(out_channels)
        self.conv2_spa = nn.Conv2d(out_channels, out_channels,3,1,1)
        self.conv2_ang = nn.Conv2d(out_channels, out_channels,3,1,1)

        # self.ACGFM = Sea_Attention(out_channels)
        self.ACGFM = nn.Sequential(
            nn.Conv2d(out_channels * 2,out_channels, 1),
            nn.Conv2d(out_channels, out_channels, 3,1,1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_channels, out_channels, 3,1,1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_channels, out_channels, 3,1,1),
        )

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = torch.nn.Conv2d(in_channels,
                                                     out_channels,
                                                     kernel_size=3,
                                                     stride=1,
                                                     padding=1)
            else:
                self.nin_shortcut = torch.nn.Conv2d(in_channels,
                                                    out_channels,
                                                    kernel_size=1,
                                                    stride=1,
                                                    padding=0)

    def forward(self, x, ref, temb):
        '''
        Args:
            x:
            temb:           [50, 256]
            angular_temb:   [480,480,2,64]

        Returns:

        '''
        b, c,height,width = x.shape

        h = x
        # print(h.shape)
        # h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)
        # print(h.shape)
        # print(temb.shape)
        # print(self.temb_proj(nonlinearity(temb)).shape)

        '''
        # h 的维度是             [50, 64, 96, 96]
        # temb 的维度是          [50, 256]
        self.temb_proj(nonlinearity(temb))          [50, 64]
        # self.temb_proj(nonlinearity(temb))[:, :, None, None]      [50, 64, 1, 1]
        # angular_temb 的维度是  [50, 2, 256]
        self.angular_proj(nonlinearity(angular_temb))    [50, 2, 64]
        '''

        # 时间t嵌入
        h = h + self.temb_proj(nonlinearity(temb))[:, :, None, None]
        if ref !=None:
            # h = self.ACGFM(h, ref)

            h = torch.cat([h, ref], dim=1)
            h = self.ACGFM(h)

        # h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2_spa(h)
        h = nonlinearity(h)
        h = rearrange(h, '(b an1 an2) c h w->(b h w) c an1 an2', an1=self.args.angRes, an2=self.args.angRes,h=height,w=width)
        h = self.conv2_ang(h)
        h = rearrange(h, '(b h w) c an1 an2->(b an1 an2) c h w', an1=self.args.angRes, an2=self.args.angRes,h=height,w=width)



        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x+h

class Upsample(nn.Module):
    def __init__(self, in_channels,out_channels, with_conv=True):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = torch.nn.Conv2d(in_channels,
                                        out_channels,
                                        kernel_size=1)
        self.up_conv = nn.ConvTranspose2d(in_channels=out_channels,
                                                 out_channels=out_channels,
                                                 kernel_size=3,
                                                 stride=2,
                                                 padding=1,
                                                 output_padding=1)

        self.conv_1x1 = nn.Conv2d(out_channels, out_channels,1)

    def forward(self, x):
        if self.with_conv:
            x = self.conv(x)
        x = self.up_conv(x)
        x = nonlinearity(x)
        x = self.conv_1x1(x)
        return x


class Downsample(nn.Module):
    def __init__(self, in_channels,out_channels, with_conv = True):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=3,
                                        padding=1)

        self.down_conv = nn.Conv2d(in_channels=in_channels,
                                   out_channels=out_channels,
                                   kernel_size=3,
                                   stride=2,
                                   padding=1)

        self.conv_1x1 = nn.Conv2d(out_channels, out_channels,1)

    def forward(self, x):
        if self.with_conv:
            x = self.conv(x)
        x = self.down_conv(x)
        x = nonlinearity(x)
        x = self.conv_1x1(x)
        return x

class AttnBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.k = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.v = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.proj_out = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b, c, h, w = q.shape
        q = q.reshape(b, c, h*w)
        q = q.permute(0, 2, 1)   # b,hw,c
        k = k.reshape(b, c, h*w)  # b,c,hw
        w_ = torch.bmm(q, k)     # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        w_ = w_ * (int(c)**(-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = v.reshape(b, c, h*w)
        w_ = w_.permute(0, 2, 1)   # b,hw,hw (first hw of k, second of q)
        # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
        h_ = torch.bmm(v, w_)
        h_ = h_.reshape(b, c, h, w)

        h_ = self.proj_out(h_)

        return x+h_


def nonlinearity(x):
    # swish
    return x*torch.sigmoid(x)


class DiffusionUNet(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.angRes = args.angRes
        self.patch_size = args.image_size
        ch, out_ch = args.ch, args.out_ch

        in_channels = args.in_channels * 2 if args.conditional else args.in_channels
        # 提取了数据集中图像的分辨率（或图像大小），它将用于确定模型的输入分辨率
        self.resolution = args.image_size

        self.Encoder = Encoder(args)

        self.ch = ch                             # 64
        self.temb_ch = self.ch * 4               # 256
        self.in_channels = in_channels           # 1

        # timestep embedding
        # 该组件包含了多个线性层，用于将输入特征（可能与时间相关）映射到一个新的特征空间，以便模型可以更好地理解和处理与时间相关的信息
        self.temb = nn.Module()
        self.temb.dense = nn.ModuleList([
            torch.nn.Linear(self.ch,
                            self.temb_ch),
            torch.nn.Linear(self.temb_ch,
                            self.temb_ch),
        ])

        # init_conv
        self.conv_in = torch.nn.Conv2d(in_channels,
                                       self.ch,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)
        # encoder
        self.spa_ang_RB_1 = ResnetBlock(args, in_channels=self.ch, out_channels=self.ch, conv_shortcut=True)
        self.spa_ang_RB_2 = ResnetBlock(args, in_channels=self.ch * 2,out_channels=self.ch * 2, conv_shortcut=True)
        self.spa_ang_RB_3 = ResnetBlock(args, in_channels=self.ch * 3,out_channels=self.ch * 3, conv_shortcut=True)
        self.spa_ang_RB_4 = ResnetBlock(args, in_channels=self.ch * 4,out_channels=self.ch * 4, conv_shortcut=True)

        # middle

        # decoder
        self.spa_ang_RB_5 = ResnetBlock(args, in_channels=self.ch * 4,out_channels=self.ch * 4, conv_shortcut=True)
        self.spa_ang_RB_6 = ResnetBlock(args, in_channels=self.ch * 3,out_channels=self.ch * 3, conv_shortcut=True)
        self.spa_ang_RB_7 = ResnetBlock(args, in_channels=self.ch* 2,out_channels=self.ch * 2, conv_shortcut=True)
        self.spa_ang_RB_8 = ResnetBlock(args, in_channels=self.ch ,out_channels=self.ch, conv_shortcut=True)

        # downsampling
        self.down1 = Downsample(self.ch, self.ch * 2)
        self.down2 = Downsample(self.ch * 2, self.ch * 3)
        self.down3 = Downsample(self.ch * 3, self.ch * 4)

        # upsampling
        self.up1 = Upsample(self.ch * 4, self.ch * 3)
        self.up2 = Upsample(self.ch * 3, self.ch * 2)
        self.up3 = Upsample(self.ch * 2, self.ch)

        # self.sas = SAS_Conv2D(self.ch, args)
        self.conv_out = nn.Conv2d(self.ch, 1,3,1,1)

        self.conv_1 = nn.Conv2d(in_channels=self.ch * 6 ,out_channels=self.ch * 3, kernel_size=1)
        self.conv_2 = nn.Conv2d(in_channels=self.ch * 4 ,out_channels=self.ch * 2, kernel_size=1)
        self.conv_3 = nn.Conv2d(in_channels=self.ch * 2,out_channels=self.ch , kernel_size=1)



        # end

    def forward(self, x, t):
        assert x.shape[2] == x.shape[3] == self.resolution

        # get data
        image_add_noise = x[:,0,:,:]
        lr = x[:,1,:,:]
        ref = x[:,2,:,:]

        ref = ref[:,None,:,:]
        lr = lr[:,None,:,:]
        image_add_noise = image_add_noise[:,None,:,:]

        input_ = torch.cat([image_add_noise, lr], dim=1)

        # timestep embedding
        temb = get_timestep_embedding(t, self.ch)   # temb [50,64]
        temb = self.temb.dense[0](temb)
        temb = nonlinearity(temb)
        temb = self.temb.dense[1](temb)


        # 编码ref， LR

        ref_fea0, ref_fea1, ref_fea2, ref_fea3 = self.Encoder(ref)      # [50.64，96，96]
        # fea.append(ref_fea0)        # [25，32, 96, 96]
        # fea.append(ref_fea1)        # [25，64, 48, 48]
        # fea.append(ref_fea2)        # [25，128, 24, 24]
        # fea.append(ref_fea3)        # [25，256, 12, 12]

        lr_fea = self.conv_in(input_)   # [25，32, 96, 96]
        # down
        fea0 = self.spa_ang_RB_1(lr_fea, ref_fea0, temb) # [25，32, 96, 96]
        fea1 = self.down1(fea0)     # [25，64, 48, 48]
        # print(fea1.shape)
        fea2_1 = self.spa_ang_RB_2(fea1, ref_fea1, temb)  # [25，64, 48, 48]
        fea2 = self.down2(fea2_1)     # [25，64, 24, 24]
        fea3_1 = self.spa_ang_RB_3(fea2, ref_fea2, temb)  # [25，128, 24, 24]
        fea3 = self.down3(fea3_1)     # [25，256, 12, 12]
        fea4 = self.spa_ang_RB_4(fea3, ref_fea3, temb)   # [25，256, 12, 12]

        # middle


        # up
        fea5 = self.spa_ang_RB_5(fea4, ref=None,temb=temb)  # [25，256, 12, 12]
        fea5 = self.up1(fea5)
        fea5 = torch.cat([fea5, fea3_1],dim=1)
        fea5 = self.conv_1(fea5)
        fea6 = self.spa_ang_RB_6(fea5, ref=None,temb=temb)  # [25，128, 24, 24]
        fea6 = self.up2(fea6)
        fea6 = torch.cat([fea6, fea2_1],dim=1)
        fea6 = self.conv_2(fea6)
        fea7 = self.spa_ang_RB_7(fea6, ref=None,temb=temb)  # [25，64, 48, 48]
        fea7 = self.up3(fea7)

        fea7 = torch.cat([fea7, fea0],dim=1)
        fea7 = self.conv_3(fea7)
        fea8 = self.spa_ang_RB_8(fea7, ref=None,temb=temb)  # [25，32, 96, 96]

        # feaout = self.sas(fea8)

        out = self.conv_out(fea8)

        return out

class Encoder(nn.Module):
    def __init__(self, args):
        super(Encoder, self).__init__()
        self.args = args
        self.channels = args.ch

        self.conv_init = nn.Conv2d(1, self.channels, kernel_size=3, stride=1, padding=1)

        self.conv = nn.Sequential(
            nn.Conv2d(self.channels, self.channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.channels, self.channels, kernel_size=3, stride=1, padding=1),
            # VSSBlock(self.channels, self.channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.channels, self.channels, kernel_size=3, stride=1, padding=1),
        )
        # down 1
        self.conv_down_1 = nn.Sequential(
            nn.Conv2d(self.channels, self.channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.channels, self.channels, kernel_size=3, stride=1, padding=1),
            # VSSBlock(self.channels, self.channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.channels, self.channels * 2, kernel_size=3, stride=2, padding=1),
        )

        self.conv_down_2 = nn.Sequential(
            nn.Conv2d(self.channels * 2, self.channels * 2, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.channels * 2, self.channels * 2, kernel_size=3, stride=1, padding=1),
            # VSSBlock(self.channels * 2, self.channels * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.channels * 2, self.channels * 3, kernel_size=3, stride=2, padding=1),
        )

        self.conv_down_3 = nn.Sequential(
            nn.Conv2d(self.channels * 3, self.channels * 3, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.channels * 3, self.channels * 3, kernel_size=3, stride=1, padding=1),
            # VSSBlock(self.channels * 3, self.channels * 3),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.channels * 3, self.channels * 4, kernel_size=3, stride=2, padding=1),
        )

    def forward(self, x):
        x_fea_init = self.conv_init(x)
        x_fea0 = self.conv(x_fea_init)
        x_fea1 = self.conv_down_1(x_fea0)
        x_fea2 = self.conv_down_2(x_fea1)
        x_fea3 = self.conv_down_3(x_fea2)
        return x_fea0, x_fea1, x_fea2, x_fea3




class SqueezeAxialPositionalEmbedding(nn.Module):
    def __init__(self, dim, shape):
        super().__init__()

        self.pos_embed = nn.Parameter(torch.randn([1, dim, shape]))

    def forward(self, x):
        B, C, N = x.shape
        x = x + F.interpolate(self.pos_embed, size=(N), mode='linear', align_corners=False)

        return x

class Sea_Attention(torch.nn.Module):
    def __init__(self, dim, key_dim=32, num_heads=4,
                 attn_ratio=2,
                 # activation=nn.LeakyReLU,
                 norm_cfg=dict(type='BN', requires_grad=True), ):
        super().__init__()
        self.num_heads = num_heads
        self.scale = key_dim ** -0.5
        self.key_dim = key_dim
        self.nh_kd = nh_kd = key_dim * num_heads  # 32
        self.d = int(attn_ratio * key_dim)
        self.dh = int(attn_ratio * key_dim) * num_heads
        self.attn_ratio = attn_ratio

        self.to_q1 = nn.Conv2d(dim, nh_kd, 1)
        self.to_q2 = nn.Conv2d(dim, nh_kd, 1)
        self.to_k1 = nn.Conv2d(dim, nh_kd, 1)
        self.to_k2 = nn.Conv2d(dim, nh_kd, 1)
        self.to_v1 = nn.Conv2d(dim, self.dh, 1)
        self.to_v2 = nn.Conv2d(dim, self.dh, 1)
        self.to_v3 = nn.Conv2d(dim, self.dh, 1)

        self.proj = torch.nn.Sequential(nn.LeakyReLU(0.2, inplace=True),
                                        nn.Conv2d(dim, dim, 3,1,1)
                                        )

        self.proj_encode_row = torch.nn.Sequential(nn.LeakyReLU(0.2, inplace=True),
                                                   nn.Conv2d(self.dh, dim, 1)
                                                   )
        self.pos_emb_rowq = SqueezeAxialPositionalEmbedding(nh_kd, 16)
        self.pos_emb_rowk = SqueezeAxialPositionalEmbedding(nh_kd, 16)

        self.proj_encode_column = torch.nn.Sequential(nn.LeakyReLU(0.2, inplace=True),
                                                      nn.Conv2d(self.dh, dim, 1))
        self.pos_emb_columnq = SqueezeAxialPositionalEmbedding(nh_kd, 16)
        self.pos_emb_columnk = SqueezeAxialPositionalEmbedding(nh_kd, 16)

        self.dwconv = nn.Conv2d(self.dh, self.dh, 1)

        self.act = nn.LeakyReLU(0.2, inplace=True)
        self.pwconv = nn.Conv2d(self.dh, dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, lr, ref):
        B, C, H, W = lr.shape

        lr_q1 = self.to_q1(lr)
        lr_q2 = self.to_q2(lr)

        ref_k1 = self.to_k1(ref)
        ref_k2 = self.to_k2(ref)

        ref_v1 = self.to_v1(ref)
        ref_v2 = self.to_v2(ref)
        ref_v3 = self.to_v3(ref)

        # detail enhance
        ref_v3 = self.act(self.dwconv(ref_v3))
        ref_v3 = self.pwconv(ref_v3)

        # squeeze axial attention
        ## squeeze row
        qrow = self.pos_emb_rowq(lr_q1.mean(-1)).reshape(B, self.num_heads, -1, H).permute(0, 1, 3, 2)
        krow = self.pos_emb_rowk(ref_k1.mean(-1)).reshape(B, self.num_heads, -1, H)
        vrow = ref_v1.mean(-1).reshape(B, self.num_heads, -1, H).permute(0, 1, 3, 2)
        attn_row = torch.matmul(qrow, krow) * self.scale
        attn_row = attn_row.softmax(dim=-1)
        xx_row = torch.matmul(attn_row, vrow)  # B nH H C
        xx_row = self.proj_encode_row(xx_row.permute(0, 1, 3, 2).reshape(B, self.dh, H, 1))

        ## squeeze column
        qcolumn = self.pos_emb_columnq(lr_q2.mean(-2)).reshape(B, self.num_heads, -1, W).permute(0, 1, 3, 2)
        kcolumn = self.pos_emb_columnk(ref_k2.mean(-2)).reshape(B, self.num_heads, -1, W)
        vcolumn = ref_v2.mean(-2).reshape(B, self.num_heads, -1, W).permute(0, 1, 3, 2)
        attn_column = torch.matmul(qcolumn, kcolumn) * self.scale
        attn_column = attn_column.softmax(dim=-1)
        xx_column = torch.matmul(attn_column, vcolumn)  # B nH W C
        xx_column = self.proj_encode_column(xx_column.permute(0, 1, 3, 2).reshape(B, self.dh, 1, W))

        xx = xx_row.add(xx_column)
        # xx = ref_v2.add(xx)
        xx = self.proj(xx)

        xx = self.sigmoid(xx) * ref_v3

        return xx


from thop import profile
# from torchinfo import summary
if __name__ == '__main__':
    from config import parse_args_and_config
    args = parse_args_and_config()
    from models.ddm import *
    data = torch.randn([25, 32, 96, 96]).cuda()
    betas = get_beta_schedule(
        beta_schedule=args.beta_schedule,
        beta_start=args.beta_start,
        beta_end=args.beta_end,
        num_diffusion_timesteps=args.num_diffusion_timesteps,
    )
    betas = torch.from_numpy(betas).float().to('cuda:0')
    num_timesteps = betas.shape[0]
    t = torch.randint(low=0, high=num_timesteps, size=(25 // 2 + 1,)).to('cuda:0')
    # 根据batchsize设置t
    t = torch.cat([t, num_timesteps - t - 1], dim=0)[:25]
    # print(t.shape)
    model = DiffusionUNet(args).cuda()
    summary(model, input_size=[(25, 32, 96, 96),(25,)])
    flops, params = profile(model, (data,t, ))
    print("flops: ", flops / 1e9, "params: ", params / 1e6)
