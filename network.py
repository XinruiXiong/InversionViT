# © 2022. Triad National Security, LLC. All rights reserved.

# This program was produced under U.S. Government contract 89233218CNA000001 for Los Alamos

# National Laboratory (LANL), which is operated by Triad National Security, LLC for the U.S.

# Department of Energy/National Nuclear Security Administration. All rights in the program are

# reserved by Triad National Security, LLC, and the U.S. Department of Energy/National Nuclear

# Security Administration. The Government is granted for itself and others acting on its behalf a

# nonexclusive, paid-up, irrevocable worldwide license in this material to reproduce, prepare

# derivative works, distribute copies to the public, perform publicly and display publicly, and to permit

# others to do so.

import torch
import torch.nn as nn
import torch.nn.functional as F
from math import ceil
from collections import OrderedDict

NORM_LAYERS = { 'bn': nn.BatchNorm2d, 'in': nn.InstanceNorm2d, 'ln': nn.LayerNorm }

# Replace the key names in the checkpoint in which legacy network building blocks are used 
def replace_legacy(old_dict):
    li = []
    for k, v in old_dict.items():
        k = (k.replace('Conv2DwithBN', 'layers')
              .replace('Conv2DwithBN_Tanh', 'layers')
              .replace('Deconv2DwithBN', 'layers')
              .replace('ResizeConv2DwithBN', 'layers'))
        li.append((k, v))
    return OrderedDict(li)

class Conv2DwithBN(nn.Module):
    def __init__(self, in_fea, out_fea, 
                kernel_size=3, stride=1, padding=1,
                bn=True, relu_slop=0.2, dropout=None):
        super(Conv2DwithBN,self).__init__()
        layers = [nn.Conv2d(in_channels=in_fea, out_channels=out_fea, kernel_size=kernel_size, stride=stride, padding=padding)]
        if bn:
            layers.append(nn.BatchNorm2d(num_features=out_fea))
        layers.append(nn.LeakyReLU(relu_slop, inplace=True))
        if dropout:
            layers.append(nn.Dropout2d(0.8))
        self.Conv2DwithBN = nn.Sequential(*layers)

    def forward(self, x):
        return self.Conv2DwithBN(x)

class ResizeConv2DwithBN(nn.Module):
    def __init__(self, in_fea, out_fea, scale_factor=2, mode='nearest'):
        super(ResizeConv2DwithBN, self).__init__()
        layers = [nn.Upsample(scale_factor=scale_factor, mode=mode)]
        layers.append(nn.Conv2d(in_channels=in_fea, out_channels=out_fea, kernel_size=3, stride=1, padding=1))
        layers.append(nn.BatchNorm2d(num_features=out_fea))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.ResizeConv2DwithBN = nn.Sequential(*layers)

    def forward(self, x):
        return self.ResizeConv2DwithBN(x)
 
class Conv2DwithBN_Tanh(nn.Module):
    def __init__(self, in_fea, out_fea, kernel_size=3, stride=1, padding=1):
        super(Conv2DwithBN_Tanh, self).__init__()
        layers = [nn.Conv2d(in_channels=in_fea, out_channels=out_fea, kernel_size=kernel_size, stride=stride, padding=padding)]
        layers.append(nn.BatchNorm2d(num_features=out_fea))
        layers.append(nn.Tanh())
        self.Conv2DwithBN = nn.Sequential(*layers)

    def forward(self, x):
        return self.Conv2DwithBN(x)

class ConvBlock(nn.Module):
    def __init__(self, in_fea, out_fea, kernel_size=3, stride=1, padding=1, norm='bn', relu_slop=0.2, dropout=None):
        super(ConvBlock,self).__init__()
        layers = [nn.Conv2d(in_channels=in_fea, out_channels=out_fea, kernel_size=kernel_size, stride=stride, padding=padding)]
        if norm in NORM_LAYERS:
            layers.append(NORM_LAYERS[norm](out_fea))
        layers.append(nn.LeakyReLU(relu_slop, inplace=True))
        if dropout:
            layers.append(nn.Dropout2d(0.8))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class ConvBlock_Tanh(nn.Module):
    def __init__(self, in_fea, out_fea, kernel_size=3, stride=1, padding=1, norm='bn'):
        super(ConvBlock_Tanh, self).__init__()
        layers = [nn.Conv2d(in_channels=in_fea, out_channels=out_fea, kernel_size=kernel_size, stride=stride, padding=padding)]
        if norm in NORM_LAYERS:
            layers.append(NORM_LAYERS[norm](out_fea))
        layers.append(nn.Tanh())
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class DeconvBlock(nn.Module):
    def __init__(self, in_fea, out_fea, kernel_size=2, stride=2, padding=0, output_padding=0, norm='bn'):
        super(DeconvBlock, self).__init__()
        layers = [nn.ConvTranspose2d(in_channels=in_fea, out_channels=out_fea, kernel_size=kernel_size, stride=stride, padding=padding, output_padding=output_padding)]
        if norm in NORM_LAYERS:
            layers.append(NORM_LAYERS[norm](out_fea))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class ResizeBlock(nn.Module):
    def __init__(self, in_fea, out_fea, scale_factor=2, mode='nearest', norm='bn'):
        super(ResizeBlock, self).__init__()
        layers = [nn.Upsample(scale_factor=scale_factor, mode=mode)]
        layers.append(nn.Conv2d(in_channels=in_fea, out_channels=out_fea, kernel_size=3, stride=1, padding=1))
        if norm in NORM_LAYERS:
            layers.append(NORM_LAYERS[norm](out_fea))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)



# FlatFault/CurveFault
# 1000, 70 -> 70, 70
class InversionNet(nn.Module):
    def __init__(self, dim1=32, dim2=64, dim3=128, dim4=256, dim5=512, sample_spatial=1.0, **kwargs):
        super(InversionNet, self).__init__()
        self.convblock1 = ConvBlock(5, dim1, kernel_size=(7, 1), stride=(2, 1), padding=(3, 0))
        self.convblock2_1 = ConvBlock(dim1, dim2, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0))
        self.convblock2_2 = ConvBlock(dim2, dim2, kernel_size=(3, 1), padding=(1, 0))
        self.convblock3_1 = ConvBlock(dim2, dim2, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0))
        self.convblock3_2 = ConvBlock(dim2, dim2, kernel_size=(3, 1), padding=(1, 0))
        self.convblock4_1 = ConvBlock(dim2, dim3, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0))
        self.convblock4_2 = ConvBlock(dim3, dim3, kernel_size=(3, 1), padding=(1, 0))
        self.convblock5_1 = ConvBlock(dim3, dim3, stride=2)
        self.convblock5_2 = ConvBlock(dim3, dim3)
        self.convblock6_1 = ConvBlock(dim3, dim4, stride=2)
        self.convblock6_2 = ConvBlock(dim4, dim4)
        self.convblock7_1 = ConvBlock(dim4, dim4, stride=2)
        self.convblock7_2 = ConvBlock(dim4, dim4)
        self.convblock8 = ConvBlock(dim4, dim5, kernel_size=(8, ceil(70 * sample_spatial / 8)), padding=0)
        
        self.deconv1_1 = DeconvBlock(dim5, dim5, kernel_size=5)
        self.deconv1_2 = ConvBlock(dim5, dim5)
        self.deconv2_1 = DeconvBlock(dim5, dim4, kernel_size=4, stride=2, padding=1)
        self.deconv2_2 = ConvBlock(dim4, dim4)
        self.deconv3_1 = DeconvBlock(dim4, dim3, kernel_size=4, stride=2, padding=1)
        self.deconv3_2 = ConvBlock(dim3, dim3)
        self.deconv4_1 = DeconvBlock(dim3, dim2, kernel_size=4, stride=2, padding=1)
        self.deconv4_2 = ConvBlock(dim2, dim2)
        self.deconv5_1 = DeconvBlock(dim2, dim1, kernel_size=4, stride=2, padding=1)
        self.deconv5_2 = ConvBlock(dim1, dim1)
        self.deconv6 = ConvBlock_Tanh(dim1, 1)
        
    def forward(self,x):
        # Encoder Part
        x = self.convblock1(x) # (None, 32, 500, 70)
        x = self.convblock2_1(x) # (None, 64, 250, 70)
        x = self.convblock2_2(x) # (None, 64, 250, 70)
        x = self.convblock3_1(x) # (None, 64, 125, 70)
        x = self.convblock3_2(x) # (None, 64, 125, 70)
        x = self.convblock4_1(x) # (None, 128, 63, 70) 
        x = self.convblock4_2(x) # (None, 128, 63, 70)
        x = self.convblock5_1(x) # (None, 128, 32, 35) 
        x = self.convblock5_2(x) # (None, 128, 32, 35)
        x = self.convblock6_1(x) # (None, 256, 16, 18) 
        x = self.convblock6_2(x) # (None, 256, 16, 18)
        x = self.convblock7_1(x) # (None, 256, 8, 9) 
        x = self.convblock7_2(x) # (None, 256, 8, 9)
        x = self.convblock8(x) # (None, 512, 1, 1)
        
        # Decoder Part 
        x = self.deconv1_1(x) # (None, 512, 5, 5)
        x = self.deconv1_2(x) # (None, 512, 5, 5)
        x = self.deconv2_1(x) # (None, 256, 10, 10) 
        x = self.deconv2_2(x) # (None, 256, 10, 10)
        x = self.deconv3_1(x) # (None, 128, 20, 20) 
        x = self.deconv3_2(x) # (None, 128, 20, 20)
        x = self.deconv4_1(x) # (None, 64, 40, 40) 
        x = self.deconv4_2(x) # (None, 64, 40, 40)
        x = self.deconv5_1(x) # (None, 32, 80, 80)
        x = self.deconv5_2(x) # (None, 32, 80, 80)
        x = F.pad(x, [-5, -5, -5, -5], mode="constant", value=0) # (None, 32, 70, 70) 125, 100
        x = self.deconv6(x) # (None, 1, 70, 70)
        return x

class FCN4_Deep_Resize_2(nn.Module):
    def __init__(self, dim1=32, dim2=64, dim3=128, dim4=256, dim5=512, ratio=1.0, upsample_mode='nearest'):
        super(FCN4_Deep_Resize_2, self).__init__()
        self.convblock1 = Conv2DwithBN(5, dim1, kernel_size=(7, 1), stride=(2, 1), padding=(3, 0))
        self.convblock2_1 = Conv2DwithBN(dim1, dim2, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0))
        self.convblock2_2 = Conv2DwithBN(dim2, dim2, kernel_size=(3, 1), padding=(1, 0))
        self.convblock3_1 = Conv2DwithBN(dim2, dim2, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0))
        self.convblock3_2 = Conv2DwithBN(dim2, dim2, kernel_size=(3, 1), padding=(1, 0))
        self.convblock4_1 = Conv2DwithBN(dim2, dim3, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0))
        self.convblock4_2 = Conv2DwithBN(dim3, dim3, kernel_size=(3, 1), padding=(1, 0))
        self.convblock5_1 = Conv2DwithBN(dim3, dim3, stride=2)
        self.convblock5_2 = Conv2DwithBN(dim3, dim3)
        self.convblock6_1 = Conv2DwithBN(dim3, dim4, stride=2)
        self.convblock6_2 = Conv2DwithBN(dim4, dim4)
        self.convblock7_1 = Conv2DwithBN(dim4, dim4, stride=2)
        self.convblock7_2 = Conv2DwithBN(dim4, dim4)
        self.convblock8 = Conv2DwithBN(dim4, dim5, kernel_size=(8, ceil(70 * ratio / 8)), padding=0)
        
        self.deconv1_1 = ResizeConv2DwithBN(dim5, dim5, scale_factor=5, mode=upsample_mode)
        self.deconv1_2 = Conv2DwithBN(dim5, dim5)
        self.deconv2_1 = ResizeConv2DwithBN(dim5, dim4, scale_factor=2, mode=upsample_mode)
        self.deconv2_2 = Conv2DwithBN(dim4, dim4)
        self.deconv3_1 = ResizeConv2DwithBN(dim4, dim3, scale_factor=2, mode=upsample_mode)
        self.deconv3_2 = Conv2DwithBN(dim3, dim3)
        self.deconv4_1 = ResizeConv2DwithBN(dim3, dim2, scale_factor=2, mode=upsample_mode)
        self.deconv4_2 = Conv2DwithBN(dim2, dim2)
        self.deconv5_1 = ResizeConv2DwithBN(dim2, dim1, scale_factor=2, mode=upsample_mode)
        self.deconv5_2 = Conv2DwithBN(dim1, dim1)
        self.deconv6 = Conv2DwithBN_Tanh(dim1, 1)
        
    def forward(self,x):
        # Encoder Part
        x = self.convblock1(x) # (None, 32, 500, 70)
        x = self.convblock2_1(x) # (None, 64, 250, 70)
        x = self.convblock2_2(x) # (None, 64, 250, 70)
        x = self.convblock3_1(x) # (None, 64, 125, 70)
        x = self.convblock3_2(x) # (None, 64, 125, 70)
        x = self.convblock4_1(x) # (None, 128, 63, 70)
        x = self.convblock4_2(x) # (None, 128, 63, 70)
        x = self.convblock5_1(x) # (None, 128, 32, 35)
        x = self.convblock5_2(x) # (None, 128, 32, 35)
        x = self.convblock6_1(x) # (None, 256, 16, 18)
        x = self.convblock6_2(x) # (None, 256, 16, 18)
        x = self.convblock7_1(x) # (None, 256, 8, 9)
        x = self.convblock7_2(x) # (None, 256, 8, 9)
        x = self.convblock8(x) # (None, 512, 1, 1)
        
        # Decoder Part 
        x = self.deconv1_1(x) # (None, 512, 5, 5)
        x = self.deconv1_2(x) # (None, 512, 5, 5)
        x = self.deconv2_1(x) # (None, 256, 10, 10)
        x = self.deconv2_2(x) # (None, 256, 10, 10)
        x = self.deconv3_1(x) # (None, 128, 20, 20)
        x = self.deconv3_2(x) # (None, 128, 20, 20)
        x = self.deconv4_1(x) # (None, 64, 40, 40)
        x = self.deconv4_2(x) # (None, 64, 40, 40)
        x = self.deconv5_1(x) # (None, 32, 80, 80)
        x = self.deconv5_2(x) # (None, 32, 80, 80)
        x = F.pad(x, [-5, -5, -5, -5], mode="constant", value=0) # (None, 32, 70, 70)
        x = self.deconv6(x) # (None, 1, 70, 70)
        return x


class Discriminator(nn.Module):
    def __init__(self, dim1=32, dim2=64, dim3=128, dim4=256, **kwargs):
        super(Discriminator, self).__init__()
        self.convblock1_1 = ConvBlock(1, dim1, stride=2)
        self.convblock1_2 = ConvBlock(dim1, dim1)
        self.convblock2_1 = ConvBlock(dim1, dim2, stride=2)
        self.convblock2_2 = ConvBlock(dim2, dim2)
        self.convblock3_1 = ConvBlock(dim2, dim3, stride=2)
        self.convblock3_2 = ConvBlock(dim3, dim3)
        self.convblock4_1 = ConvBlock(dim3, dim4, stride=2)
        self.convblock4_2 = ConvBlock(dim4, dim4)
        self.convblock5 = ConvBlock(dim4, 1, kernel_size=5, padding=0)

    def forward(self, x):
        x = self.convblock1_1(x)
        x = self.convblock1_2(x)
        x = self.convblock2_1(x)
        x = self.convblock2_2(x)
        x = self.convblock3_1(x)
        x = self.convblock3_2(x)
        x = self.convblock4_1(x)
        x = self.convblock4_2(x)
        x = self.convblock5(x)
        x = x.view(x.shape[0], -1)
        return x


class Conv_HPGNN(nn.Module):
    def __init__(self, in_fea, out_fea, kernel_size=None, stride=None, padding=None, **kwargs):
        super(Conv_HPGNN, self).__init__()
        layers = [
            ConvBlock(in_fea, out_fea, relu_slop=0.1, dropout=0.8),
            ConvBlock(out_fea, out_fea, relu_slop=0.1, dropout=0.8),
        ]
        if kernel_size is not None:
            layers.append(nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=padding))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class Deconv_HPGNN(nn.Module):
    def __init__(self, in_fea, out_fea, kernel_size, **kwargs):
        super(Deconv_HPGNN, self).__init__()
        layers = [
            nn.ConvTranspose2d(in_fea, in_fea, kernel_size=kernel_size, stride=2, padding=0),
            ConvBlock(in_fea, out_fea, relu_slop=0.1, dropout=0.8),
            ConvBlock(out_fea, out_fea, relu_slop=0.1, dropout=0.8)
        ]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)





# new class for the transformer of InversionNet:
class PatchEmbedding(nn.Module):
    def __init__(self, in_channels=5, embed_dim=256, patch_size=(10, 10)):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # x: [B, C=5, H=1000, W=70]
        x = self.proj(x)  # -> [B, embed_dim, H//p1, W//p2]
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # -> [B, N, C]
        return x, (H, W)

class TransformerEncoderBlock(nn.Module):
    def __init__(self, embed_dim=256, num_heads=8, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(embed_dim * mlp_ratio), embed_dim),
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.mlp(self.norm2(x))
        return x

# class InversionViT(nn.Module):
#     def __init__(self, in_channels=5, embed_dim=256, patch_size=(10, 10), depth=6, num_heads=8, sample_spatial=1.0, **kwargs):
#         super().__init__()
#         self.patch_embed = PatchEmbedding(in_channels, embed_dim, patch_size)
#         self.pos_embed = None
#         self.transformer = nn.Sequential(
#             *[TransformerEncoderBlock(embed_dim, num_heads) for _ in range(depth)]
#         )

#         # Decoder: upsample and predict 70x70
#         self.decoder_conv = nn.Sequential(
#             nn.ConvTranspose2d(embed_dim, 128, kernel_size=4, stride=2, padding=1),  # 100x7 -> 200x14
#             nn.BatchNorm2d(128), nn.ReLU(),

#             nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # 200x14 -> 400x28
#             nn.BatchNorm2d(64), nn.ReLU(),

#             nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # 400x28 -> 800x56
#             nn.BatchNorm2d(32), nn.ReLU(),

#             nn.Conv2d(32, 1, kernel_size=3, padding=1),  # → 800x56
#             nn.Upsample(size=(70, 70), mode='bilinear', align_corners=False),  # 强制调成最终尺寸
#             nn.Tanh()
#         )

#     def forward(self, x):
#         x, (H, W) = self.patch_embed(x)  # -> [B, N, D], (H=100, W=7)
#         if self.pos_embed is None:
#             N = x.shape[1]
#             self.pos_embed = nn.Parameter(torch.zeros(1, N, x.shape[2], device=x.device))
#             nn.init.trunc_normal_(self.pos_embed, std=0.02)

#         x = x + self.pos_embed
#         x = self.transformer(x)  # [B, N, D]
#         x = x.transpose(1, 2).reshape(-1, x.shape[-1], H, W)  # -> [B, D, H', W']
#         x = self.decoder_conv(x)  # -> [B, 1, 70, 70]
#         return x
class InversionViT(nn.Module):
    def __init__(self, in_channels=5, embed_dim=256, patch_size=(10, 10), depth=6, num_heads=8, sample_spatial=1.0, **kwargs):
        super().__init__()
        self.patch_embed = PatchEmbedding(in_channels, embed_dim, patch_size)

        # <<< 新增 >>> 初始化时直接计算 patch 数量
        num_patches = (1000 // patch_size[0]) * (70 // patch_size[1])  # H // p1 * W // p2
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))  # <<< moved here
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.transformer = nn.Sequential(
            *[TransformerEncoderBlock(embed_dim, num_heads) for _ in range(depth)]
        )

        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(),

            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(),

            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(),

            nn.Conv2d(32, 1, kernel_size=3, padding=1),
            nn.Upsample(size=(70, 70), mode='bilinear', align_corners=False),
            nn.Tanh()
        )

    def forward(self, x):
        x, (H, W) = self.patch_embed(x)
        x = x + self.pos_embed  # <<< 直接使用已有 pos_embed，不再创建
        x = self.transformer(x)
        x = x.transpose(1, 2).reshape(-1, x.shape[-1], H, W)
        x = self.decoder_conv(x)
        return x

# ----------------------------------------------
#             Ultimate InversionViT (UViT)
# ----------------------------------------------


# class ConvBlock(nn.Module):
#     def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1):
#         super().__init__()
#         self.conv = nn.Sequential(
#             nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding),
#             nn.BatchNorm2d(out_ch),
#             nn.ReLU(inplace=True)
#         )

#     def forward(self, x):
#         return self.conv(x)

# class UpBlock(nn.Module):
#     def __init__(self, in_ch, out_ch, skip_ch=0):
#         super().__init__()
#         self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1)
#         self.conv = nn.Sequential(
#             nn.BatchNorm2d(out_ch + skip_ch),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(out_ch + skip_ch, out_ch, kernel_size=3, padding=1),
#             nn.BatchNorm2d(out_ch),
#             nn.ReLU(inplace=True)
#         )

#     def forward(self, x, skip=None):
#         x = self.up(x)
#         if skip is not None:
#             # 对 skip 特征图下采样以匹配 x 的空间尺寸
#             skip = F.adaptive_avg_pool2d(skip, output_size=x.shape[-2:])
#             x = torch.cat([x, skip], dim=1)
#         return self.conv(x)

# class PatchEmbedding(nn.Module):
#     def __init__(self, in_channels=256, embed_dim=256, patch_size=(10, 10)):
#         super().__init__()
#         self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

#     def forward(self, x):
#         x = self.proj(x)  # B, D, H, W
#         H, W = x.shape[2], x.shape[3]
#         x = x.flatten(2).transpose(1, 2)  # B, N, D
#         return x, (H, W)

# class TransformerEncoderBlock(nn.Module):
#     def __init__(self, embed_dim=256, num_heads=8, mlp_ratio=4.0, dropout=0.1):
#         super().__init__()
#         self.norm1 = nn.LayerNorm(embed_dim)
#         self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
#         self.norm2 = nn.LayerNorm(embed_dim)
#         self.mlp = nn.Sequential(
#             nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
#             nn.GELU(),
#             nn.Linear(int(embed_dim * mlp_ratio), embed_dim),
#         )

#     def forward(self, x):
#         x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
#         x = x + self.mlp(self.norm2(x))
#         return x

# class UViT(nn.Module):
#     def __init__(self, in_channels=5, embed_dim=256, patch_size=(10, 10), depth=6, num_heads=8):
#         super().__init__()
#         # Encoder
#         self.stem = ConvBlock(in_channels, 64)
#         self.encoder_conv1 = ConvBlock(64, 128, stride=2)
#         self.encoder_conv2 = ConvBlock(128, 256, stride=2)

#         # Patch Embedding from deep features
#         self.patch_embed = PatchEmbedding(in_channels=256, embed_dim=embed_dim, patch_size=patch_size)
#         self.pos_embed = nn.Parameter(torch.zeros(1, 12 * 1, embed_dim))  # Assuming input (125, 18) // (10, 10) ≈ (12, 1)
#         nn.init.trunc_normal_(self.pos_embed, std=0.02)

#         # Transformer
#         self.transformer = nn.Sequential(
#             *[TransformerEncoderBlock(embed_dim, num_heads) for _ in range(depth)]
#         )

#         # Decoder with skip connections
#         self.decoder = nn.ModuleList([
#             UpBlock(embed_dim, 128, skip_ch=256),
#             UpBlock(128, 64, skip_ch=128),
#             UpBlock(64, 32, skip_ch=64)
#         ])

#         self.final = nn.Sequential(
#             nn.Conv2d(32, 1, kernel_size=3, padding=1),
#             nn.Upsample(size=(70, 70), mode='bilinear', align_corners=False),
#             nn.Tanh()
#         )

#     def forward(self, x):
#         # Encoder
#         skip0 = self.stem(x)               # [B, 64, 500, 70]
#         skip1 = self.encoder_conv1(skip0)  # [B, 128, 250, 35]
#         skip2 = self.encoder_conv2(skip1)  # [B, 256, 125, 18]

#         # Transformer
#         x, (H, W) = self.patch_embed(skip2)  # [B, N, D], e.g. N=25
#         if self.pos_embed.shape[1] != x.shape[1]:
#             # 动态插值位置编码（确保与 x 匹配）
#             pos_embed = nn.functional.interpolate(
#                 self.pos_embed.permute(0, 2, 1),  # [1, D, N]
#                 size=x.shape[1],                 # target N
#                 mode='linear',
#                 align_corners=False
#             ).permute(0, 2, 1)  # [1, N, D]
#         else:
#             pos_embed = self.pos_embed

#         x = x + pos_embed

#         x = self.transformer(x)                   # [B, N, D]
#         x = x.transpose(1, 2).reshape(-1, x.shape[-1], H, W)  # [B, D, H, W]

#         # Decoder with 3 skip levels
#         x = self.decoder[0](x, skip2)
#         x = self.decoder[1](x, skip1)
#         x = self.decoder[2](x, skip0)

#         # Output
#         x = self.final(x)
#         return x


# ----------------------------------------------
#             ResAttUNetTransformerFWI
# ----------------------------------------------

class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=(1,1), kernel_size=(3,3), padding=(1,1)):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size, (1,1), padding)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.skip = nn.Conv2d(in_ch, out_ch, 1, stride) if in_ch != out_ch or stride != (1,1) else nn.Identity()

    def forward(self, x):
        identity = self.skip(x)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return self.relu(x + identity)

class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, 1), nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, 1), nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, 1), nn.BatchNorm2d(1), nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels, embed_dim, patch_size):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)
        H, W = x.shape[2], x.shape[3]
        x = x.flatten(2).transpose(1, 2)
        return x, (H, W)

class TransformerEncoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads=8, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(embed_dim * mlp_ratio), embed_dim)
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.mlp(self.norm2(x))
        return x

class UpBlock(nn.Module):
    def __init__(self, in_ch, out_ch, skip_ch=0):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, 4, stride=2, padding=1)
        self.attn = AttentionGate(F_g=out_ch, F_l=skip_ch, F_int=out_ch//2)
        self.conv = ResidualBlock(out_ch + skip_ch, out_ch)

    def forward(self, x, skip=None):
        x = self.up(x)
        if skip is not None:
            skip = self.attn(x, skip)
            x = torch.cat([x, skip], dim=1)
        return self.conv(x)

class ResAttUNetTransformerFWI(nn.Module):
    def __init__(self, in_channels=5, embed_dim=512, patch_size=(4, 4), depth=4, num_heads=8):
        super().__init__()
        # Encoder
        self.enc1 = ResidualBlock(in_channels, 64, stride=(2,1), kernel_size=(7,1), padding=(3,0))
        self.enc2 = ResidualBlock(64, 128, stride=(2,1), kernel_size=(3,1), padding=(1,0))
        self.enc3 = ResidualBlock(128, 256, stride=(2,2))
        self.enc4 = ResidualBlock(256, 512, stride=(2,2))

        # Patch Embedding + Transformer
        self.patch_embed = PatchEmbedding(512, embed_dim, patch_size)
        num_patches = (32 // patch_size[0]) * (9 // patch_size[1])
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.transformer = nn.Sequential(
            *[TransformerEncoderBlock(embed_dim, num_heads) for _ in range(depth)]
        )

        # Decoder
        self.up1 = UpBlock(embed_dim, 256, 512)
        self.up2 = UpBlock(256, 128, 256)
        self.up3 = UpBlock(128, 64, 128)
        self.up4 = UpBlock(64, 32, 64)

        self.final = nn.Sequential(
            nn.Conv2d(32, 1, kernel_size=3, padding=1),
            nn.Upsample(size=(70, 70), mode='bilinear', align_corners=False),
            nn.Tanh()
        )

    def forward(self, x):
        # Encoder
        skip0 = self.enc1(x)
        skip1 = self.enc2(skip0)
        skip2 = self.enc3(skip1)
        x = self.enc4(skip2)

        # Transformer
        x, (H, W) = self.patch_embed(x)
        pos_embed = F.interpolate(self.pos_embed.permute(0,2,1), size=x.shape[1], mode='linear').permute(0,2,1)
        x = self.transformer(x + pos_embed)
        x = x.transpose(1, 2).reshape(-1, x.shape[-1], H, W)

        # Decoder with attention skip connections
        x = self.up1(x, skip2)
        x = self.up2(x, skip1)
        x = self.up3(x, skip0)
        x = self.up4(x, None)
        return self.final(x)


model_dict = {
    'InversionNet': InversionNet,
    'Discriminator': Discriminator,
    'UPFWI': FCN4_Deep_Resize_2,
    'InversionViT': InversionViT,
    # 'UViT': UViT
    'ResAttUNetTransformerFWI': ResAttUNetTransformerFWI
}