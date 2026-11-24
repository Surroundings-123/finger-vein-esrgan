"""ESRGAN：增强超分辨率生成对抗网络（视频超分专利 ZL201910707218.3 的修复模型）

- 生成器：RRDBNet（23 个 RRDB 块，默认 4 倍放大）
- 判别器：VGG-128 风格（用于相对平均判别 RaGAN）

专利本身未新造网络，沿用 ESRGAN 论文结构，创新点在于"关键帧修复 + 帧分组采样"的
视频处理流水线（见数据预处理脚本）。
"""
import torch
import torch.nn as nn


class ResidualDenseBlock(nn.Module):
    """残差密集块（RDB）：密集连接 + 残差缩放"""

    def __init__(self, nf=64, gc=32):
        super().__init__()
        self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1)
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1)
        self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1)
        self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1)
        self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(0.2, True)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


class RRDB(nn.Module):
    """残差中的残差密集块（RRDB）：3 个 RDB 串联 + 残差缩放"""

    def __init__(self, nf=64, gc=32):
        super().__init__()
        self.rdb1 = ResidualDenseBlock(nf, gc)
        self.rdb2 = ResidualDenseBlock(nf, gc)
        self.rdb3 = ResidualDenseBlock(nf, gc)

    def forward(self, x):
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        return out * 0.2 + x


class RRDBNet(nn.Module):
    """ESRGAN 生成器（RRDBNet），默认 23 个 RRDB 块、4 倍放大"""

    def __init__(self, in_nc=3, out_nc=3, nf=64, nb=23, gc=32, scale=4):
        super().__init__()
        self.scale = scale
        self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1)
        self.body = nn.Sequential(*[RRDB(nf, gc) for _ in range(nb)])
        self.conv_body = nn.Conv2d(nf, nf, 3, 1, 1)

        # 上采样：每步 2x，共 log2(scale) 步
        upsample = []
        n_up = 0
        s = 1
        while s < scale:
            s *= 2
            n_up += 1
        for _ in range(n_up):
            upsample.append(nn.Upsample(scale_factor=2, mode="nearest"))
            upsample.append(nn.Conv2d(nf, nf, 3, 1, 1))
            upsample.append(nn.LeakyReLU(0.2, True))
        self.upsample = nn.Sequential(*upsample)

        self.conv_hr = nn.Conv2d(nf, nf, 3, 1, 1)
        self.conv_last = nn.Conv2d(nf, out_nc, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(0.2, True)

    def forward(self, x):
        feat = self.conv_first(x)
        body = self.conv_body(self.body(feat))
        feat = feat + body
        feat = self.upsample(feat)
        out = self.conv_last(self.lrelu(self.conv_hr(feat)))
        return out


class VGGStyleDiscriminator(nn.Module):
    """VGG-128 风格判别器，输入 128x128，输出真实度标量（用于相对平均 GAN）"""

    def __init__(self, in_nc=3, num_feat=64):
        super().__init__()

        def conv_block(in_f, out_f, downsample=False):
            stride = 2 if downsample else 1
            return nn.Sequential(
                nn.Conv2d(in_f, out_f, 3, stride, 1),
                nn.LeakyReLU(0.2, True),
            )

        self.conv0_0 = conv_block(in_nc, num_feat)               # 128x128
        self.conv0_1 = conv_block(num_feat, num_feat, True)      # 64x64
        self.conv1_0 = conv_block(num_feat, num_feat * 2)        # 64x64
        self.conv1_1 = conv_block(num_feat * 2, num_feat * 2, True)  # 32x32
        self.conv2_0 = conv_block(num_feat * 2, num_feat * 4)    # 32x32
        self.conv2_1 = conv_block(num_feat * 4, num_feat * 4, True)  # 16x16
        self.conv3_0 = conv_block(num_feat * 4, num_feat * 8)    # 16x16
        self.conv3_1 = conv_block(num_feat * 8, num_feat * 8, True)  # 8x8
        self.conv4_0 = conv_block(num_feat * 8, num_feat * 8)    # 8x8
        self.conv4_1 = conv_block(num_feat * 8, num_feat * 8, True)  # 4x4

        self.fc1 = nn.Linear(num_feat * 8 * 4 * 4, 100)
        self.fc2 = nn.Linear(100, 1)
        self.lrelu = nn.LeakyReLU(0.2, True)

    def forward(self, x):
        feat = self.conv0_0(x)
        feat = self.conv0_1(feat)
        feat = self.conv1_0(feat)
        feat = self.conv1_1(feat)
        feat = self.conv2_0(feat)
        feat = self.conv2_1(feat)
        feat = self.conv3_0(feat)
        feat = self.conv3_1(feat)
        feat = self.conv4_0(feat)
        feat = self.conv4_1(feat)
        feat = feat.view(feat.size(0), -1)
        feat = self.lrelu(self.fc1(feat))
        return self.fc2(feat)


if __name__ == "__main__":
    g = RRDBNet(in_nc=3, out_nc=3, nb=23, scale=4)
    x = torch.randn(1, 3, 64, 64)
    y = g(x)
    print("RRDBNet 输出形状:", tuple(y.shape))  # 期望 (1, 3, 256, 256)
    print("生成器参数量(M):", sum(p.numel() for p in g.parameters()) / 1e6)

    d = VGGStyleDiscriminator(in_nc=3, num_feat=64)
    z = torch.randn(1, 3, 128, 128)
    print("判别器输出形状:", tuple(d(z).shape))  # 期望 (1, 1)
    print("判别器参数量(M):", sum(p.numel() for p in d.parameters()) / 1e6)
