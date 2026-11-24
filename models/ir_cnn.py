"""IRCNN：图像增强网络（静脉识别专利 ZL201910620017.X 的预处理模块）

结构：7 层扩张卷积（dilated convolution）
  扩张因子：1 -> 2 -> 3 -> 4 -> 3 -> 2 -> 1
  - 第 1 层    ：3x3 扩张卷积 + ReLU
  - 中间 5 层  ：3x3 扩张卷积 + BatchNorm + ReLU
  - 最后 1 层  ：3x3 扩张卷积（输出，无激活）

用途：对低质量静脉图像做增强/锐化，提升后续 MTCNN 特征提取的鲁棒性。
"""
import torch
import torch.nn as nn


class IRCNN(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, n_feats=64,
                 dilations=(1, 2, 3, 4, 3, 2, 1)):
        super().__init__()
        layers = []
        n = len(dilations)
        for i, d in enumerate(dilations):
            # 3x3 卷积，padding=dilation 保持输出尺寸不变
            if i == 0:
                layers.append(nn.Conv2d(in_channels, n_feats, 3, padding=d, dilation=d))
                layers.append(nn.ReLU(inplace=True))
            elif i == n - 1:
                layers.append(nn.Conv2d(n_feats, out_channels, 3, padding=d, dilation=d))
            else:
                layers.append(nn.Conv2d(n_feats, n_feats, 3, padding=d, dilation=d))
                layers.append(nn.BatchNorm2d(n_feats))
                layers.append(nn.ReLU(inplace=True))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


if __name__ == "__main__":
    m = IRCNN(in_channels=3, out_channels=3)
    x = torch.randn(2, 3, 300, 100)  # FV-USM ROI 尺寸
    y = m(x)
    print("IRCNN 输出形状:", tuple(y.shape))  # 期望 (2, 3, 300, 100)
    print("参数量:", sum(p.numel() for p in m.parameters()))
