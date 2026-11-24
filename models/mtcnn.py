"""MTCNN：多任务级联卷积网络（静脉识别专利 ZL201910620017.X 的特征提取模块）

三级级联：P-Net -> R-Net -> O-Net
每个网络输出三个分支：分类(cls) + 边界框回归(box) + 特征点定位(landmark)

静脉识别专利中特征点数量为 55（55 个点 x 2 坐标 = 110 输出）。
架构沿用原版 MTCNN（P/R/O-Net），FC 输入维度根据输入尺寸自动计算，避免手算出错。
"""
import torch
import torch.nn as nn


class PNet(nn.Module):
    """候选网络：12x12x3 输入，快速粗筛，全卷积输出"""

    def __init__(self, num_landmarks=55):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 10, 3, 1),    # 12x12 -> 10x10
            nn.PReLU(),
            nn.MaxPool2d(2, 2),         # 10x10 -> 5x5
            nn.Conv2d(10, 16, 3, 1),   # 5x5 -> 3x3
            nn.PReLU(),
            nn.Conv2d(16, 32, 3, 1),   # 3x3 -> 1x1
            nn.PReLU(),
        )
        self.conv_cls = nn.Conv2d(32, 1, 1, 1)
        self.conv_box = nn.Conv2d(32, 4, 1, 1)
        self.conv_land = nn.Conv2d(32, num_landmarks * 2, 1, 1)

    def forward(self, x):
        f = self.features(x)
        cls = torch.sigmoid(self.conv_cls(f))
        box = self.conv_box(f)
        land = self.conv_land(f)
        return cls, box, land


class RNet(nn.Module):
    """精炼网络：24x24x3 输入"""

    def __init__(self, num_landmarks=55):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 28, 3, 1),    # 24 -> 22
            nn.PReLU(),
            nn.MaxPool2d(3, 2),         # 22 -> 10
            nn.Conv2d(28, 48, 3, 1),   # 10 -> 8
            nn.PReLU(),
            nn.MaxPool2d(3, 2),         # 8 -> 3
            nn.Conv2d(48, 64, 2, 1),   # 3 -> 2
            nn.PReLU(),
        )
        fc_in = self._compute_fc_in(24)
        self.fc1 = nn.Linear(fc_in, 128)
        self.prelu = nn.PReLU()
        self.fc_cls = nn.Linear(128, 1)
        self.fc_box = nn.Linear(128, 4)
        self.fc_land = nn.Linear(128, num_landmarks * 2)

    def _compute_fc_in(self, size):
        with torch.no_grad():
            f = self.features(torch.zeros(1, 3, size, size))
            return f.view(1, -1).size(1)

    def forward(self, x):
        f = self.features(x)
        f = f.view(f.size(0), -1)
        f = self.prelu(self.fc1(f))
        cls = torch.sigmoid(self.fc_cls(f))
        box = self.fc_box(f)
        land = self.fc_land(f)
        return cls, box, land


class ONet(nn.Module):
    """输出网络：48x48x3 输入，最精细"""

    def __init__(self, num_landmarks=55):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1),    # 48 -> 46
            nn.PReLU(),
            nn.MaxPool2d(3, 2),         # 46 -> 22
            nn.Conv2d(32, 64, 3, 1),   # 22 -> 20
            nn.PReLU(),
            nn.MaxPool2d(3, 2),         # 20 -> 9
            nn.Conv2d(64, 64, 3, 1),   # 9 -> 7
            nn.PReLU(),
            nn.MaxPool2d(2, 2),         # 7 -> 3
            nn.Conv2d(64, 128, 2, 1),  # 3 -> 2
            nn.PReLU(),
        )
        fc_in = self._compute_fc_in(48)
        self.fc1 = nn.Linear(fc_in, 256)
        self.prelu = nn.PReLU()
        self.fc_cls = nn.Linear(256, 1)
        self.fc_box = nn.Linear(256, 4)
        self.fc_land = nn.Linear(256, num_landmarks * 2)

    def _compute_fc_in(self, size):
        with torch.no_grad():
            f = self.features(torch.zeros(1, 3, size, size))
            return f.view(1, -1).size(1)

    def forward(self, x):
        f = self.features(x)
        f = f.view(f.size(0), -1)
        f = self.prelu(self.fc1(f))
        cls = torch.sigmoid(self.fc_cls(f))
        box = self.fc_box(f)
        land = self.fc_land(f)
        return cls, box, land


class MTCNN(nn.Module):
    """三级网络整体封装，便于统一调用"""

    def __init__(self, num_landmarks=55):
        super().__init__()
        self.pnet = PNet(num_landmarks)
        self.rnet = RNet(num_landmarks)
        self.onet = ONet(num_landmarks)

    def forward(self, x):
        return self.pnet(x), self.rnet(x), self.onet(x)


if __name__ == "__main__":
    for net, size in [(PNet(), 12), (RNet(), 24), (ONet(), 48)]:
        x = torch.randn(2, 3, size, size)
        cls, box, land = net(x)
        print(f"{net.__class__.__name__}: "
              f"cls{tuple(cls.shape)} box{tuple(box.shape)} land{tuple(land.shape)}")
