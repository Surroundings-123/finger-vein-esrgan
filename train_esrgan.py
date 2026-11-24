"""ESRGAN 视频超分训练脚本（专利 ZL201910707218.3 复现）

专利流程：关键帧修复 + 帧分组采样，修复模型即 ESRGAN（RRDBNet 生成器 + VGG-128 判别器）。
本脚本训练 ESRGAN 修复模型：
  - 生成器：RRDBNet（23 个 RRDB，4x 放大）
  - 判别器：VGGStyleDiscriminator（相对平均判别 RaGAN）
  - 损失：L1 像素损失 + VGG19 感知损失 + RaGAN 对抗损失
  - 数据：从 HR 图像/帧随机裁剪 patch，双三次下采样得到 LR（可选加噪）

用法（视频需先抽帧）：
    ffmpeg -i video.mp4 -q:v 2 frames/f_%06d.png
    python train_esrgan.py --hr_dir ./frames --epochs 20
"""
import argparse
import os

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import models

from models import RRDBNet, VGGStyleDiscriminator


class PatchDataset(Dataset):
    """从 HR 图像随机裁剪 patch，并双三次下采样生成 LR 配对"""

    IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp"}

    def __init__(self, hr_dir, patch_size=128, scale=4, noise_sigma=0.0):
        self.patch_size = patch_size
        self.scale = scale
        self.noise_sigma = noise_sigma
        self.paths = [os.path.join(hr_dir, f) for f in sorted(os.listdir(hr_dir))
                      if os.path.splitext(f)[1].lower() in self.IMG_EXTS]

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = cv2.imread(self.paths[idx], cv2.IMREAD_COLOR)  # BGR
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        ps = self.patch_size
        if h < ps or w < ps:
            s = max(ps / h, ps / w)
            img = cv2.resize(img, (int(w * s) + 1, int(h * s) + 1),
                             interpolation=cv2.INTER_CUBIC)
            h, w = img.shape[:2]
        top = np.random.randint(0, h - ps + 1)
        left = np.random.randint(0, w - ps + 1)
        hr = img[top:top + ps, left:left + ps]
        if np.random.rand() < 0.5:
            hr = np.fliplr(hr).copy()

        lr_h = lr_w = ps // self.scale
        lr = cv2.resize(hr, (lr_w, lr_h), interpolation=cv2.INTER_CUBIC)
        if self.noise_sigma > 0:
            lr = lr.astype(np.float32)
            lr += np.random.normal(0, self.noise_sigma, lr.shape).astype(np.float32)
            lr = np.clip(lr, 0, 255)

        hr_t = torch.from_numpy(hr.astype(np.float32) / 255.0).permute(2, 0, 1)
        lr_t = torch.from_numpy(lr.astype(np.float32) / 255.0).permute(2, 0, 1)
        return lr_t, hr_t


class VGGFeatureExtractor(nn.Module):
    """VGG19 感知特征提取器（ImageNet 预训练，冻结）"""

    def __init__(self, layer_index=35):
        super().__init__()
        vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1)
        vgg.eval()
        self.features = nn.Sequential(*list(vgg.features.children())[:layer_index + 1])
        for p in self.features.parameters():
            p.requires_grad = False
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x):
        x = (x - self.mean) / self.std
        return self.features(x)


def ragan_d_loss(real_pred, fake_pred):
    """相对平均判别损失（判别器）"""
    real_logit = real_pred - torch.mean(fake_pred)
    fake_logit = fake_pred - torch.mean(real_pred)
    l_real = F.binary_cross_entropy_with_logits(real_logit, torch.ones_like(real_logit))
    l_fake = F.binary_cross_entropy_with_logits(fake_logit, torch.zeros_like(fake_logit))
    return (l_real + l_fake) / 2


def ragan_g_loss(real_pred, fake_pred):
    """相对平均对抗损失（生成器）"""
    real_logit = real_pred - torch.mean(fake_pred)
    fake_logit = fake_pred - torch.mean(real_pred)
    l_real = F.binary_cross_entropy_with_logits(real_logit, torch.zeros_like(real_logit))
    l_fake = F.binary_cross_entropy_with_logits(fake_logit, torch.ones_like(fake_logit))
    return (l_real + l_fake) / 2


def main():
    ap = argparse.ArgumentParser(description="ESRGAN 视频超分训练")
    ap.add_argument("--hr_dir", required=True, help="HR 图像/视频帧目录")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--patch_size", type=int, default=128, help="HR patch 边长（与判别器 128 输入匹配）")
    ap.add_argument("--scale", type=int, default=4)
    ap.add_argument("--noise_sigma", type=float, default=0.0, help="LR 加噪标准差（0 表示不加噪）")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--w_l1", type=float, default=1.0)
    ap.add_argument("--w_perceptual", type=float, default=1.0)
    ap.add_argument("--w_gan", type=float, default=0.1)
    ap.add_argument("--no_gan", action="store_true", help="只训生成器（像素+感知损失），不启用对抗")
    ap.add_argument("--no_perceptual", action="store_true", help="不启用 VGG 感知损失")
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out_dir", default="checkpoints_esrgan")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device) if args.device else \
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("设备:", device)

    ds = PatchDataset(args.hr_dir, patch_size=args.patch_size,
                      scale=args.scale, noise_sigma=args.noise_sigma)
    if len(ds) == 0:
        print(f"[错误] {args.hr_dir} 下没有找到图片，请先抽帧或检查路径")
        return
    print(f"HR 图像数: {len(ds)}")
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.num_workers)

    gen = RRDBNet(in_nc=3, out_nc=3, nb=23, gc=32, scale=args.scale).to(device)
    g_opt = torch.optim.Adam(gen.parameters(), lr=args.lr, betas=(0.9, 0.99))
    l1_loss = nn.L1Loss()

    use_gan = not args.no_gan
    use_perceptual = not args.no_perceptual
    if use_gan:
        disc = VGGStyleDiscriminator(in_nc=3, num_feat=64).to(device)
        d_opt = torch.optim.Adam(disc.parameters(), lr=args.lr, betas=(0.9, 0.99))
    if use_perceptual:
        try:
            vgg = VGGFeatureExtractor().to(device)
            vgg.eval()
        except Exception as e:
            print(f"[警告] VGG19 权重加载失败（可能无法联网），已禁用感知损失: {e}")
            use_perceptual = False

    os.makedirs(args.out_dir, exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        g_l1_sum = g_perc_sum = g_gan_sum = d_sum = 0.0
        n = 0
        for lr, hr in loader:
            lr, hr = lr.to(device), hr.to(device)
            sr = gen(lr)

            # 更新生成器
            g_opt.zero_grad()
            loss_l1 = l1_loss(sr, hr)
            g_loss = loss_l1 * args.w_l1
            if use_perceptual:
                loss_perc = F.mse_loss(vgg(sr), vgg(hr))
                g_loss = g_loss + loss_perc * args.w_perceptual
                g_perc_sum += loss_perc.item()
            if use_gan:
                real_pred = disc(hr).detach()
                fake_pred = disc(sr)
                loss_gan = ragan_g_loss(real_pred, fake_pred)
                g_loss = g_loss + loss_gan * args.w_gan
                g_gan_sum += loss_gan.item()
            g_loss.backward()
            g_opt.step()

            # 更新判别器
            if use_gan:
                d_opt.zero_grad()
                real_pred = disc(hr)
                fake_pred = disc(sr.detach())
                loss_d = ragan_d_loss(real_pred, fake_pred)
                loss_d.backward()
                d_opt.step()
                d_sum += loss_d.item()

            g_l1_sum += loss_l1.item()
            n += 1

        msg = f"[Epoch {epoch:03d}] L1 {g_l1_sum / n:.4f}"
        if use_perceptual:
            msg += f" | Perceptual {g_perc_sum / n:.4f}"
        if use_gan:
            msg += f" | G_gan {g_gan_sum / n:.4f} | D {d_sum / n:.4f}"
        print(msg)

        torch.save(gen.state_dict(), os.path.join(args.out_dir, f"generator_epoch{epoch}.pth"))
        if use_gan:
            torch.save(disc.state_dict(), os.path.join(args.out_dir, f"discriminator_epoch{epoch}.pth"))
    print(f"训练完成，权重保存在 {args.out_dir}/")


if __name__ == "__main__":
    main()
