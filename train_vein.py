"""静脉识别训练脚本（专利 ZL201910620017.X 复现）

专利原始流程：IRCNN 图像增强 -> MTCNN 提取 55 个特征点 -> 身份比对。
由于公开静脉数据集（FV-USM 等）只有身份标签、没有 55 点 landmark 标注，
本脚本默认采用"身份分类"作为可行的训练路线：

    [IRCNN 增强(可选)] -> ResNet18 骨干 -> 分类头（手指身份分类）

- 分类损失：CrossEntropyLoss
- 支持训练/验证、断点保存、最佳权重保存
- landmark/bbox 多任务分支：仅在提供 --landmark_dir 标注时启用（对应专利 MTCNN 监督目标）

用法：
    # 1) 生成合成数据 + manifest（无真实数据时的冒烟测试）
    python data_prepare_fvusm.py --root ./fvusm_syn --make_synthetic --out_csv manifest.csv
    # 2) 训练
    python train_vein.py --manifest manifest.csv --epochs 20
"""
import argparse
import csv
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import models

from data_prepare_fvusm import VeinDataset, build_label_index
from models import IRCNN


class VeinClassifier(nn.Module):
    """静脉身份分类器：可选 IRCNN 增强 + ResNet18 骨干 + 分类头。

    若 use_landmark=True，额外输出 bbox(4) 与 landmark(55*2) 回归结果。
    """

    def __init__(self, num_classes, use_enhance=False, pretrained=False,
                 use_landmark=False, num_landmarks=55):
        super().__init__()
        self.use_enhance = use_enhance
        self.use_landmark = use_landmark
        self.enhance = IRCNN(in_channels=3, out_channels=3) if use_enhance else None

        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        base = models.resnet18(weights=weights)
        self.backbone = nn.Sequential(*list(base.children())[:-1])  # 去掉最后的 fc
        self.fc = nn.Linear(512, num_classes)

        if use_landmark:
            self.bbox_head = nn.Linear(512, 4)
            self.landmark_head = nn.Linear(512, num_landmarks * 2)

    def forward(self, x):
        if self.enhance is not None:
            x = self.enhance(x)
        feat = torch.flatten(self.backbone(x), 1)
        logits = self.fc(feat)
        if self.use_landmark:
            return logits, self.bbox_head(feat), self.landmark_head(feat)
        return logits


class VeinLandmarkDataset(VeinDataset):
    """在身份标签之外，额外读取 landmark/bbox 标注（.npy = 4 bbox + num_landmarks*2 landmark）"""

    def __init__(self, samples, id2idx, landmark_dir, size=(300, 100), num_landmarks=55):
        super().__init__(samples, id2idx, size=size, to_rgb=True)
        self.landmark_dir = landmark_dir
        self.num_landmarks = num_landmarks
        self.total = 4 + num_landmarks * 2

    def __getitem__(self, idx):
        img, label = super().__getitem__(idx)
        stem = os.path.splitext(os.path.basename(self.samples[idx]["path"]))[0]
        npy_path = os.path.join(self.landmark_dir, stem + ".npy")
        if os.path.exists(npy_path):
            arr = np.load(npy_path).astype(np.float32)
        else:
            arr = np.zeros(self.total, dtype=np.float32)
        return img, label, torch.from_numpy(arr[:4]), torch.from_numpy(arr[4:])


def load_samples(manifest):
    """读取 manifest.csv -> 样本列表"""
    samples = []
    with open(manifest, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            samples.append({"path": row["path"], "subject": row["subject"],
                            "finger": row["finger"], "session": row["session"]})
    return samples


def split_by_identity(samples, val_ratio=0.2, seed=42):
    """按身份做封闭集划分：每个身份内部随机拆 train/val（val 与 train 共享身份但不同图）。"""
    rng = np.random.RandomState(seed)
    train, val = [], []
    identities = sorted({(s["subject"], s["finger"]) for s in samples})
    for ident in identities:
        sub = [s for s in samples if (s["subject"], s["finger"]) == ident]
        rng.shuffle(sub)
        n_val = max(1, int(len(sub) * val_ratio))
        val.extend(sub[:n_val])
        train.extend(sub[n_val:])
    return train, val


def _step(model, batch, criterion, device, use_landmark, w_lm):
    if use_landmark:
        x, y, bbox, lm = [b.to(device) for b in batch]
    else:
        x, y = [b.to(device) for b in batch]
    if use_landmark:
        logits, pred_bbox, pred_lm = model(x)
        loss = criterion(logits, y) + w_lm * (
            F.mse_loss(pred_bbox, bbox) + F.mse_loss(pred_lm, lm))
    else:
        logits = model(x)
        loss = criterion(logits, y)
    return loss, logits, y


def train_one_epoch(model, loader, opt, criterion, device, use_landmark, w_lm):
    model.train()
    total = correct = 0
    loss_sum = 0.0
    for batch in loader:
        opt.zero_grad()
        loss, logits, y = _step(model, batch, criterion, device, use_landmark, w_lm)
        loss.backward()
        opt.step()
        total += y.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        loss_sum += loss.item() * y.size(0)
    return loss_sum / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device, use_landmark, w_lm):
    model.eval()
    total = correct = 0
    loss_sum = 0.0
    for batch in loader:
        loss, logits, y = _step(model, batch, criterion, device, use_landmark, w_lm)
        total += y.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        loss_sum += loss.item() * y.size(0)
    return loss_sum / total, correct / total


def evaluate_invariance(model, samples, id2idx, device, batch_size, size, num_workers=0):
    """在不同旋转/缩放下评估，验证尺度旋转不变性

    size 必须与训练时一致（(H, W)），否则输入尺寸错位会导致准确率暴跌。
    """
    configs = [
        ("原图", None, None),
        ("旋转+15°", 15.0, None),
        ("旋转+30°", 30.0, None),
        ("旋转+45°", 45.0, None),
        ("缩放0.8", None, 0.8),
        ("缩放1.2", None, 1.2),
        ("旋转30°+缩放1.2", 30.0, 1.2),
    ]
    criterion = nn.CrossEntropyLoss()
    print("\n==== 尺度旋转不变性验证 ====")
    for name, angle, scale in configs:
        ds = VeinDataset(samples, id2idx, size=size, to_rgb=True,
                         fixed_angle=angle, fixed_scale=scale)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        _, acc = evaluate(model, loader, criterion, device, False, 0.0)
        print(f"  {name:16s} 准确率 {acc:.4f}")


def main():
    ap = argparse.ArgumentParser(description="静脉识别训练")
    ap.add_argument("--manifest", required=True, help="data_prepare_fvusm.py 生成的 manifest.csv")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val_ratio", type=float, default=0.2)
    ap.add_argument("--use_enhance", action="store_true", help="启用 IRCNN 增强前端")
    ap.add_argument("--pretrained", action="store_true", help="ResNet18 使用 ImageNet 预训练权重")
    ap.add_argument("--landmark_dir", default=None,
                    help="可选：landmark/bbox 标注目录（.npy，启用 MTCNN 多任务监督）")
    ap.add_argument("--landmark_weight", type=float, default=0.1,
                    help="landmark+bbox 回归损失的权重")
    ap.add_argument("--num_landmarks", type=int, default=55,
                    help="landmark 点数（专利 55；FingerVeinSyn-5M 用 4）")
    ap.add_argument("--size", default="300,100",
                    help="图像缩放目标 (H,W)，如 300,100 或 150,300")
    ap.add_argument("--aug_rotation", type=float, default=15.0,
                    help="训练时随机旋转角度范围（度），0 表示关闭")
    ap.add_argument("--scale_min", type=float, default=0.8, help="训练时随机缩放下限")
    ap.add_argument("--scale_max", type=float, default=1.2, help="训练时随机缩放上限")
    ap.add_argument("--eval_invariance", action="store_true",
                    help="训练后在不同旋转/缩放下评估，验证尺度旋转不变性")
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--device", default=None, help="cuda / cpu，默认自动选择")
    ap.add_argument("--out_dir", default="checkpoints_vein", help="权重保存目录")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device) if args.device else \
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("设备:", device)

    samples = load_samples(args.manifest)
    train_samples, val_samples = split_by_identity(samples, args.val_ratio, args.seed)
    id2idx, idx2id = build_label_index(samples)  # 全量身份 -> 类别
    num_classes = len(id2idx)
    print(f"样本: train {len(train_samples)} / val {len(val_samples)}, 身份类别 {num_classes}")

    use_landmark = args.landmark_dir is not None
    augment = args.aug_rotation > 0 or args.scale_min != args.scale_max
    scale_range = (args.scale_min, args.scale_max) if augment else None
    img_size = tuple(int(x) for x in args.size.split(","))
    if use_landmark:
        print("已启用 landmark/bbox 多任务监督（专利 MTCNN 监督目标）")
        train_ds = VeinLandmarkDataset(train_samples, id2idx, args.landmark_dir,
                                       size=img_size, num_landmarks=args.num_landmarks)
        val_ds = VeinLandmarkDataset(val_samples, id2idx, args.landmark_dir,
                                     size=img_size, num_landmarks=args.num_landmarks)
    else:
        print("未提供 --landmark_dir，仅做身份分类（默认路线）")
        train_ds = VeinDataset(train_samples, id2idx, size=img_size,
                               augment=augment, rotation_deg=args.aug_rotation,
                               scale_range=scale_range)
        val_ds = VeinDataset(val_samples, id2idx, size=img_size)
    if augment:
        print(f"已启用尺度旋转不变性增强: 旋转±{args.aug_rotation}°, "
              f"缩放[{args.scale_min},{args.scale_max}]")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers)

    model = VeinClassifier(num_classes, use_enhance=args.use_enhance,
                           pretrained=args.pretrained, use_landmark=use_landmark,
                           num_landmarks=args.num_landmarks).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    os.makedirs(args.out_dir, exist_ok=True)
    best_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, opt, criterion, device,
                                          use_landmark, args.landmark_weight)
        va_loss, va_acc = evaluate(model, val_loader, criterion, device,
                                   use_landmark, args.landmark_weight)
        print(f"[Epoch {epoch:03d}] train loss {tr_loss:.4f} acc {tr_acc:.4f} | "
              f"val loss {va_loss:.4f} acc {va_acc:.4f}")
        if va_acc > best_acc:
            best_acc = va_acc
            torch.save(model.state_dict(), os.path.join(args.out_dir, "best.pth"))
    print(f"训练完成，最佳验证准确率 {best_acc:.4f}，权重保存在 {args.out_dir}/best.pth")

    if args.eval_invariance:
        model.load_state_dict(torch.load(os.path.join(args.out_dir, "best.pth"),
                                          map_location=device))
        evaluate_invariance(model, val_samples, id2idx, device,
                            args.batch_size, img_size, args.num_workers)


if __name__ == "__main__":
    main()
