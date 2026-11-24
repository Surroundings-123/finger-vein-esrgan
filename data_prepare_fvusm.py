"""FV-USM 手指静脉数据集：下载 + 预处理脚本（静脉识别专利复现用）

================ 数据集说明 ================
FV-USM（Universiti Sains Malaysia）
  - 123 人 / 492 根手指 / 5904 张 ROI 图
  - ROI 尺寸：300 x 100（近红外静脉灰度图）
  - 官方申请页：http://drfendi.com/fv_usm_database/

================ 下载步骤（需手动申请） ================
  1. 打开 http://drfendi.com/fv_usm_database/ 填写申请表单
  2. 收到邮件后下载压缩包并解压，记下解压后的根目录（下称 <root>）
  3. 运行本脚本：python data_prepare_fvusm.py --root <root>

================ 脚本功能 ================
  1. 递归扫描 <root> 下所有静脉图（.bmp/.jpg/.png/.tif）
  2. 从文件名/目录名推断"被试ID + 手指"（可自定义 parse_subject 适配不同目录结构）
  3. 按"被试"划分 train/val/test（避免同一根手指跨集合造成数据泄漏）
  4. 生成 manifest.csv 清单 + PyTorch VeinDataset 数据类
"""
import argparse
import csv
import os
import re
from collections import defaultdict

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

IMG_EXTS = {".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff"}


def parse_subject(path, filename):
    """从文件路径推断 (subject_id, finger)。

    FV-USM 文件名通常以 3 位数字开头表示被试编号，例如 "001_....bmp"。
    若你的解压目录结构不同，只需改这个函数即可。
    """
    p_lower = path.lower()
    parts = [p for p in path.replace("\\", "/").split("/") if p]

    # 1) 被试编号：优先文件名开头的 3 位数字；否则在目录层级里向上找纯数字目录
    #    （FV-USM 官方结构为 <subject>/Session N/<finger>/1.bmp，文件名无编号）
    subject = "unknown"
    m = re.search(r"(\d{3,})", filename)
    if m:
        subject = m.group(1)
    else:
        for part in reversed(parts[:-1]):  # 排除文件名本身
            if re.fullmatch(r"\d+", part):
                subject = part
                break

    # 2) 手指：在路径层级里匹配关键词（比整串匹配更准，避免目录名干扰）
    #    目录名可能是 "Left Index"(空格) 或 "left_index"(下划线)，先归一化再匹配
    def _norm(s):
        return s.lower().replace(" ", "_").replace("-", "_")

    finger = "unknown"
    for key, val in [
        ("left_index", "left_index"),
        ("left_middle", "left_middle"),
        ("right_index", "right_index"),
        ("right_middle", "right_middle"),
        ("index", "index"),
        ("middle", "middle"),
    ]:
        if any(key in _norm(p) for p in parts):
            finger = val
            break

    # 3) session
    session = "1" if "session 1" in p_lower else ("2" if "session 2" in p_lower else "0")
    return subject, finger, session


def scan_dataset(root):
    """递归扫描根目录，返回样本列表 [{path, subject, finger, session}]"""
    samples = []
    for dirpath, _, files in os.walk(root):
        for fname in sorted(files):
            if os.path.splitext(fname)[1].lower() not in IMG_EXTS:
                continue
            path = os.path.join(dirpath, fname)
            subject, finger, session = parse_subject(path, fname)
            samples.append({"path": path, "subject": subject,
                            "finger": finger, "session": session})
    return samples


def split_by_subject(samples, val_ratio=0.15, test_ratio=0.15, seed=42):
    """按被试划分，保证同一被试的所有手指只出现在一个集合中。"""
    rng = np.random.RandomState(seed)
    subjects = sorted({s["subject"] for s in samples})
    rng.shuffle(subjects)
    n = len(subjects)
    n_test = max(1, int(n * test_ratio))
    n_val = max(1, int(n * val_ratio))
    test_subj = set(subjects[:n_test])
    val_subj = set(subjects[n_test:n_test + n_val])
    train_subj = set(subjects[n_test + n_val:])

    splits = {"train": [], "val": [], "test": []}
    for s in samples:
        if s["subject"] in train_subj:
            splits["train"].append(s)
        elif s["subject"] in val_subj:
            splits["val"].append(s)
        else:
            splits["test"].append(s)
    return splits


def build_label_index(samples):
    """把 (subject, finger) 映射成类别ID，返回 {identity -> class_id} 和 逆映射"""
    identities = sorted({(s["subject"], s["finger"]) for s in samples})
    id2idx = {k: i for i, k in enumerate(identities)}
    idx2id = {i: k for k, i in id2idx.items()}
    return id2idx, idx2id


class VeinDataset(Dataset):
    """手指静脉 PyTorch 数据集。

    Args:
        samples: scan_dataset 返回的样本列表
        id2idx: (subject, finger) -> class_id 的映射
        size: 缩放目标 (H, W)，None 则保持原图
        to_rgb: 是否把单通道灰度图复制成 3 通道（匹配 3 通道模型）
        augment: 是否启用随机旋转/缩放增强（尺度旋转不变性）
        rotation_deg: 随机旋转最大角度（度）
        scale_range: 随机缩放范围 (min, max)
        fixed_angle / fixed_scale: 固定旋转角/缩放（用于不变性验证，优先于随机）
    """

    def __init__(self, samples, id2idx, size=(300, 100), to_rgb=True,
                 augment=False, rotation_deg=0.0, scale_range=None,
                 fixed_angle=None, fixed_scale=None):
        self.samples = samples
        self.id2idx = id2idx
        self.size = size
        self.to_rgb = to_rgb
        self.augment = augment
        self.rotation_deg = rotation_deg
        self.scale_range = scale_range
        self.fixed_angle = fixed_angle
        self.fixed_scale = fixed_scale

    def __len__(self):
        return len(self.samples)

    def _transform(self, img):
        """对已缩放到目标尺寸的灰度图做旋转/缩放（不变性增强或固定变换验证）"""
        if self.fixed_angle is not None:
            angle = self.fixed_angle
        elif self.augment:
            angle = np.random.uniform(-self.rotation_deg, self.rotation_deg)
        else:
            angle = 0.0

        if self.fixed_scale is not None:
            scale = self.fixed_scale
        elif self.augment and self.scale_range is not None:
            scale = np.random.uniform(*self.scale_range)
        else:
            scale = 1.0

        if angle == 0.0 and scale == 1.0:
            return img
        h, w = img.shape[:2]
        M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, scale)
        return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    def __getitem__(self, idx):
        s = self.samples[idx]
        img = cv2.imread(s["path"], cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"无法读取图片: {s['path']}")
        if self.size is not None:
            h, w = self.size
            img = cv2.resize(img, (w, h), interpolation=cv2.INTER_CUBIC)
        img = self._transform(img)
        img = img.astype(np.float32) / 255.0          # 归一化到 [0,1]
        img = np.expand_dims(img, axis=0)             # (1, H, W)
        if self.to_rgb:
            img = np.repeat(img, 3, axis=0)           # (3, H, W)
        tensor = torch.from_numpy(img)
        label = self.id2idx[(s["subject"], s["finger"])]
        return tensor, label


def write_manifest(splits, out_csv):
    """把划分结果写成 CSV，便于复现"""
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["split", "path", "subject", "finger", "session"])
        w.writeheader()
        for split in ["train", "val", "test"]:
            for s in splits[split]:
                w.writerow({"split": split, **s})
    print(f"manifest 已写入: {out_csv}")


def _make_vein_base(h, w, rng):
    """生成某身份固定的静脉走向模板（背景亮度 + 暗色静脉曲线）"""
    img = np.full((h, w), 160, dtype=np.uint8)
    n_veins = rng.randint(4, 7)
    for _ in range(n_veins):
        y = np.linspace(0, h - 1, 12)
        x0 = rng.uniform(5, w - 5)
        x = x0 + np.cumsum(rng.uniform(-1.5, 1.5, size=y.size))
        x = np.clip(x, 2, w - 3)
        pts = np.stack([x, y], axis=1).astype(np.int32)
        cv2.polylines(img, [pts], False,
                      int(rng.uniform(55, 90)), int(rng.uniform(2, 4)))
    return cv2.GaussianBlur(img, (3, 3), 0)


def make_synthetic_vein_image(base, rng=None, brightness_jitter=10.0, shift=3):
    """在身份模板基础上叠加噪声/亮度抖动/轻微平移，模拟同一手指的多次采集。"""
    rng = rng or np.random.RandomState()
    h, w = base.shape[:2]
    img = base.astype(np.float32)
    img += rng.normal(0, 5, (h, w))                        # 传感器噪声
    img += rng.uniform(-brightness_jitter, brightness_jitter)  # 亮度抖动
    dx = rng.randint(-shift, shift + 1)
    dy = rng.randint(-shift, shift + 1)
    M = np.float32([[1, 0, dx], [0, 1, dy]])               # 轻微平移
    img = cv2.warpAffine(img, M, (w, h))
    grad = np.linspace(0.9, 1.1, w, dtype=np.float32)[None, :]  # 横向光照渐变
    img = img * grad
    return np.clip(img, 0, 255).astype(np.uint8)


def generate_synthetic_data(root, n_subjects=20, fingers_per_subject=4,
                            n_per_finger=6, size=(300, 100), seed=42):
    """生成合成静脉数据集，目录结构: root/<subject>/<finger>/session N/<index>.png

    每个 (subject, finger) 使用固定的静脉模板（身份一致），再叠加噪声/平移，
    使同一身份的样本可被分类器学习区分。含 session 1 / 2 两次采集（模拟 FV-USM 协议）。
    """
    rng = np.random.RandomState(seed)
    h, w = size
    fingers = ["left_index", "left_middle", "right_index", "right_middle"]
    created = 0
    for si in range(n_subjects):
        subject = f"{si + 1:03d}"
        for fi in range(fingers_per_subject):
            finger = fingers[fi % len(fingers)]
            # 固定身份模板：用 (seed, subject, finger) 派生独立随机流，保证同身份一致
            ident_rng = np.random.RandomState(seed * 100000 + si * 100 + fi)
            base = _make_vein_base(h, w, ident_rng)
            for session in (1, 2):
                subdir = os.path.join(root, subject, finger, f"session {session}")
                os.makedirs(subdir, exist_ok=True)
                for k in range(n_per_finger):
                    img = make_synthetic_vein_image(base, rng)
                    fname = f"{subject}_{finger.replace('_', '')}_s{session}_{k + 1:03d}.png"
                    cv2.imwrite(os.path.join(subdir, fname), img)
                    created += 1
    print(f"已生成 {created} 张合成静脉图 -> {root}")
    return created


def main():
    ap = argparse.ArgumentParser(description="FV-USM 数据预处理")
    ap.add_argument("--root", required=True, help="FV-USM 解压根目录（或合成数据生成目录）")
    ap.add_argument("--out_csv", default="manifest.csv", help="清单输出路径")
    ap.add_argument("--val_ratio", type=float, default=0.15)
    ap.add_argument("--test_ratio", type=float, default=0.15)
    ap.add_argument("--make_synthetic", action="store_true",
                    help="生成合成静脉样图（无需真实数据，用于冒烟测试）")
    ap.add_argument("--n_subjects", type=int, default=20, help="合成数据：被试数")
    ap.add_argument("--fingers_per_subject", type=int, default=4, help="合成数据：每被试手指数")
    ap.add_argument("--n_per_finger", type=int, default=6, help="合成数据：每手指每 session 张数")
    args = ap.parse_args()

    if args.make_synthetic:
        generate_synthetic_data(
            args.root,
            n_subjects=args.n_subjects,
            fingers_per_subject=args.fingers_per_subject,
            n_per_finger=args.n_per_finger,
        )

    samples = scan_dataset(args.root)
    if not samples:
        print("[错误] 未在指定目录找到任何图片，请检查 --root 路径")
        return
    print(f"扫描到 {len(samples)} 张图片")

    # 目录结构校验：统计被试数与手指(身份)数
    subjects = {s["subject"] for s in samples}
    identities = {(s["subject"], s["finger"]) for s in samples}
    print(f"  被试数: {len(subjects)}, 手指(身份)数: {len(identities)}")

    splits = split_by_subject(samples, args.val_ratio, args.test_ratio)
    id2idx, idx2id = build_label_index(samples)
    print(f"类别数（被试x手指）: {len(id2idx)}")

    for split in ["train", "val", "test"]:
        print(f"  {split}: {len(splits[split])} 张")

    write_manifest(splits, args.out_csv)

    # 冒烟测试：加载 train 集前 4 张
    ds = VeinDataset(splits["train"], id2idx)
    print("\n冒烟测试（train 集前 2 张）:")
    for i in range(min(2, len(ds))):
        img, label = ds[i]
        print(f"  [{i}] 图像形状 {tuple(img.shape)}, 类别 {label} ({(idx2id[label])})")


if __name__ == "__main__":
    main()
