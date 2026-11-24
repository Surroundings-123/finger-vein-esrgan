"""视频超分流水线（专利 ZL201910707218.3 复现）：关键帧修复 + 帧分组采样

专利流程：对视频先做关键帧检测，再用 ESRGAN(RRDBNet) 修复关键帧，
最后按帧分组采样，降低冗余、提升整体画质。

本脚本实现四步：
  1. 抽帧：从视频抽取帧（或用现成帧目录）
  2. 关键帧检测：相邻帧灰度差(MAD)超过阈值 → 标记关键帧
  3. 关键帧修复：用训练好的 RRDBNet 对关键帧做 scale 倍超分
  4. 帧分组采样：按 GOP 分组，每组保留关键帧（已修复）+ 均匀采样帧

用法：
    # 从视频（需先训练好 RRDBNet，见 train_esrgan.py）
    python video_sr_pipeline.py --input video.mp4 --ckpt checkpoints_esrgan/generator_epoch20.pth
    # 从帧目录
    python video_sr_pipeline.py --input ./frames --ckpt checkpoints_esrgan/generator_epoch20.pth
"""
import argparse
import os

import cv2
import numpy as np
import torch

from models import RRDBNet

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp"}


def extract_frames(video_path, out_dir, max_frames=None):
    """用 cv2 从视频抽帧，返回帧文件路径列表"""
    os.makedirs(out_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    paths, i = [], 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        p = os.path.join(out_dir, f"{i:06d}.png")
        cv2.imwrite(p, frame)
        paths.append(p)
        i += 1
        if max_frames is not None and i >= max_frames:
            break
    cap.release()
    return paths


def load_frames(frame_dir):
    """加载目录下所有帧文件路径（按文件名排序）"""
    return sorted([os.path.join(frame_dir, f) for f in os.listdir(frame_dir)
                   if os.path.splitext(f)[1].lower() in IMG_EXTS])


def load_generator(ckpt_path, scale=4, device="cuda"):
    """加载训练好的 RRDBNet 生成器"""
    model = RRDBNet(in_nc=3, out_nc=3, nb=23, gc=32, scale=scale).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    return model


def detect_keyframes(frame_paths, diff_thresh=30.0, resize=(160, 90)):
    """基于相邻帧灰度差(MAD)检测关键帧，返回关键帧索引列表"""
    key = {0}  # 首帧必为关键帧
    prev = None
    for i, p in enumerate(frame_paths):
        g = cv2.resize(cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2GRAY), resize)
        if prev is not None:
            mad = float(np.abs(g.astype(np.float32) - prev).mean())
            if mad > diff_thresh:
                key.add(i)
        prev = g
    return sorted(key)


@torch.no_grad()
def repair_frame(model, frame_bgr, device="cuda"):
    """用 RRDBNet 对单帧 BGR 做 scale 倍超分，返回 BGR 图"""
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    x = torch.from_numpy(rgb.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(device)
    y = model(x).squeeze(0).clamp(0, 1)
    y = (y.permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)
    return cv2.cvtColor(y, cv2.COLOR_RGB2BGR)


def group_and_sample(frame_paths, key_indices, group_size=16):
    """按 GOP 分组，每组保留关键帧 + 均匀采样帧，返回 {group_id: [帧索引]}"""
    groups = {}
    key_set = set(key_indices)
    n = len(frame_paths)
    for gid, start in enumerate(range(0, n, group_size)):
        end = min(start + group_size, n)
        idxs = list(range(start, end))
        sampled = set(i for i in idxs if i in key_set)  # 组内关键帧
        step = max(1, len(idxs) // 4)                    # 均匀采样补齐
        sampled.update(idxs[::step])
        groups[gid] = sorted(sampled)
    return groups


def main():
    ap = argparse.ArgumentParser(description="视频超分流水线：关键帧修复 + 帧分组采样")
    ap.add_argument("--input", required=True, help="视频文件或帧目录")
    ap.add_argument("--ckpt", required=True, help="RRDBNet 生成器权重 .pth")
    ap.add_argument("--out_dir", default="sr_output")
    ap.add_argument("--scale", type=int, default=4)
    ap.add_argument("--diff_thresh", type=float, default=30.0, help="关键帧检测 MAD 阈值")
    ap.add_argument("--group_size", type=int, default=16, help="帧分组大小(GOP)")
    ap.add_argument("--max_frames", type=int, default=None, help="从视频抽帧的最大帧数")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    device = torch.device(args.device) if args.device else \
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("设备:", device)

    # 1. 抽帧
    if os.path.isdir(args.input):
        frame_paths = load_frames(args.input)
    else:
        frame_paths = extract_frames(args.input, os.path.join(args.out_dir, "_frames"),
                                     args.max_frames)
    if not frame_paths:
        print("[错误] 未获取到帧，请检查 --input")
        return
    print(f"共 {len(frame_paths)} 帧")

    # 2. 关键帧检测
    key_indices = detect_keyframes(frame_paths, args.diff_thresh)
    print(f"检测到 {len(key_indices)} 个关键帧")

    # 3. 关键帧修复
    model = load_generator(args.ckpt, args.scale, device)
    os.makedirs(args.out_dir, exist_ok=True)
    for i in key_indices:
        sr = repair_frame(model, cv2.imread(frame_paths[i]), device)
        cv2.imwrite(os.path.join(args.out_dir, f"key_{i:06d}_x{args.scale}.png"), sr)
    print(f"关键帧修复完成 -> {args.out_dir}/key_*.png")

    # 4. 帧分组采样
    groups = group_and_sample(frame_paths, key_indices, args.group_size)
    with open(os.path.join(args.out_dir, "groups.txt"), "w", encoding="utf-8") as f:
        for gid, idxs in groups.items():
            f.write(f"group {gid}: {idxs}\n")
    print(f"帧分组: {len(groups)} 组 (GOP={args.group_size}) -> {args.out_dir}/groups.txt")


if __name__ == "__main__":
    main()
