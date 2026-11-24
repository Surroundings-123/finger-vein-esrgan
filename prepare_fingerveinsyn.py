"""FingerVeinSyn-5M 数据集适配脚本：解析 raw_images + annotations -> manifest + landmark

FingerVeinSyn-5M（ACM MM 2025）：
  - 50000 身份 × 100 张 = 500 万张合成静脉图，尺寸 600×300 灰度 PNG
  - 每张图配一个 XML 标注，含 bbox + 4 个关节腔关键点（landmark）
  - 命名：[ID]_[SampleNum]_[VariationType].png / .xml

本脚本生成两类产物，直接对接 train_vein.py：
  1. manifest.csv          —— 身份分类路线（path, subject, finger, session）
  2. landmarks/*.npy       —— MTCNN landmark/bbox 多任务监督
     每个 .npy = [bbox(4), landmark(num_points*2)]，全部归一化到 [0,1]

用法：
    python prepare_fingerveinsyn.py --root <FingerVeinSyn5M目录> --out_csv manifest_fvsyn.csv --landmark_dir landmarks
"""
import argparse
import csv
import os
import re
import xml.etree.ElementTree as ET

import numpy as np


def parse_annotation(xml_path):
    """解析 XML，返回 (bbox[4], landmarks[2*N])，均归一化到 [0,1]"""
    tree = ET.parse(xml_path)
    img = tree.find("image")
    W = float(img.get("width"))
    H = float(img.get("height"))

    b = img.find("bbox")
    bbox = [
        float(b.find("center_x").text) / W,
        float(b.find("center_y").text) / H,
        float(b.find("width").text) / W,
        float(b.find("height").text) / H,
    ]

    pts = []
    for p in img.find("joint_points").findall("point"):
        pts += [float(p.find("x").text) / W, float(p.find("y").text) / H]
    return bbox, pts


def main():
    ap = argparse.ArgumentParser(description="FingerVeinSyn-5M 数据适配")
    ap.add_argument("--root", required=True,
                    help="FingerVeinSyn5M 目录（内含 raw_images/ 和 annotations/）")
    ap.add_argument("--out_csv", default="manifest_fvsyn.csv")
    ap.add_argument("--landmark_dir", default="landmarks")
    args = ap.parse_args()

    raw_dir = os.path.join(args.root, "raw_images")
    ann_dir = os.path.join(args.root, "annotations")
    os.makedirs(args.landmark_dir, exist_ok=True)

    rows = []
    n_img = n_ann = 0
    n_pts = None
    for dirpath, _, files in os.walk(raw_dir):
        for fname in sorted(files):
            if not fname.lower().endswith(".png"):
                continue
            img_path = os.path.join(dirpath, fname)
            rel = os.path.relpath(img_path, raw_dir)
            xml_path = os.path.join(ann_dir, os.path.splitext(rel)[0] + ".xml")
            if not os.path.exists(xml_path):
                print("[跳过] 缺标注:", rel)
                continue

            n_img += 1
            m = re.match(r"(\d+)_", fname)
            subject = m.group(1) if m else "unknown"
            bbox, pts = parse_annotation(xml_path)
            n_ann += 1
            if n_pts is None:
                n_pts = len(pts) // 2

            # 保存 landmark .npy（bbox 在前，landmark 在后）
            arr = np.asarray(bbox + pts, dtype=np.float32)
            stem = os.path.splitext(fname)[0]
            np.save(os.path.join(args.landmark_dir, stem + ".npy"), arr)

            rows.append({"path": img_path, "subject": subject,
                         "finger": "fv", "session": "0"})

    with open(args.out_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["path", "subject", "finger", "session"])
        w.writeheader()
        w.writerows(rows)

    ids = sorted({r["subject"] for r in rows})
    print(f"图片 {n_img} 张 / 标注 {n_ann} 份 / 身份 {len(ids)} 个 / 每份 {n_pts} 个关键点")
    print(f"每个 .npy 形状: {4 + (n_pts or 0) * 2} (=4 bbox + {n_pts}*2)")
    print(f"产物: {args.out_csv} + {args.landmark_dir}/*.npy")


if __name__ == "__main__":
    main()
