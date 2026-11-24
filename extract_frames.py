"""用 OpenCV 从视频抽帧（替代 ffmpeg，无需额外安装）

环境里已装 opencv-python，直接用它抽帧，避免依赖 ffmpeg。

用法：
    python extract_frames.py video.mp4 --out_dir frames
    python extract_frames.py video.mp4 --out_dir frames --max_frames 100
"""
import argparse
import os

import cv2


def main():
    ap = argparse.ArgumentParser(description="用 OpenCV 从视频抽帧（替代 ffmpeg）")
    ap.add_argument("video", help="输入视频路径")
    ap.add_argument("--out_dir", default="frames", help="输出帧目录（默认 frames）")
    ap.add_argument("--max_frames", type=int, default=0, help="最多抽多少帧，0 表示全部")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"无法打开视频：{args.video}")

    n = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        out = os.path.join(args.out_dir, f"f_{n:06d}.png")
        cv2.imwrite(out, frame)
        n += 1
        if args.max_frames and n >= args.max_frames:
            break
    cap.release()
    print(f"抽帧完成：{n} 张，保存在 {args.out_dir}/")


if __name__ == "__main__":
    main()
