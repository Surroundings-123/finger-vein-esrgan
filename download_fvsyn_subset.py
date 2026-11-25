#!/usr/bin/env python3
"""下载 FingerVeinSyn-5M 前 N 个身份的子集（并发下载 PNG + XML 标注）

数据布局（下载后，可直接对接 prepare_fingerveinsyn.py）：
    {out_dir}/
        raw_images/   10001_001_Shift.png ...
        annotations/  10001_001_Shift.xml ...

用法：
    python download_fvsyn_subset.py [身份数] [并发线程数] [输出目录]
    例：python download_fvsyn_subset.py 100 8 /root/autodl-tmp/fvsyn
"""
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

from kaggle.api.kaggle_api_extended import KaggleApi

DATASET = "evanwang98/fingerveinsyn-5m"


def main():
    num_ids = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    out_dir = sys.argv[3] if len(sys.argv) > 3 else "/root/autodl-tmp/fvsyn"

    api = KaggleApi()
    api.authenticate()

    raw_dir = os.path.join(out_dir, "raw_images")
    ann_dir = os.path.join(out_dir, "annotations")
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(ann_dir, exist_ok=True)

    # 1. 分页列出文件，收集前 num_ids 个身份的所有文件（PNG 与 XML）
    id_re = re.compile(r"/(\d{5})/\d{5}_")
    targets = []          # [(远程名, 本地basename, 目标子目录)]
    seen_ids = []
    page_token = None
    print(f"[1/2] 列出文件，目标 {num_ids} 个身份 ...")

    while True:
        result = api.dataset_list_files(DATASET, page_token=page_token, page_size=1000)
        batch = getattr(result, "files", []) or []
        if not batch:
            break

        stop = False
        for f in batch:
            name = f.name if hasattr(f, "name") else str(f)
            m = id_re.search(name)
            if not m:
                continue
            iid = m.group(1)
            if iid not in seen_ids:
                if len(seen_ids) >= num_ids:
                    stop = True          # 已收集够身份，退出
                    break
                seen_ids.append(iid)
            base = os.path.basename(name)
            sub = raw_dir if name.lower().endswith(".png") else ann_dir
            targets.append((name, base, sub))

        if stop:
            break
        page_token = getattr(result, "next_page_token", None)
        if not page_token:
            break

    print(f"    身份 {len(seen_ids)} 个，文件 {len(targets)} 个")
    if not targets:
        print("[错误] 未获取到任何文件，请检查 kaggle 认证与数据集 ref")
        return

    # 2. 并发下载
    def download(item):
        remote, base, sub = item
        try:
            api.dataset_download_file(DATASET, remote, path=sub, quiet=True)
            return True, base
        except Exception as e:
            return False, f"{base}: {e}"

    print(f"[2/2] 并发下载（{workers} 线程）...")
    ok = fail = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(download, t) for t in targets]
        for i, fut in enumerate(as_completed(futs), 1):
            good, info = fut.result()
            if good:
                ok += 1
            else:
                fail += 1
                if fail <= 8:
                    print(f"    [失败] {info}")
            if i % 200 == 0 or i == len(targets):
                print(f"    进度 {i}/{len(targets)}（成功 {ok} / 失败 {fail}）")

    print(f"完成：成功 {ok}，失败 {fail}")
    print(f"数据目录：{out_dir}（raw_images/ + annotations/）")
    print("下一步：python prepare_fingerveinsyn.py --root %s --out_csv manifest_fvsyn.csv --landmark_dir landmarks" % out_dir)


if __name__ == "__main__":
    main()
