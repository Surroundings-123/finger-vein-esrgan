#!/usr/bin/env python3
"""下载 FingerVeinSyn-5M 前 N 个身份的子集（并发下载 PNG + XML 标注）

关键：绕过 kaggle 1.7.4.5 的单文件下载 bug（它把 file_name 错拼进 URL 路径导致 404）。
正确做法是用 requests 直接请求：
    https://www.kaggle.com/api/v1/datasets/download/{owner}/{dataset}?file_name={URL编码后的完整路径}

为规避 Kaggle 下载端点限流：
    - 并发降到 3
    - 失败按状态码退避重试（最多 3 次）

数据布局（下载后，可直接对接 prepare_fingerveinsyn.py）：
    {out_dir}/
        raw_images/   10001_001_Shift.png ...
        annotations/  10001_001_Shift.xml ...

用法：
    python download_fvsyn_subset.py [身份数] [输出目录]
    例：python download_fvsyn_subset.py 100 /root/autodl-tmp/fvsyn
"""
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from kaggle.api.kaggle_api_extended import KaggleApi

DS = "evanwang98/fingerveinsyn-5m"
BASE = "https://www.kaggle.com/api/v1/datasets/download/" + DS


def main():
    num_ids = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    out_dir = sys.argv[2] if len(sys.argv) > 2 else "/root/autodl-tmp/fvsyn"

    raw_dir = os.path.join(out_dir, "raw_images")
    ann_dir = os.path.join(out_dir, "annotations")
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(ann_dir, exist_ok=True)

    cfg = json.load(open(os.path.expanduser("~/.kaggle/kaggle.json")))
    auth = (cfg["username"], cfg["key"])

    api = KaggleApi()
    api.authenticate()

    # 1. 分页列出文件，收集前 num_ids 个身份的所有文件
    id_re = re.compile(r"/(\d{5})/\d{5}_")
    targets = []
    ids = []
    token = None
    print(f"[1/2] 列出文件，目标 {num_ids} 个身份 ...")
    while True:
        result = api.dataset_list_files(DS, page_token=token, page_size=1000)
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
            if iid not in ids:
                if len(ids) >= num_ids:
                    stop = True
                    break
                ids.append(iid)
            sub = raw_dir if name.lower().endswith(".png") else ann_dir
            targets.append((name, os.path.join(sub, os.path.basename(name))))
        if stop:
            break
        token = getattr(result, "next_page_token", None)
        if not token:
            break

    n_png = sum(1 for n, _ in targets if n.lower().endswith(".png"))
    print(f"    身份 {len(ids)} 个，文件 {len(targets)} 个（PNG {n_png} / XML {len(targets) - n_png}）")
    if not targets:
        print("[错误] 未获取到任何文件")
        return

    # 2. 并发下载（requests + 正确 URL 格式 + 限流退避重试）
    def download(item):
        name, path = item
        if os.path.exists(path) and os.path.getsize(path) > 0:
            return True, 200
        url = BASE + "?file_name=" + requests.utils.quote(name, safe="")
        for attempt in range(3):
            try:
                r = requests.get(url, auth=auth, timeout=120)
                if r.status_code == 200:
                    with open(path, "wb") as fp:
                        fp.write(r.content)
                    return True, 200
                if r.status_code in (429, 500, 502, 503, 504):
                    time.sleep(2 * (attempt + 1))
                    continue
                return False, r.status_code
            except Exception:
                if attempt < 2:
                    time.sleep(2 * (attempt + 1))
                    continue
                return False, -1
        return False, -1

    print("[2/2] 并发下载（3 线程，限流自动重试）...")
    ok = fail = 0
    code_stat = {}
    with ThreadPoolExecutor(max_workers=3) as ex:
        futs = [ex.submit(download, t) for t in targets]
        for i, fut in enumerate(as_completed(futs), 1):
            good, code = fut.result()
            if good:
                ok += 1
            else:
                fail += 1
                code_stat[code] = code_stat.get(code, 0) + 1
            if i % 200 == 0 or i == len(targets):
                print(f"    进度 {i}/{len(targets)}（成功 {ok} / 失败 {fail}）")

    print(f"完成：成功 {ok}，失败 {fail}")
    if code_stat:
        print(f"失败状态码分布：{code_stat}（429=限流 / -1=网络异常）")
    print(f"数据目录：{out_dir}（raw_images/ + annotations/）")


if __name__ == "__main__":
    main()
