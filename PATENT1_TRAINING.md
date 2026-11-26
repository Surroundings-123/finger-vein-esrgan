# 专利1（静脉识别 ZL201910620017.X）完整训练步骤

> 适用环境：AutoDL RTX 4090 24GB，从零开始租机到训练完成。
> 核心链路：租机 → 克隆代码 → 装依赖 → 配 Kaggle 认证 → 下载数据 → 生成 manifest → 训练。

---

## 阶段 0：租机

- 机型 **RTX 4090 24GB**，确认「空闲 GPU ≥ 1」
- 镜像选 **PyTorch 2.x + Python 3.x + CUDA 12.x**（AutoDL 基础镜像，自带 Miniconda 和 torch）
- 数据盘 50GB 够用（下子集 10000 张）

---

## 阶段 1：克隆代码

```bash
cd /root/autodl-tmp
source /etc/network_turbo                      # 加速 GitHub（否则可能 TLS 中断）
git clone https://github.com/Surroundings-123/finger-vein-esrgan.git
cd finger-vein-esrgan
```

---

## 阶段 2：装依赖（torch 不动，只补 numpy/opencv 等）

```bash
pip install -U pip -i https://pypi.tuna.tsinghua.edu.cn/simple/
pip install "numpy<2" "opencv-python<5" scikit-learn tqdm matplotlib scipy kaggle -i https://pypi.tuna.tsinghua.edu.cn/simple/

# 验证
python -c "import torch, cv2, numpy; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'numpy', numpy.__version__)"
```

> `numpy<2` 是关键，否则会破坏 torch 桥接（之前踩过坑）。

---

## 阶段 3：配置 Kaggle 认证（重点，别用错方式）

`download_fvsyn_subset.py` 读的是 **`~/.kaggle/kaggle.json`（username + key）**，不是 access_token：

```bash
mkdir -p ~/.kaggle
cat > ~/.kaggle/kaggle.json << 'EOF'
{"username":"surroundings","key":"<你的Kaggle API Key>"}
EOF
chmod 600 ~/.kaggle/kaggle.json
```

`<你的Kaggle API Key>` 获取方式（二选一）：

- **旧机器复制**：在旧 AutoDL 上 `cat ~/.kaggle/kaggle.json`，把整串内容原样粘过来；
- **重新生成**：Kaggle 网页 → Settings → API → Create New API Token，下载的 kaggle.json 里就是 `username` + `key`。

---

## 阶段 4：下载数据（100 身份 / 10000 张）

```bash
cd /root/autodl-tmp/finger-vein-esrgan
unset http_proxy https_proxy all_proxy        # 关代理，让 Kaggle 走直连
python download_fvsyn_subset.py 100 /root/autodl-tmp/fvsyn
```

产物：`/root/autodl-tmp/fvsyn/raw_images/`（10000 张 600×300 灰度 PNG）。

---

## 阶段 5：生成 manifest

```bash
python prepare_fingerveinsyn.py \
    --root /root/autodl-tmp/fvsyn \
    --out_csv manifest_fvsyn.csv
```

会打印 `图片 10000 张 / 身份 100 个（无 landmark 标注，仅身份分类路线）`。

> 此 Kaggle 版 FingerVeinSyn-5M **只有 PNG、无 XML 标注**，因此走「身份分类」路线，不涉及 MTCNN 55 点 landmark。

---

## 阶段 6：训练（静脉识别 + 尺度旋转不变性）

```bash
python train_vein.py \
    --manifest manifest_fvsyn.csv \
    --epochs 30 \
    --batch_size 64 \
    --use_enhance \
    --size 150,300 \
    --eval_invariance \
    --num_workers 4 \
    --out_dir checkpoints_vein
```

参数含义（关键）：

| 参数 | 值 | 说明 |
|---|---|---|
| `--use_enhance` | 开 | 启用 IRCNN 图像增强前端（专利1 的 IRCNN 部分） |
| `--size` | `150,300` | 缩放目标 `(H,W)`，600×300 缩小一半，保持 2:1 比例 |
| `--eval_invariance` | 开 | 训练后验证尺度旋转不变性（专利1 核心创新） |
| `--aug_rotation` | 默认 15 | 训练随机旋转 ±15°（默认值，无需显式传） |
| `--scale_min/max` | 默认 0.8/1.2 | 训练随机缩放范围（默认值） |
| ~~`--landmark_dir`~~ | **不传** | 本数据集无 landmark 标注，不传走默认身份分类 |

---

## 预期结果

训练日志尾部会出现：

```
训练完成，最佳验证准确率 1.0000，权重保存在 checkpoints_vein/best.pth

==== 尺度旋转不变性验证 ====
  原图               准确率 1.0000
  旋转+15°           准确率 0.9955
  旋转+30°           准确率 0.7660
  旋转+45°           准确率 0.0510
  缩放0.8            准确率 0.9980
  缩放1.2            准确率 0.9975
  旋转30°+缩放1.2      准确率 0.5085
```

「原图 / ±15° / 缩放 0.8~1.2」≈ 1.0 说明模型已学到训练范围内的尺度旋转不变性；±30°、±45° 超出训练增强范围，下降属正常。

---

## 注意事项

1. **Kaggle 认证方式**：`setup_autodl.sh` 第 3 步配的是 `~/.kaggle/access_token`，但 `download_fvsyn_subset.py` 实际读 `~/.kaggle/kaggle.json`（username + key）。请按本文件「阶段 3」手动配置 kaggle.json，不要依赖脚本。
2. **训练完记得关机**（不扣 GPU 费，只扣少量存储费）。
3. **磁盘**：只下子集，定期 `df -h` 检查。
4. **路径**：Linux 路径 `/root/autodl-tmp/...`，与本地 `C:\...` 不同。
