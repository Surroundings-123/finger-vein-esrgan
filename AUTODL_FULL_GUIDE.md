# AutoDL 完整训练指南（专利1 静脉识别 + 专利2 视频超分）

> 权威版，适用于全新 RTX 4090 从零跑通两个专利。
> 代码仓库：https://github.com/Surroundings-123/finger-vein-esrgan.git

---

## 〇、租新机前：必须保存的东西（先做，否则会丢）

### 1. Kaggle 认证（最关键，唯一凭证）

数据下载靠 `~/.kaggle/kaggle.json`（username + key）。旧机器释放后会丢失，**关机前先备份**：

```bash
cat ~/.kaggle/kaggle.json
```

把输出内容（形如 `{"username":"surroundings","key":"xxxxx"}`）保存到本地记事本/文件。

> 若已无法登旧机器：Kaggle 网页 → Settings → API → Create New API Token，重新下载 kaggle.json。

### 2. 训练结果

`results.tar.gz`（含 checkpoints_vein/best.pth、checkpoints_esrgan/generator_epoch50.pth、sr_output、manifest_fvsyn.csv）在 AutoDL 控制台「下载」到本地即可。

### 3. 代码（已在 GitHub，无需手动保存）

仓库 `Surroundings-123/finger-vein-esrgan` 已包含全部代码（train_vein.py / train_esrgan.py / video_sr_pipeline.py / models/ / 下载脚本 / prepare 脚本）。新机器 `git clone` 即可。

### 可重新生成（无需保存）

- 静脉图数据（10000 张）：用 `download_fvsyn_subset.py` 重新下载
- 依赖环境：重新安装

---

## 一、租机

- 机型 **RTX 4090 24GB**，确认「空闲 GPU ≥ 1」
- 镜像 **PyTorch 2.x + Python 3.x + CUDA 12.x**（自带 Miniconda + torch）
- 数据盘 50GB 够用

---

## 二、克隆代码 + 装依赖

```bash
cd /root/autodl-tmp
source /etc/network_turbo                      # 加速 GitHub
git clone https://github.com/Surroundings-123/finger-vein-esrgan.git
cd finger-vein-esrgan

pip install -U pip -i https://pypi.tuna.tsinghua.edu.cn/simple/
pip install "numpy<2" "opencv-python<5" scikit-learn tqdm matplotlib scipy kaggle -i https://pypi.tuna.tsinghua.edu.cn/simple/

python -c "import torch, cv2, numpy; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'numpy', numpy.__version__)"
```

> `numpy<2` 是关键，否则破坏 torch 桥接。

---

## 三、配置 Kaggle 认证

```bash
mkdir -p ~/.kaggle
cat > ~/.kaggle/kaggle.json << 'EOF'
{"username":"surroundings","key":"<你的Kaggle API Key>"}
EOF
chmod 600 ~/.kaggle/kaggle.json
```

`<你的Kaggle API Key>` 用「〇-1」备份的值填入。

> 注意：`download_fvsyn_subset.py` 读的是 `kaggle.json`（username + key），不是 access_token。`setup_autodl.sh` 里写的是 access_token 方式，与下载脚本不匹配，请用上面的手动方式。

---

## 四、专利1 静脉识别（ZL201910620017.X）

### 4.1 下载数据（100 身份 / 10000 张）

```bash
cd /root/autodl-tmp/finger-vein-esrgan
unset http_proxy https_proxy all_proxy        # 关代理，Kaggle 走直连
python download_fvsyn_subset.py 100 /root/autodl-tmp/fvsyn
```

产物：`/root/autodl-tmp/fvsyn/raw_images/`（10000 张 600×300 灰度 PNG）。

### 4.2 生成 manifest

```bash
python prepare_fingerveinsyn.py \
    --root /root/autodl-tmp/fvsyn \
    --out_csv manifest_fvsyn.csv
```

会打印 `图片 10000 张 / 身份 100 个（无 landmark 标注，仅身份分类路线）`。

> 此 Kaggle 版 FingerVeinSyn-5M 只有 PNG、无 XML 标注，走「身份分类」路线，不涉及 MTCNN 55 点 landmark。

### 4.3 训练

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

参数：`--use_enhance` 启用 IRCNN 前端；`--size 150,300` 是 (H,W)；`--eval_invariance` 训练后验证尺度旋转不变性；**不传 `--landmark_dir`**。

### 4.4 预期结果

```
训练完成，最佳验证准确率 1.0000，权重保存在 checkpoints_vein/best.pth

==== 尺度旋转不变性验证 ====
  原图               准确率 1.0000
  旋转+15°           准确率 0.9955
  缩放0.8            准确率 0.9980
  缩放1.2            准确率 0.9975
  ...
```

---

## 五、专利2 视频超分（ZL201910707218.3）

### 5.1 训练 ESRGAN（RRDBNet）

```bash
cd /root/autodl-tmp/finger-vein-esrgan

python train_esrgan.py \
    --hr_dir /root/autodl-tmp/fvsyn/raw_images \
    --epochs 50 \
    --batch_size 16 \
    --patch_size 128 \
    --scale 4 \
    --num_workers 4 \
    --out_dir checkpoints_esrgan
```

- VGG19 感知损失若下载失败会自动禁用（降级为 L1 + RaGAN），不影响流程。
- 显存不足时 `batch_size` 降到 8。
- 权重产物：`checkpoints_esrgan/generator_epoch50.pth`。

### 5.2 关键帧修复 + 帧分组采样推理

```bash
# 用同一身份 100 张 variation 图模拟「连续视频帧」
mkdir -p /root/autodl-tmp/fvsyn/frames_seq
cp /root/autodl-tmp/fvsyn/raw_images/10001_*.png /root/autodl-tmp/fvsyn/frames_seq/

python video_sr_pipeline.py \
    --input /root/autodl-tmp/fvsyn/frames_seq \
    --ckpt checkpoints_esrgan/generator_epoch50.pth \
    --scale 4 \
    --diff_thresh 30 \
    --group_size 16 \
    --out_dir sr_output
```

预期：`共 100 帧 → 检测到 N 个关键帧 → 帧分组 M 组 → sr_output/key_*.png + groups.txt`。

> `10001` 若不在下载的身份里，换成实际存在的 ID（`ls /root/autodl-tmp/fvsyn/raw_images/ | head` 看开头编号）。

---

## 六、收尾（打包下载 + 关机）

```bash
cd /root/autodl-tmp/finger-vein-esrgan
tar -czf results.tar.gz checkpoints_vein checkpoints_esrgan sr_output manifest_fvsyn.csv
```

- AutoDL 控制台「下载」`results.tar.gz` 到本地。
- 下载完成后点「**关机**」（只扣存储费，不扣 GPU 费）。

---

## 关键提醒

| 项 | 要点 |
|---|---|
| Kaggle 认证 | 关机前先 `cat ~/.kaggle/kaggle.json` 备份 |
| 磁盘 | 只下子集，定期 `df -h` |
| 计费 | ¥2.18/时，训练完关机 |
| 路径 | Linux `/root/autodl-tmp/...` |
