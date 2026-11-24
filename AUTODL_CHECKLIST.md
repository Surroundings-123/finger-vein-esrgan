# AutoDL 完整操作清单

从租机到训练结果，按阶段一步步照做即可。

---

## 阶段 0：租机 + 选镜像

1. 租一台 **RTX 4090 24GB**，确认 **空闲GPU ≥ 1**（空闲 0/8 的机型租不了）
2. 镜像下拉选：**PyTorch 2.x + Python 3.11 + CUDA 12.x**（基础镜像即可，自带 Miniconda）
3. 数据盘 50GB 够用（下子集），直接「创建并开机」

---

## 阶段 1：上传代码到 AutoDL

```bash
# 方式A：本地打包上传（AutoDL 控制台右上角「上传」，传 zip 后解压）
# 本地先打 zip（PowerShell）：
#   Compress-Archive -Path .\* -DestinationPath project.zip
# 上传后：
cd /root/autodl-tmp && unzip project.zip -d project

# 方式B：若代码已推 GitHub/Gitee
git clone <你的仓库地址> /root/autodl-tmp/project
```

---

## 阶段 2：一键环境配置

```bash
cd /root/autodl-tmp/project
export KAGGLE_API_TOKEN=KGAT_你的token
bash setup_autodl.sh
```

> 脚本会自动：检查 CUDA → 装依赖（basicsr/opencv/sklearn/tqdm/kaggle）→ 配 Kaggle token → 建目录 → patch basicsr → 验证模型可导入。

---

## 阶段 3：下载数据（子集，注意 80GB 上限）

```bash
# 包1：原始静脉图（文件名自带身份 ID，够做身份分类+不变性）
kaggle datasets download -d evanwang98/fingerveinsyn-5m -p /root/autodl-tmp/fvsyn

# 包2：标注(annotations) + mask（要做 landmark 多任务才需要）
kaggle datasets download -d evanwang98/fingerveinsyn-5m-part -p /root/autodl-tmp/fvsyn
```

> 下载前先 `df -h` 看空间。**优先下包1**（身份分类 + 尺度旋转不变性只需图片，文件名带 ID）；要做 MTCNN landmark 分支才需要包2 的 XML 标注。空间不足就只下包1。

---

## 阶段 4：解压 + 数据预处理

```bash
cd /root/autodl-tmp/fvsyn
unzip fingerveinsyn-5m.zip
unzip fingerveinsyn-5m-part.zip

# 确认目录结构（看 raw_images/ 和 annotations/ 在哪一层）
ls -R | head -50

# 生成 manifest + landmarks
cd /root/autodl-tmp/project
python prepare_fingerveinsyn.py --root <含raw_images和annotations的目录> \
    --out_csv manifest_fvsyn.csv --landmark_dir landmarks
```

---

## 阶段 5：训练（专利1 + 专利2）

```bash
# ===== 专利1：静脉识别（多身份 + 尺度旋转不变性） =====
python train_vein.py --manifest manifest_fvsyn.csv --epochs 30 --use_enhance \
    --landmark_dir landmarks --num_landmarks 4 --size 150,300 --eval_invariance

# ===== 专利2：视频超分（ESRGAN） =====
python extract_frames.py video.mp4 --out_dir frames    # 抽帧（OpenCV）
python train_esrgan.py --hr_dir ./frames --epochs 50 --batch_size 8
python video_sr_pipeline.py --input ./frames --ckpt checkpoints_esrgan/generator_epoch50.pth
```

> 4090 显存 24GB，`--batch_size` 可以开到 8~16，比本地快很多。ESRGAN 网络稳时去掉 `--no_perceptual` 可启用 VGG19 感知损失。

---

## 阶段 6：下载结果回本地

```bash
# 训练完后，把权重和日志打包下载（AutoDL 控制台文件列表可右键下载）
cd /root/autodl-tmp/project
tar -czf results.tar.gz checkpoints_vein checkpoints_esrgan manifest_fvsyn.csv
```

---

## 关键提醒

| 项 | 要点 |
|---|---|
| 磁盘 | 80GB 上限，只下子集，定期 `df -h` 检查 |
| 计费 | ¥2.18/时，训练完记得**关机**（关机不扣 GPU 费，只扣少量存储费） |
| 镜像 | 没选框架前不能开机 |
| 路径 | Linux 路径 `/root/autodl-tmp/...`，和本地 `C:\...` 不同 |
