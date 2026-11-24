# 术视智联 · 手术教学智能眼镜 —— 两篇发明专利复现

本项目复现两篇发明专利的算法流程，代码基于 PyTorch 实现，全程中文注释。

| 专利号 | 主题 | 核心路线 |
|---|---|---|
| ZL201910620017.X | 手指静脉识别（尺度旋转不变性） | **IRCNN 增强 → MTCNN 特征点定位 → 身份比对** |
| ZL201910707218.3 | 视频超分辨率 | **ESRGAN(RRDBNet) → 关键帧修复 + 帧分组采样** |

> 专利本身未新造网络结构，创新点在**处理流水线**（静脉的尺度旋转不变性增强；视频的关键帧修复+帧分组采样）。本仓库按此思路完整实现了网络、训练、数据增强、损失函数与推理流水线。

---

## 一、项目结构

```
default/
├── models/                      # 模型定义层（纯网络结构）
│   ├── __init__.py              # 统一导出入口
│   ├── ir_cnn.py                # 【专利1】IRCNN 图像增强
│   ├── mtcnn.py                 # 【专利1】MTCNN 多任务级联（特征点定位）
│   └── esrgan.py                # 【专利2】RRDBNet + VGG 判别器
│
├── data_prepare_fvusm.py        # 【专利1】FV-USM 数据准备（含合成样图）
├── prepare_fingerveinsyn.py     # 【专利1】FingerVeinSyn-5M 数据适配（含 landmark）
├── train_vein.py                # 【专利1】静脉识别训练
│
├── train_esrgan.py              # 【专利2】视频超分训练
└── video_sr_pipeline.py         # 【专利2】关键帧修复+帧分组 推理流水线
```

## 二、每个文件的作用

### 模型层（models/）

| 文件 | 专利 | 作用 | 关键类 |
|---|---|---|---|
| `models/ir_cnn.py` | 专利1 | 7 层扩张卷积增强网络，对低质量静脉图做锐化/增强，作为 MTCNN 前置 | `IRCNN`（扩张因子 1-2-3-4-3-2-1） |
| `models/mtcnn.py` | 专利1 | 三级级联（P→R→O），每级输出分类+边界框+特征点三分支 | `PNet` / `RNet` / `ONet` / `MTCNN` |
| `models/esrgan.py` | 专利2 | 超分生成器（23 个 RRDB 块，4×放大）+ VGG-128 风格判别器 | `RRDBNet` / `VGGStyleDiscriminator` |
| `models/__init__.py` | 共用 | 导出所有类，`from models import IRCNN, MTCNN, RRDBNet` | — |

### 数据准备层（专利1）

| 文件 | 作用 | 关键函数 |
|---|---|---|
| `data_prepare_fvusm.py` | FV-USM 数据扫描 / 按被试划分 / `VeinDataset` 数据类；`--make_synthetic` 生成合成静脉图（无需真实数据即可跑通） | `scan_dataset` / `split_by_subject` / `VeinDataset` |
| `prepare_fingerveinsyn.py` | 解析 FingerVeinSyn-5M 图片 + XML，提取身份 + 4 个关节腔 landmark，生成 `manifest.csv` + `landmarks/*.npy` | `parse_annotation` |

### 训练/推理层

| 文件 | 专利 | 作用 |
|---|---|---|
| `train_vein.py` | 专利1 | `IRCNN(可选)→ResNet18→分类头` + 可选 landmark/bbox 多任务头；含**尺度旋转不变性增强 + 验证** |
| `train_esrgan.py` | 专利2 | ESRGAN 训练：RRDBNet + 判别器，损失 = L1 + VGG19 感知 + RaGAN |
| `video_sr_pipeline.py` | 专利2 | 视频超分流水线：抽帧 → 关键帧检测 → 关键帧 4×修复 → 帧分组采样 |

---

## 三、环境要求

- Python 3.11 + PyTorch 2.x（CUDA）
- 依赖：`torchvision`、`numpy`、`opencv-python`、`matplotlib`、`tqdm`、`scikit-learn`、`scipy`、`basicsr`（ESRGAN 用）

```bash
pip install basicsr opencv-python scikit-learn tqdm -i https://mirrors.aliyun.com/pypi/simple/
```

> 本地 Windows 环境是独立 Python 3.11（非 conda 管理），位于 `C:\Users\hj\.conda\envs\patent`。本地用 `& $PY` 全路径调用、用 `& $PY -m pip` 装依赖，不走 conda。

---

## 四、本地训练

### 4.1 冒烟测试（快速验证能跑通，几分钟）

```powershell
$PY = "C:\Users\hj\.conda\envs\patent\python.exe"

# 专利1：静脉识别（合成数据）
& $PY data_prepare_fvusm.py --root .\fvusm_syn --make_synthetic --out_csv manifest.csv
& $PY train_vein.py --manifest manifest.csv --epochs 2 --use_enhance --aug_rotation 30 --eval_invariance

# 专利2：视频超分（先抽帧，用 OpenCV 无需 ffmpeg）
& $PY extract_frames.py video.mp4 --out_dir frames
& $PY train_esrgan.py --hr_dir .\frames --epochs 2 --batch_size 4 --no_perceptual
& $PY video_sr_pipeline.py --input .\frames --ckpt checkpoints_esrgan\generator_epoch2.pth
```

### 4.2 完整本地训练命令（正式训练，直接复制）

```powershell
$PY = "C:\Users\hj\.conda\envs\patent\python.exe"
cd C:\Users\hj\Documents\Trae-CN\default

# ========== 专利1：静脉识别（尺度旋转不变性） ==========
# 1) 生成合成数据（换成真实数据时，去掉 --make_synthetic、--root 指向解压目录）
& $PY data_prepare_fvusm.py --root .\fvusm_syn --make_synthetic --out_csv manifest.csv

# 2) 完整训练 30 epoch（IRCNN 增强 + 旋转±30° + 缩放0.8~1.2 + 训练后不变性验证）
& $PY train_vein.py --manifest manifest.csv --epochs 30 --use_enhance `
    --aug_rotation 30 --scale_min 0.8 --scale_max 1.2 --eval_invariance

# ========== 专利2：视频超分（ESRGAN） ==========
# 1) 抽帧（用 OpenCV，无需 ffmpeg）
& $PY extract_frames.py video.mp4 --out_dir frames

# 2) 完整训练 50 epoch（网络稳时去掉 --no_perceptual 以启用 VGG19 感知损失）
& $PY train_esrgan.py --hr_dir .\frames --epochs 50 --batch_size 4 --no_perceptual

# 3) 关键帧修复 + 帧分组流水线
& $PY video_sr_pipeline.py --input .\frames --ckpt checkpoints_esrgan\generator_epoch50.pth
```

### 静脉训练关键参数

| 参数 | 含义 | 默认 |
|---|---|---|
| `--use_enhance` | 启用 IRCNN 增强前端 | 关 |
| `--aug_rotation` | 随机旋转 ±N 度（尺度旋转不变性核心） | 15 |
| `--scale_min/max` | 随机缩放范围 | 0.8 / 1.2 |
| `--eval_invariance` | 训练后自动在旋转/缩放测试集验证不变性 | 关 |
| `--landmark_dir` | 提供 landmark 标注时启用 MTCNN 多任务监督 | 无 |
| `--num_landmarks` | landmark 点数（专利 55；FingerVeinSyn-5M 用 4） | 55 |
| `--size` | 图像缩放目标 (H,W) | 300,100 |

### 超分训练关键参数

| 参数 | 含义 | 默认 |
|---|---|---|
| `--hr_dir` | HR 图像/帧目录（必填） | — |
| `--scale` | 放大倍数 | 4 |
| `--no_perceptual` | 关闭 VGG19 感知损失（免下载 548MB 权重） | 关 |
| `--no_gan` | 只训生成器，不启用对抗 | 关 |
| `--w_l1/w_perceptual/w_gan` | 三部分损失权重 | 1.0 / 1.0 / 0.1 |

---

## 五、AutoDL（4090）完整训练

代码零改动直接上传，仅路径/调用方式不同（Linux 用 `python`，Windows 用 `& $PY`）。

```bash
# 0. 配 Kaggle token（拿到 KGAT_xxx 后）
mkdir -p ~/.kaggle && echo <KGAT_xxx> > ~/.kaggle/access_token && chmod 600 ~/.kaggle/access_token

# 1. 下载 FingerVeinSyn-5M（多身份）
kaggle datasets download -d evanwang98/fingerveinsyn-5m -p /root/autodl-tmp/fvsyn
kaggle datasets download -d evanwang98/fingerveinsyn-5m-part -p /root/autodl-tmp/fvsyn
unzip /root/autodl-tmp/fvsyn/*.zip

# 2. 专利1：真实多身份 + landmark 监督训练
python prepare_fingerveinsyn.py --root <含raw_images+annotations的目录> --out_csv manifest_fvsyn.csv --landmark_dir landmarks
python train_vein.py --manifest manifest_fvsyn.csv --epochs 30 --use_enhance \
    --landmark_dir landmarks --num_landmarks 4 --size 150,300 --eval_invariance

# 3. 专利2：完整 ESRGAN（网络稳时去掉 --no_perceptual 以启用感知损失）
python train_esrgan.py --hr_dir ./frames --epochs 20 --batch_size 8
python video_sr_pipeline.py --input ./video.mp4 --ckpt checkpoints_esrgan/generator_epoch20.pth
```

---

## 六、数据集说明

| 数据集 | 获取方式 | 说明 |
|---|---|---|
| **FV-USM** | 官网 drfendi.com 下载 + 发邮件 `fendi75@gmail.com`（标题 `FV-USM Database`）索要解压密码 | 123 人 / 492 手指 / 5904 张 ROI，300×100 |
| **FingerVeinSyn-5M** | Kaggle：`evanwang98/fingerveinsyn-5m`（+ `-part`） | 5 万身份 × 100 张，600×300，**自带 4 个 landmark 标注** |
| 合成数据 | `data_prepare_fvusm.py --make_synthetic` | 无需下载，本地即时生成，用于冒烟测试 |

---

## 七、已验证结果（RTX 5060 / CUDA）

| 任务 | 结果 |
|---|---|
| 静脉识别（合成，30 epoch） | 最佳验证准确率 **99.17%** |
| 尺度旋转不变性验证 | 原图 99.2% / 旋转±15°、±30° **100%** / 旋转±45° 97.1% / 缩放0.8 **100%** / 缩放1.2 97.5% / 旋转30°+缩放1.2 **100%** |
| 静脉 landmark 多任务（FingerVeinSyn 示例） | IRCNN + landmark/bbox + 分类，loss 正常收敛 |
| ESRGAN 训练 | L1 0.145 → 0.052（2 epoch 冒烟；完整训练更久收敛更好） |
| 关键帧修复流水线 | 64 帧 → 检测 58 关键帧 → 4×修复 → 4 组分组，全流程跑通 |
