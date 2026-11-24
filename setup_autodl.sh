#!/bin/bash
# setup_autodl.sh —— AutoDL（RTX 4090）一键环境配置脚本
#
# 用法（在项目根目录，即含 models/ 和 *.py 的那层执行）：
#   bash setup_autodl.sh
#
# 可选：先导出 Kaggle token 以自动配置下载凭证：
#   export KAGGLE_API_TOKEN=KGAT_xxx && bash setup_autodl.sh

set -e

echo "===== [1/5] 检查 GPU 与 PyTorch ====="
nvidia-smi || echo "警告：未检测到 GPU"
python -c "import torch; print('torch', torch.__version__); print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"

echo ""
echo "===== [2/5] 安装依赖（numpy<2 + opencv<5 以兼容 torch） ====="
pip install -U pip -i https://pypi.tuna.tsinghua.edu.cn/simple/
pip install "numpy<2" "opencv-python<5" scikit-learn tqdm matplotlib scipy kaggle -i https://pypi.tuna.tsinghua.edu.cn/simple/

echo ""
echo "===== [3/5] 配置 Kaggle token ====="
if [ -n "$KAGGLE_API_TOKEN" ]; then
    mkdir -p ~/.kaggle
    echo "$KAGGLE_API_TOKEN" > ~/.kaggle/access_token
    chmod 600 ~/.kaggle/access_token
    echo "已写入 ~/.kaggle/access_token"
    kaggle datasets list 2>/dev/null | head -5 || echo "（token 已写入，可手动运行 kaggle datasets list 验证）"
else
    echo "未设置 KAGGLE_API_TOKEN，跳过。需要下载数据时执行："
    echo "  mkdir -p ~/.kaggle && echo <KGAT_xxx> > ~/.kaggle/access_token && chmod 600 ~/.kaggle/access_token"
fi

echo ""
echo "===== [4/5] 创建数据/输出目录 ====="
mkdir -p /root/autodl-tmp/fvsyn checkpoints_vein checkpoints_esrgan landmarks frames
echo "已创建：/root/autodl-tmp/fvsyn + checkpoints_vein/checkpoints_esrgan/landmarks/frames"

echo ""
echo "===== [5/5] 验证导入 ====="
python -c "from models import IRCNN, MTCNN, RRDBNet, VGGStyleDiscriminator; print('模型导入成功')"
python -c "import cv2, sklearn, tqdm, scipy, numpy; print('依赖导入成功, numpy', numpy.__version__)"

echo ""
echo "========== 配置完成 =========="
echo "下一步见 README.md 第五节：下载数据 + 完整训练"
