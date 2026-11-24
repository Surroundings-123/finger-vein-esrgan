"""术视智联 —— 专利复现模型包

包含两篇专利的 PyTorch 网络结构：
- IRCNN / MTCNN  ：静脉识别专利（ZL201910620017.X）
- RRDBNet / VGG   ：视频超分专利（ZL201910707218.3，基于 ESRGAN）
"""
from .ir_cnn import IRCNN
from .mtcnn import MTCNN, PNet, RNet, ONet
from .esrgan import RRDBNet, VGGStyleDiscriminator

__all__ = [
    "IRCNN",
    "MTCNN", "PNet", "RNet", "ONet",
    "RRDBNet", "VGGStyleDiscriminator",
]
