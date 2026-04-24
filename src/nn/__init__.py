"""
简单神经网络核心包。

功能：
1. 组织数据、层、损失函数、模型与训练流程
2. 为 main.py 提供统一的包级入口
"""

from . import data, layers, losses, models, optimizers, persistence, training

__all__ = [
    "data",
    "layers",
    "losses",
    "models",
    "optimizers",
    "persistence",
    "training",
]
