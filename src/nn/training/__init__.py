"""
训练流程相关子包。

功能：
1. 管理训练循环
2. 管理评估指标
3. 组织日志记录与训练调度
"""

from src.nn.training.metrics import calculateAccuracy, calculateMeanSquaredError
from src.nn.training.trainer import Trainer

__all__: list[str] = ["Trainer", "calculateAccuracy", "calculateMeanSquaredError"]
