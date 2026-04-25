"""
损失函数相关子包。

功能：
1. 定义分类任务损失函数
2. 定义回归任务损失函数
3. 为反向传播提供损失梯度入口
"""

from src.nn.losses.crossEntropyLoss import CrossEntropyLoss

__all__: list[str] = ["CrossEntropyLoss"]
