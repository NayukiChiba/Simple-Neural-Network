"""
优化器相关子包。

功能：
1. 管理参数更新策略
2. 为训练循环提供统一优化接口
3. 负责根据梯度更新模型参数
"""

from src.nn.optimizers.sgdOptimizer import SGDOptimizer

__all__: list[str] = ["SGDOptimizer"]
