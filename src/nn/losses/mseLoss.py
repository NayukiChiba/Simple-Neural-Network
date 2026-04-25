"""
均方误差函数的实现

1. 实现回归任务的均方误差损失函数
2. 支持前向传播计算损失值
3. 支持反向传播计算梯度值
"""

import numpy as np


class MSELoss:
    """
    均方误差损失函数类

    说明:
        1. 输入为模型的预测值和真实目标值
        2. 预测值和目标值必须具有相同的形状
        3. 返回整个批次的平均损失值

    计算公式:
        loss = mean((predictions - targets) ** 2)
    """

    def __init__(self):
        """
        初始化均方误差损失函数

        """
        # 存储预测值和目标值以供反向传播使用
        self.predictions: np.ndarray | None = None
        self.targets: np.ndarray | None = None
        # 存储元素数量以计算平均损失
        self.elementCount: int | None = None

    def forward(self, predictions: np.ndarray, targets: np.ndarray) -> float:
        """
        计算均方误差损失

        Args:
            predictions (np.ndarray): 模型预测值
            targets (np.ndarray): 真实目标值

        Returns:
            float: 当前批次的平均损失
        """
        if predictions.ndim == 0:
            raise ValueError("predictions 不能是标量")
        if targets.ndim == 0:
            raise ValueError("targets 不能是标量")

        if predictions.shape != targets.shape:
            raise ValueError(
                "predictions 和 targets 的形状必须相同"
                f"但得到 predictions.shape={predictions.shape}"
                f"targets.shape={targets.shape}"
            )

        if predictions.size == 0:
            raise ValueError("predictions 不能为空")

        # 计算元素数量以便后续平均
        difference = predictions - targets
        loss = np.mean(difference**2)

        self.predictions = predictions
        self.targets = targets
        self.elementCount = predictions.size

        return float(loss)

    def backward(self) -> np.ndarray:
        """
        计算损失函数对预测值的梯度

        Returns:
            np.ndarray: 预测值对应的梯度

        计算公式:
            dLoss / dPredictions = 2 * (predictions - targets) / N
        """
        if self.predictions is None:
            raise ValueError("必须先调用 forward 方法计算损失值")
        if self.targets is None:
            raise ValueError("targets 缓存为空, 无法执行 backward")
        if self.elementCount is None:
            raise ValueError("elementCount 缓存为空, 无法执行 backward")

        inputGradient = 2 * (self.predictions - self.targets) / self.elementCount

        return inputGradient
