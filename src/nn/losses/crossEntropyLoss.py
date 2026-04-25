"""
交叉熵损失函数的实现

1. 实现 softmax 和 cross-entropy 的联合损失
2. 支持前向传播计算损失值
3. 支持对 logits 的反向传播计算输入梯度
"""

import numpy as np


class CrossEntropyLoss:
    """
    交叉熵损失函数

    说明
        1. 输入为模型输出的 logits, 而不是softmax 后的概率分布
        2. 标签应该是整数形式的类别索引, 而不是 one-hot 编码
        3. 内部自动完成 softmax 和交叉熵的计算, 以提高数值稳定性

    计算公式:
    1. softmax(logits) = exp(logits) / sum(exp(logits))
    2. cross_entropy_loss = -log(softmax(logits)[true_class])

    """

    def __init__(self, epsilon: float = 1e-12) -> None:
        """
        初始化交叉熵损失函数

        Args:
            epsilon(float): 用于数值稳定性的微小常数, 避免 log(0) 的情况
        """
        if epsilon <= 0.0:
            raise ValueError("epsilon 必须是一个正数")

        self.epsilon = epsilon
        # 缓存 softmax 输出的概率分布和标签, 以供反向传播使用
        self.probabilities: np.ndarray | None = None
        # 标签应该是整数形式的类别索引, 而不是 one-hot 编码
        self.targetLabels: np.ndarray | None = None
        self.batchSize: int | None = None

    def forward(self, logits: np.ndarray, targetLabels: np.ndarray) -> float:
        """
        计算交叉熵损失

        Args:
            logits (np.ndarray): 模型输出的 logits, 形状为 (batchSize, numClasses)
            targetLabels (np.ndarray): 真实标签的类别索引, 形状为 (batchSize,)

        Returns:
            float: 当前批次的平均损失


        """
        if logits.ndim != 2:
            raise ValueError("logits 必须是二维数组")

        if targetLabels.ndim != 1:
            raise ValueError("targetLabels 必须是一维数组")

        batchSize, classCount = logits.shape

        if batchSize == 0:
            raise ValueError("batchSize 必须大于 0")

        if classCount == 0:
            raise ValueError("classCount 必须大于 0")

        if targetLabels.shape[0] != batchSize:
            raise ValueError(
                f"targetLabels 的 batchSize 不匹配, 期望 {batchSize}, 实际为 {targetLabels.shape[0]}"
            )

        if not np.issubdtype(targetLabels.dtype, np.integer):
            raise ValueError("targetLabels 必须是整数类型的数组")

        if np.any(targetLabels < 0) or np.any(targetLabels >= classCount):
            raise ValueError("targetLabels 中的类别索引必须在 [0, classCount) 范围内")

        # 为了数值稳定性, 在计算 softmax 之前先减去每行的最大值
        shiftedLogits = logits - np.max(logits, axis=1, keepdims=True)

        # 计算 softmax 输出的指数部分
        expLogits = np.exp(shiftedLogits)
        # 计算 softmax 输出的概率分布
        probabilities = expLogits / np.sum(expLogits, axis=1, keepdims=True)

        # 缓存 softmax 输出的概率分布和标签, 以供反向传播使用
        selectedprobabilities = probabilities[np.arange(batchSize), targetLabels]
        # 为了数值稳定性, 避免 log(0) 的情况, 对概率值进行裁剪
        clippedprobabilities = np.clip(selectedprobabilities, self.epsilon, 1.0)

        loss = -np.mean(np.log(clippedprobabilities))
        # 缓存 softmax 输出的概率分布和标签, 以供反向传播使用
        self.probabilities = probabilities
        self.targetLabels = targetLabels
        self.batchSize = batchSize

        return float(loss)

    def backward(self) -> np.ndarray:
        """
        计算输入 logits 的梯度

        Returns:
            np.ndarray: 输入 logits 的梯度, 形状为 (batchSize, numClasses)

        计算公式:
            inputGradient = (softmax(logits) - one_hot(targetLabels)) / batchSize
        """
        if self.probabilities is None:
            raise ValueError("调用 backward 之前必须先调用 forward")
        if self.targetLabels is None:
            raise ValueError("targetLabels 缓存为空，无法执行 backward")
        if self.batchSize is None:
            raise ValueError("batchSize 缓存为空，无法执行 backward")

        inputGradient = self.probabilities.copy()
        inputGradient[np.arange(self.batchSize), self.targetLabels] -= 1.0
        inputGradient /= self.batchSize

        return inputGradient
