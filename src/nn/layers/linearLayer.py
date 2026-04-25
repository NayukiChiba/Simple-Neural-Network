"""
线形层

1. 实现全连接线性变换
2. 支持前向传播和反向传播
3. 管理权重、偏置参数和对应的梯度

"""

from typing import Optional

import numpy as np

from src.nn.layers.baseLayer import BaseLayer


class LinearLayer(BaseLayer):
    """
    全连接线性层

    计算公式:
        output = input @ weights + bias


    """

    def __init__(
        self,
        inputDim: int,
        outputDim: int,
        useBias: bool = True,
        randomSeed: Optional[int] = None,
    ) -> None:
        """
        初始化全连接线性层

        Args:
            inputDim(int): 输入维度
            outputDim(int): 输出维度
            useBias(bool): 是否使用偏置
            randomSeed(Optional[int]): 随机种子
        """

        super().__init__()

        if inputDim <= 0:
            raise ValueError("inputDim 必须大于0")
        if outputDim <= 0:
            raise ValueError("outputDim 必须大于0")

        self.inputDim = inputDim
        self.outputDim = outputDim
        self.useBias = useBias

        rng = np.random.default_rng(randomSeed)

        # 使用 Xavier 均匀初始化, 适合多层感知机
        limit = np.sqrt(6.0 / (inputDim + outputDim))
        self.weights = rng.uniform(
            low=-limit,
            high=limit,
            size=(inputDim, outputDim),
        ).astype(np.float64)

        # 初始化梯度矩阵为0
        self.gradWeights = np.zeros_like(self.weights)

        if self.useBias:
            self.bias = np.zeros(shape=(1, outputDim), dtype=np.float64)
            self.gradBias = np.zeros_like(self.bias)
        else:
            self.bias = None
            self.gradBias = None

    def forward(self, inputData: np.ndarray) -> np.ndarray:
        """
        前向传播

        Args:
            inputData (np.ndarray): 输入数据, 形状为 (batchSize, inputDim)

        Returns:
            np.ndarray: 输出数据, 形状为 (batchSize, outputDim)


        """

        if inputData.ndim != 2:
            raise ValueError("LinearLayer 的输入必须是二维数组")

        if inputData.shape[1] != self.inputDim:
            raise ValueError(
                f"输入的维度不匹配, 期望 {self.inputDim}, 实际为 {inputData.shape[1]}"
            )

        self.inputCache = inputData

        outputData = inputData @ self.weights

        if self.useBias and self.bias is not None:
            outputData = outputData + self.bias

        self.outputCache = outputData

        return outputData

    def backward(self, outputGradient: np.ndarray) -> np.ndarray:
        """
        反向传播

        Args:
            outputGradient(np.ndarray): 当前层输出的梯度, 形状为 (batchSize, outputDim)

        Returns:
            np.ndarray: 当前层输入的梯度, 形状为 (batchSize, inputDim)



        """
        if self.inputCache is None:
            raise ValueError("调用 backward 之前必须先调用 forward")

        if outputGradient.ndim != 2:
            raise ValueError("LinearLayer 的输出梯度必须是二维数组")

        # pr #8: 输出梯度的 batchSize 必须与输入数据的 batchSize 匹配
        if outputGradient.shape[0] != self.inputCache.shape[0]:
            raise ValueError(
                f"输出梯度的 batchSize 不匹配, 期望 {self.inputCache.shape[0]}, 实际为 {outputGradient.shape[0]}"
            )

        if outputGradient.shape[1] != self.outputDim:
            raise ValueError(
                f"输出梯度的维度不匹配, 期望 {self.outputDim}, 实际为 {outputGradient.shape[1]}"
            )

        # 计算输入梯度
        inputData = self.inputCache

        # 对输入的梯度: dL/dInput = dL/dOutput @ W^T
        inputGradient = outputGradient @ self.weights.T

        # 计算权重梯度: dL/dW = Input^T @ dL/dOutput
        self.gradWeights[...] = inputData.T @ outputGradient

        # 计算偏置梯度: dL/dBias = sum(dL/dOutput, axis=0, keepdims=True)
        if self.useBias and self.gradBias is not None:
            self.gradBias[...] = np.sum(outputGradient, axis=0, keepdims=True)

        return inputGradient

    def getParameters(self) -> list[np.ndarray]:
        """
        获取层的参数

        Returns:
            list[np.ndarray]: 参数列表
        """
        if self.useBias and self.bias is not None:
            return [self.weights, self.bias]
        return [self.weights]

    def getGradients(self) -> list[np.ndarray]:
        """
        获取层的梯度

        Returns:
            list[np.ndarray]: 梯度列表
        """
        if self.useBias and self.gradBias is not None:
            return [self.gradWeights, self.gradBias]
        return [self.gradWeights]
