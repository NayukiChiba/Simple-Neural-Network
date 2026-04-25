"""
激活层

1. 常见激活函数
2. 激活层的前向与反向传播实现
3. 作为无参数层参与网络计算
"""

import numpy as np

from src.nn.layers.baseLayer import BaseLayer


class ReLULayer(BaseLayer):
    """
    ReLU 激活层

    计算公式:
        output = max(0, input)


    """

    def forward(self, inputData: np.ndarray) -> np.ndarray:
        """
        ReLU 前向传播

        Args:
            inputData (np.ndarray): 输入数据

        Returns:
            np.ndarray: 输出数据

        """
        # 缓存输入数据以供反向传播使用
        self.inputCache = inputData
        outputData = np.maximum(0, inputData)
        self.outputCache = outputData
        return outputData

    def backward(self, outputGradient: np.ndarray) -> np.ndarray:
        """
        反向传播

        Args:
            outputGradient (np.ndarray): 输出梯度

        Returns:
            np.ndarray: 输入梯度


        计算公式:
            inputGradient = outputGradient * (inputData > 0)
        """
        if self.inputCache is None:
            raise ValueError("调用 backward 之前必须先调用 forward")

        # ReLU 的输入梯度等于输出梯度乘以 ReLU 的导数
        reluMask = self.inputCache > 0.0
        inputGradient = outputGradient * reluMask
        return inputGradient


class SigmoidLayer(BaseLayer):
    """
    Sigmoid 激活层

    计算公式:
        output = 1 / (1 + exp(-input))
    """

    def forward(self, inputData: np.ndarray) -> np.ndarray:
        """
        前向传播

        Args:
            inputData (np.ndarray): 输入数据

        Returns:
            np.ndarray: 激活输出数据


        """

        self.inputCache = inputData
        outputData = 1.0 / (1.0 + np.exp(-inputData))
        self.outputCache = outputData
        return outputData

    def backward(self, outputGradient: np.ndarray) -> np.ndarray:
        """
        反向传播

        Args:
            outputGradient (np.ndarray): 输出梯度

        Returns:
            np.ndarray: 输入梯度


        计算公式:
            inputGradient = outputGradient * (outputData * (1 - outputData))
        """
        if self.outputCache is None:
            raise ValueError("调用 backward 之前必须先调用 forward")

        sigmoidOutput = self.outputCache
        inputGradient = outputGradient * sigmoidOutput * (1.0 - sigmoidOutput)
        return inputGradient


class TanhLayer(BaseLayer):
    """
    Tanh 激活层

    计算公式:
        output = tanh(input)
    """

    def forward(self, inputData: np.ndarray) -> np.ndarray:
        """
        前向传播

        Args:
            inputData (np.ndarray): 输入数据

        Returns:
            np.ndarray: 激活后的输出数据
        """
        self.inputCache = inputData
        outputData = np.tanh(inputData)
        self.outputCache = outputData
        return outputData

    def backward(self, outputGradient: np.ndarray) -> np.ndarray:
        """
        反向传播

        Args:
            outputGradient (np.ndarray): 当前层输出梯度

        Returns:
            np.ndarray: 当前层输入梯度
        """
        if self.outputCache is None:
            raise ValueError("调用 backward 之前必须先调用 forward")

        tanhOutput = self.outputCache
        inputGradient = outputGradient * (1.0 - tanhOutput**2)
        return inputGradient
