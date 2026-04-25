"""
src/nn/layers/baseLayer.py
神经网络层的基类

1. 定义神经网络层的基本接口
2. 包含前向传播和反向传播的抽象方法
3. 提供参数、梯度和更新方法的基础实现
"""

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np


class BaseLayer(ABC):
    """
    网络层基类

    1. 所有具体网络层都应该继承自这个基类
    2. forward 负责前向传播
    3. backward 负责反向传播
    4. 无参数层可以直接使用默认的参数和梯度实现

    """

    def __init__(self) -> None:
        """
        初始化基础网络层状态


        """
        self.isTraining = True  # 是否处于训练模式
        self.inputCache: Optional[np.ndarray] = None  # 前向传播时缓存输入数据
        self.outputCache: Optional[np.ndarray] = None  # 前向传播时缓存输出数据

    @abstractmethod
    def forward(self, inputData: np.ndarray) -> np.ndarray:
        """
        前向传播方法

        Args:
            inputData (np.ndarray): 输入数据

        Returns:
            np.ndarray: 输出数据
        """
        raise NotImplementedError("forward方法必须在子类中实现")

    @abstractmethod
    def backward(self, outputGradient: np.ndarray) -> np.ndarray:
        """
        反向传播方法

        Args:
            outputGradient (np.ndarray): 输出梯度

        Returns:
            np.ndarray: 输入梯度
        """
        raise NotImplementedError("backward方法必须在子类中实现")

    def getParameters(self) -> list[np.ndarray]:
        """
        获取层的参数

        Returns:
            list[np.ndarray]: 参数列表
        """
        return []

    def getGradients(self) -> list[np.ndarray]:
        """
        获取层的梯度

        Returns:
            list[np.ndarray]: 梯度列表
        """
        return []

    def zeroGrad(self) -> None:
        """
        将层的梯度清零
        """
        for gradient in self.getGradients():
            gradient.fill(0.0)

    def train(self) -> None:
        """
        切换到训练模式
        """
        self.isTraining = True

    def eval(self) -> None:
        """
        切换到推理模式
        """
        self.isTraining = False

    def hasParameters(self) -> bool:
        """
        判断层是否有可训练参数

        Returns:
            bool: 如果层有参数返回True，否则返回False
        """
        return len(self.getParameters()) > 0
