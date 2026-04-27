"""
src/nn/models/sequentialModel.py
顺序模型模块

1. 按顺序组织多个网络层
2. 提供模型级别的前向处理和反向传播
3. 聚合所有层的参数、梯度和预测能力


"""

from typing import Iterable

import numpy as np

from src.nn.layers.baseLayer import BaseLayer


class SequentialModel(BaseLayer):
    """
    顺序模型容器

    说明:
        1. 按顺序组织多个网络层
        2. 反向传播时，按照层的顺序反向传播
        3. 整个模型的参数和梯度由所有层的参数和梯度组成


    """

    def __init__(self, layers: Iterable[BaseLayer] | None = None) -> None:
        """
        初始化顺序模型

        参数:
            layers(): 可选的层列表，按顺序组织
        """
        super().__init__()
        self.layers: list[BaseLayer] = []
        if layers is not None:
            for layer in layers:
                self.add_layer(layer)

    def add_layer(self, layer: BaseLayer) -> None:
        """
        向模型末尾加一个wangluoceng
        Args:
            layer(BaseLayer): 要添加的层实例

        """
        if not isinstance(layer, BaseLayer):
            raise TypeError("只能添加BaseLayer的子类实例")

        if layer is self:
            raise ValueError("不能将模型添加到自身")

        # 如果模型正在训练，设置新层为训练模式，否则设置为评估模式
        if self.isTraining:
            layer.train()
        else:
            layer.eval()

        self.layers.append(layer)

        # 模型结构发生变化，重置参数和梯度
        self.inputCache = None
        self.outputCache = None

    def forward(self, inputData: np.ndarray) -> np.ndarray:
        """
        前向传播，按顺序通过每层处理输入数据

        Args:
            inputData(np.ndarray): 输入数据，形状为(batch_size, input_dim)

        Returns:
            np.ndarray: 输出数据，形状为(batch_size, output_dim)
        """
        if len(self.layers) == 0:
            raise ValueError("模型中没有层，无法进行前向传播")

        self.inputCache = inputData

        outputData = inputData
        for layer in self.layers:
            outputData = layer.forward(outputData)

        self.outputCache = outputData
        return outputData

    def backward(self, outputGradient: np.ndarray) -> np.ndarray:
        """
        反向传播，按层的顺序反向传播输出梯度
        Args:
            outputGradient(np.ndarray): 输出层的梯度，形状为(batch_size, output_dim)
        Returns:
            np.ndarray: 输入层的梯度，形状为(batch_size, input_dim)


        """
        if len(self.layers) == 0:
            raise ValueError("模型中没有层，无法进行反向传播")

        if self.outputCache is None:
            raise ValueError("没有前向传播的输出缓存，无法进行反向传播")

        # 从输出层开始，逐层反向传播梯度
        inputGradient = outputGradient
        for layer in reversed(self.layers):
            inputGradient = layer.backward(inputGradient)

        return inputGradient

    def getParameters(self) -> list[np.ndarray]:
        """
        获取模型中所有层的参数列表
        Returns:
            list[np.ndarray]: 模型中所有层的参数列表
        """
        parameters: list[np.ndarray] = []
        for layer in self.layers:
            parameters.extend(layer.getParameters())
        return parameters

    def getGradients(self) -> list[np.ndarray]:
        """
        获取模型中所有层的梯度列表
        Returns:
            list[np.ndarray]: 模型中所有层的梯度列表
        """
        gradients: list[np.ndarray] = []
        for layer in self.layers:
            gradients.extend(layer.getGradients())
        return gradients

    def train(self):
        """
        设置模型为训练模式，所有层也设置为训练模式

        """
        super().train()
        for layer in self.layers:
            layer.train()

    def eval(self):
        """
        设置模型为评估模式，所有层也设置为评估模式

        """
        super().eval()
        for layer in self.layers:
            layer.eval()

    def predict(self, inputData: np.ndarray) -> np.ndarray:
        """
        使用模型进行预测，前向传播输入数据

        Args:
            input(np.ndarray): 输入数据，形状为(batch_size, input_dim)

        Returns:
            np.ndarray: 预测结果，形状为(batch_size, output_dim)
        """
        wasTraining = self.isTraining
        self.eval()  # 预测时使用评估模式

        try:
            predictions = self.forward(inputData)
        finally:
            if wasTraining:
                self.train()  # 恢复之前的训练模式
            else:
                self.eval()  # 恢复之前的评估模式

        return predictions

    def __len__(self) -> int:
        """
        返回模型中层的数量
        Returns:
            int: 模型中层的数量
        """
        return len(self.layers)
