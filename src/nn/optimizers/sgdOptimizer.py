"""
随机梯度下降(Stochastic Gradient Descent)优化器

1. 实现最基础的 SGD 参数更新
2. 支持对多层网络逐层更新参数
3. 支持统一清空所有层的梯度


"""

from collections.abc import Iterable

from src.nn.layers.baseLayer import BaseLayer


class SGDOptimizer:
    """
    随机梯度下降(Stochastic Gradient Descent)优化器

    参数更新公式为:
        parameter = parameter - learning_rate * gradient
    """

    def __init__(self, learning_rate: float = 0.01) -> None:
        """
        初始化 SGD 优化器

        Args    :
            learning_rate(float): 学习率
        """
        if learning_rate <= 0.0:
            raise ValueError("学习率必须为正数")
        self.learning_rate = learning_rate

    def step(self, layers: Iterable[BaseLayer]) -> None:
        """
        对所有层进行参数更新

        Args    :
            layers(Iterable[BaseLayer]): 需要更新参数的层列表
        """
        for layer in layers:
            parameters = layer.getParameters()
            gradients = layer.getGradients()

            if len(parameters) != len(gradients):
                raise ValueError("参数和梯度的数量不匹配")

            for parameter, gradient in zip(parameters, gradients):
                if parameter.shape != gradient.shape:
                    raise ValueError("参数和梯度的形状不匹配")
                # 参数更新
                parameter -= self.learning_rate * gradient

    def zeroGrad(self, layers: Iterable[BaseLayer]) -> None:
        """
        清空所有层的梯度
        Args    :
            layers(Iterable[BaseLayer]): 需要清空梯度的层列表


        """

        for layer in layers:
            layer.zeroGrad()
