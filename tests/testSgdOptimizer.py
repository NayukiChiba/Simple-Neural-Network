"""
SGD 优化器测试模块

功能：
1. 测试 SGD 初始化
2. 测试参数更新
3. 测试多层参数更新
4. 测试梯度清零
5. 测试异常场景
"""

import numpy as np
import pytest

from src.nn.layers import LinearLayer
from src.nn.optimizers import SGDOptimizer


def testInitRaisesWhenLearningRateIsInvalid() -> None:
    """
    测试学习率非法时抛出异常
    """
    with pytest.raises(ValueError):
        SGDOptimizer(learning_rate=0.0)

    with pytest.raises(ValueError):
        SGDOptimizer(learning_rate=-0.1)


def testStepUpdatesWeightsAndBias() -> None:
    """
    测试 step 会更新权重和偏置
    """
    layer = LinearLayer(inputDim=2, outputDim=2, useBias=True, randomSeed=42)
    optimizer = SGDOptimizer(learning_rate=0.1)

    layer.weights[...] = np.array(
        [
            [1.0, 2.0],
            [3.0, 4.0],
        ],
        dtype=np.float64,
    )
    layer.bias[...] = np.array([[0.5, -0.5]], dtype=np.float64)

    layer.gradWeights[...] = np.array(
        [
            [0.1, 0.2],
            [0.3, 0.4],
        ],
        dtype=np.float64,
    )
    layer.gradBias[...] = np.array([[0.5, -1.0]], dtype=np.float64)

    optimizer.step([layer])

    expectedWeights = np.array(
        [
            [0.99, 1.98],
            [2.97, 3.96],
        ],
        dtype=np.float64,
    )
    expectedBias = np.array([[0.45, -0.4]], dtype=np.float64)

    np.testing.assert_allclose(layer.weights, expectedWeights)
    np.testing.assert_allclose(layer.bias, expectedBias)


def testStepUpdatesWeightsWithoutBias() -> None:
    """
    测试不带偏置时 step 只更新权重
    """
    layer = LinearLayer(inputDim=2, outputDim=2, useBias=False, randomSeed=42)
    optimizer = SGDOptimizer(learning_rate=0.5)

    layer.weights[...] = np.array(
        [
            [2.0, 4.0],
            [6.0, 8.0],
        ],
        dtype=np.float64,
    )
    layer.gradWeights[...] = np.array(
        [
            [1.0, 2.0],
            [3.0, 4.0],
        ],
        dtype=np.float64,
    )

    optimizer.step([layer])

    expectedWeights = np.array(
        [
            [1.5, 3.0],
            [4.5, 6.0],
        ],
        dtype=np.float64,
    )

    np.testing.assert_allclose(layer.weights, expectedWeights)


def testStepUpdatesMultipleLayers() -> None:
    """
    测试 step 可以同时更新多个层
    """
    firstLayer = LinearLayer(inputDim=2, outputDim=2, useBias=True, randomSeed=42)
    secondLayer = LinearLayer(inputDim=2, outputDim=1, useBias=True, randomSeed=42)
    optimizer = SGDOptimizer(learning_rate=0.1)

    firstLayer.weights[...] = np.array(
        [
            [1.0, 1.0],
            [1.0, 1.0],
        ],
        dtype=np.float64,
    )
    firstLayer.bias[...] = np.array([[1.0, 1.0]], dtype=np.float64)
    firstLayer.gradWeights[...] = np.array(
        [
            [0.1, 0.2],
            [0.3, 0.4],
        ],
        dtype=np.float64,
    )
    firstLayer.gradBias[...] = np.array([[0.5, 0.6]], dtype=np.float64)

    secondLayer.weights[...] = np.array(
        [
            [2.0],
            [3.0],
        ],
        dtype=np.float64,
    )
    secondLayer.bias[...] = np.array([[4.0]], dtype=np.float64)
    secondLayer.gradWeights[...] = np.array(
        [
            [0.2],
            [0.4],
        ],
        dtype=np.float64,
    )
    secondLayer.gradBias[...] = np.array([[0.8]], dtype=np.float64)

    optimizer.step([firstLayer, secondLayer])

    np.testing.assert_allclose(
        firstLayer.weights,
        np.array(
            [
                [0.99, 0.98],
                [0.97, 0.96],
            ],
            dtype=np.float64,
        ),
    )
    np.testing.assert_allclose(
        firstLayer.bias,
        np.array([[0.95, 0.94]], dtype=np.float64),
    )

    np.testing.assert_allclose(
        secondLayer.weights,
        np.array(
            [
                [1.98],
                [2.96],
            ],
            dtype=np.float64,
        ),
    )
    np.testing.assert_allclose(
        secondLayer.bias,
        np.array([[3.92]], dtype=np.float64),
    )


def testZeroGradClearsAllGradients() -> None:
    """
    测试 zeroGrad 会清空所有层的梯度
    """
    firstLayer = LinearLayer(inputDim=2, outputDim=2, useBias=True, randomSeed=42)
    secondLayer = LinearLayer(inputDim=2, outputDim=1, useBias=True, randomSeed=42)
    optimizer = SGDOptimizer(learning_rate=0.1)

    firstLayer.gradWeights[...] = 3.0
    firstLayer.gradBias[...] = 4.0

    secondLayer.gradWeights[...] = 5.0
    secondLayer.gradBias[...] = 6.0

    optimizer.zeroGrad([firstLayer, secondLayer])

    np.testing.assert_array_equal(
        firstLayer.gradWeights,
        np.zeros_like(firstLayer.gradWeights),
    )
    np.testing.assert_array_equal(
        firstLayer.gradBias,
        np.zeros_like(firstLayer.gradBias),
    )
    np.testing.assert_array_equal(
        secondLayer.gradWeights,
        np.zeros_like(secondLayer.gradWeights),
    )
    np.testing.assert_array_equal(
        secondLayer.gradBias,
        np.zeros_like(secondLayer.gradBias),
    )


def testStepRaisesWhenParameterAndGradientCountMismatch() -> None:
    """
    测试参数数量和梯度数量不匹配时抛出异常
    """

    class BrokenLayer(LinearLayer):
        """
        人为制造参数和梯度数量不一致的测试层
        """

        def getGradients(self) -> list[np.ndarray]:
            return []

    layer = BrokenLayer(inputDim=2, outputDim=2, useBias=True, randomSeed=42)
    optimizer = SGDOptimizer(learning_rate=0.1)

    with pytest.raises(ValueError):
        optimizer.step([layer])


def testStepRaisesWhenParameterAndGradientShapeMismatch() -> None:
    """
    测试参数形状和梯度形状不匹配时抛出异常
    """

    class BrokenLayer(LinearLayer):
        """
        人为制造参数和梯度形状不一致的测试层
        """

        def getGradients(self) -> list[np.ndarray]:
            wrongShapeGradient = np.zeros((1, 1), dtype=np.float64)
            if self.useBias and self.gradBias is not None:
                return [wrongShapeGradient, self.gradBias]
            return [wrongShapeGradient]

    layer = BrokenLayer(inputDim=2, outputDim=2, useBias=True, randomSeed=42)
    optimizer = SGDOptimizer(learning_rate=0.1)

    with pytest.raises(ValueError):
        optimizer.step([layer])


def testZeroGradSupportsEmptyLayerList() -> None:
    """
    测试空层列表不会报错
    """
    optimizer = SGDOptimizer(learning_rate=0.1)

    optimizer.zeroGrad([])


def testStepSupportsEmptyLayerList() -> None:
    """
    测试空层列表不会报错
    """
    optimizer = SGDOptimizer(learning_rate=0.1)

    optimizer.step([])
