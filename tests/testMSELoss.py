"""
均方误差损失函数测试模块

功能：
1. 测试前向传播损失值
2. 测试反向传播梯度
3. 测试输入校验逻辑
"""

import numpy as np
import pytest

from src.nn.losses import MSELoss


def testForwardReturnsExpectedLoss() -> None:
    """
    测试前向传播返回正确损失值
    """
    lossFunction = MSELoss()

    predictions = np.array(
        [
            [1.0],
            [2.0],
            [3.0],
        ],
        dtype=np.float64,
    )
    targets = np.array(
        [
            [1.0],
            [1.0],
            [5.0],
        ],
        dtype=np.float64,
    )

    loss = lossFunction.forward(predictions, targets)

    expectedLoss = np.mean((predictions - targets) ** 2)

    np.testing.assert_allclose(loss, expectedLoss)


def testBackwardReturnsExpectedGradient() -> None:
    """
    测试反向传播返回正确梯度
    """
    lossFunction = MSELoss()

    predictions = np.array(
        [
            [1.0],
            [2.0],
            [3.0],
        ],
        dtype=np.float64,
    )
    targets = np.array(
        [
            [1.0],
            [1.0],
            [5.0],
        ],
        dtype=np.float64,
    )

    lossFunction.forward(predictions, targets)
    inputGradient = lossFunction.backward()

    expectedGradient = 2.0 * (predictions - targets) / predictions.size

    np.testing.assert_allclose(inputGradient, expectedGradient)


def testForwardRaisesWhenPredictionsIsScalar() -> None:
    """
    测试 predictions 是标量时抛出异常
    """
    lossFunction = MSELoss()

    predictions = np.array(1.0)
    targets = np.array([1.0], dtype=np.float64)

    with pytest.raises(ValueError):
        lossFunction.forward(predictions, targets)


def testForwardRaisesWhenTargetsIsScalar() -> None:
    """
    测试 targets 是标量时抛出异常
    """
    lossFunction = MSELoss()

    predictions = np.array([1.0], dtype=np.float64)
    targets = np.array(1.0)

    with pytest.raises(ValueError):
        lossFunction.forward(predictions, targets)


def testForwardRaisesWhenShapesDoNotMatch() -> None:
    """
    测试 predictions 和 targets 形状不一致时抛出异常
    """
    lossFunction = MSELoss()

    predictions = np.array(
        [
            [1.0],
            [2.0],
        ],
        dtype=np.float64,
    )
    targets = np.array([1.0, 2.0], dtype=np.float64)

    with pytest.raises(ValueError):
        lossFunction.forward(predictions, targets)


def testForwardRaisesWhenPredictionsIsEmpty() -> None:
    """
    测试 predictions 为空数组时抛出异常
    """
    lossFunction = MSELoss()

    predictions = np.empty((0, 1), dtype=np.float64)
    targets = np.empty((0, 1), dtype=np.float64)

    with pytest.raises(ValueError):
        lossFunction.forward(predictions, targets)


def testBackwardRaisesWhenForwardNotCalled() -> None:
    """
    测试未先调用 forward 时调用 backward 会抛出异常
    """
    lossFunction = MSELoss()

    with pytest.raises(ValueError):
        lossFunction.backward()


def testForwardCachesPredictionsTargetsAndElementCount() -> None:
    """
    测试 forward 会正确缓存中间状态
    """
    lossFunction = MSELoss()

    predictions = np.array(
        [
            [1.0, 2.0],
            [3.0, 4.0],
        ],
        dtype=np.float64,
    )
    targets = np.array(
        [
            [0.0, 1.0],
            [2.0, 3.0],
        ],
        dtype=np.float64,
    )

    lossFunction.forward(predictions, targets)

    np.testing.assert_array_equal(lossFunction.predictions, predictions)
    np.testing.assert_array_equal(lossFunction.targets, targets)
    assert lossFunction.elementCount == predictions.size
