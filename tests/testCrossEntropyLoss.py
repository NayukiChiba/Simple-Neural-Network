"""
交叉熵损失函数测试模块

功能：
1. 测试前向传播损失值
2. 测试反向传播梯度
3. 测试输入校验逻辑
"""

import numpy as np
import pytest

from src.nn.losses import CrossEntropyLoss


def testInitRaisesWhenEpsilonIsInvalid() -> None:
    """
    测试 epsilon 非法时抛出异常
    """
    with pytest.raises(ValueError):
        CrossEntropyLoss(epsilon=0.0)

    with pytest.raises(ValueError):
        CrossEntropyLoss(epsilon=-1e-6)


def testForwardReturnsExpectedLoss() -> None:
    """
    测试前向传播返回正确损失
    """
    lossFunction = CrossEntropyLoss()

    logits = np.array(
        [
            [2.0, 1.0, 0.1],
            [0.5, 1.5, 0.3],
        ],
        dtype=np.float64,
    )
    targetLabels = np.array([0, 1], dtype=np.int64)

    loss = lossFunction.forward(logits, targetLabels)

    shiftedLogits = logits - np.max(logits, axis=1, keepdims=True)
    expLogits = np.exp(shiftedLogits)
    probabilities = expLogits / np.sum(expLogits, axis=1, keepdims=True)
    selectedProbabilities = probabilities[np.arange(2), targetLabels]
    expectedLoss = -np.mean(np.log(selectedProbabilities))

    np.testing.assert_allclose(loss, expectedLoss)


def testBackwardReturnsExpectedGradient() -> None:
    """
    测试反向传播返回正确梯度
    """
    lossFunction = CrossEntropyLoss()

    logits = np.array(
        [
            [2.0, 1.0, 0.1],
            [0.5, 1.5, 0.3],
        ],
        dtype=np.float64,
    )
    targetLabels = np.array([0, 1], dtype=np.int64)

    lossFunction.forward(logits, targetLabels)
    inputGradient = lossFunction.backward()

    shiftedLogits = logits - np.max(logits, axis=1, keepdims=True)
    expLogits = np.exp(shiftedLogits)
    probabilities = expLogits / np.sum(expLogits, axis=1, keepdims=True)

    expectedGradient = probabilities.copy()
    expectedGradient[np.arange(2), targetLabels] -= 1.0
    expectedGradient /= 2

    np.testing.assert_allclose(inputGradient, expectedGradient)


def testForwardRaisesWhenLogitsIsNot2D() -> None:
    """
    测试 logits 不是二维数组时抛出异常
    """
    lossFunction = CrossEntropyLoss()

    logits = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    targetLabels = np.array([0], dtype=np.int64)

    with pytest.raises(ValueError):
        lossFunction.forward(logits, targetLabels)


def testForwardRaisesWhenTargetLabelsIsNot1D() -> None:
    """
    测试 targetLabels 不是一维数组时抛出异常
    """
    lossFunction = CrossEntropyLoss()

    logits = np.array([[1.0, 2.0, 3.0]], dtype=np.float64)
    targetLabels = np.array([[0]], dtype=np.int64)

    with pytest.raises(ValueError):
        lossFunction.forward(logits, targetLabels)


def testForwardRaisesWhenBatchSizeIsZero() -> None:
    """
    测试空批次输入时抛出异常
    """
    lossFunction = CrossEntropyLoss()

    logits = np.empty((0, 3), dtype=np.float64)
    targetLabels = np.empty((0,), dtype=np.int64)

    with pytest.raises(ValueError):
        lossFunction.forward(logits, targetLabels)


def testForwardRaisesWhenTargetCountMismatch() -> None:
    """
    测试标签数量与 batchSize 不一致时抛出异常
    """
    lossFunction = CrossEntropyLoss()

    logits = np.array(
        [
            [1.0, 2.0, 3.0],
            [0.5, 0.2, 1.2],
        ],
        dtype=np.float64,
    )
    targetLabels = np.array([0], dtype=np.int64)

    with pytest.raises(ValueError):
        lossFunction.forward(logits, targetLabels)


def testForwardRaisesWhenTargetLabelsAreNotIntegers() -> None:
    """
    测试标签不是整数类型时抛出异常
    """
    lossFunction = CrossEntropyLoss()

    logits = np.array([[1.0, 2.0, 3.0]], dtype=np.float64)
    targetLabels = np.array([0.0], dtype=np.float64)

    with pytest.raises(ValueError):
        lossFunction.forward(logits, targetLabels)


def testForwardRaisesWhenTargetLabelsOutOfRange() -> None:
    """
    测试标签索引越界时抛出异常
    """
    lossFunction = CrossEntropyLoss()

    logits = np.array([[1.0, 2.0, 3.0]], dtype=np.float64)
    targetLabels = np.array([3], dtype=np.int64)

    with pytest.raises(ValueError):
        lossFunction.forward(logits, targetLabels)


def testBackwardRaisesWhenForwardNotCalled() -> None:
    """
    测试未先调用 forward 时调用 backward 会抛出异常
    """
    lossFunction = CrossEntropyLoss()

    with pytest.raises(ValueError):
        lossFunction.backward()
