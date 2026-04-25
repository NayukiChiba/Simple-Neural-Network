"""
激活层测试模块

功能：
1. 测试 ReLU 激活层
2. 测试 Sigmoid 激活层
3. 测试 Tanh 激活层
4. 测试前向传播缓存与反向传播异常
"""

import numpy as np
import pytest

from src.nn.layers import ReLULayer, SigmoidLayer, TanhLayer


def testReLUForward() -> None:
    """
    测试 ReLU 前向传播
    """
    layer = ReLULayer()

    inputData = np.array(
        [
            [-2.0, -1.0, 0.0],
            [1.0, 2.0, 3.0],
        ],
        dtype=np.float64,
    )

    outputData = layer.forward(inputData)

    expectedOutput = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 2.0, 3.0],
        ],
        dtype=np.float64,
    )

    np.testing.assert_allclose(outputData, expectedOutput)
    np.testing.assert_array_equal(layer.inputCache, inputData)
    np.testing.assert_array_equal(layer.outputCache, expectedOutput)


def testReLUBackward() -> None:
    """
    测试 ReLU 反向传播
    """
    layer = ReLULayer()

    inputData = np.array(
        [
            [-2.0, 0.0, 3.0],
            [4.0, -5.0, 6.0],
        ],
        dtype=np.float64,
    )
    layer.forward(inputData)

    outputGradient = np.array(
        [
            [1.0, 1.0, 1.0],
            [2.0, 2.0, 2.0],
        ],
        dtype=np.float64,
    )

    inputGradient = layer.backward(outputGradient)

    expectedGradient = np.array(
        [
            [0.0, 0.0, 1.0],
            [2.0, 0.0, 2.0],
        ],
        dtype=np.float64,
    )

    np.testing.assert_allclose(inputGradient, expectedGradient)


def testReLUBackwardRaisesWhenForwardNotCalled() -> None:
    """
    测试 ReLU 未先前向传播时调用反向传播会抛出异常
    """
    layer = ReLULayer()
    outputGradient = np.ones((2, 2), dtype=np.float64)

    with pytest.raises(ValueError):
        layer.backward(outputGradient)


def testSigmoidForward() -> None:
    """
    测试 Sigmoid 前向传播
    """
    layer = SigmoidLayer()

    inputData = np.array(
        [
            [0.0, 1.0],
            [-1.0, 2.0],
        ],
        dtype=np.float64,
    )

    outputData = layer.forward(inputData)

    expectedOutput = 1.0 / (1.0 + np.exp(-inputData))

    np.testing.assert_allclose(outputData, expectedOutput)
    np.testing.assert_array_equal(layer.inputCache, inputData)
    np.testing.assert_allclose(layer.outputCache, expectedOutput)


def testSigmoidBackward() -> None:
    """
    测试 Sigmoid 反向传播
    """
    layer = SigmoidLayer()

    inputData = np.array(
        [
            [0.0, 1.0],
            [-1.0, 2.0],
        ],
        dtype=np.float64,
    )
    sigmoidOutput = layer.forward(inputData)

    outputGradient = np.array(
        [
            [1.0, 2.0],
            [3.0, 4.0],
        ],
        dtype=np.float64,
    )

    inputGradient = layer.backward(outputGradient)

    expectedGradient = outputGradient * sigmoidOutput * (1.0 - sigmoidOutput)

    np.testing.assert_allclose(inputGradient, expectedGradient)


def testSigmoidBackwardRaisesWhenForwardNotCalled() -> None:
    """
    测试 Sigmoid 未先前向传播时调用反向传播会抛出异常
    """
    layer = SigmoidLayer()
    outputGradient = np.ones((2, 2), dtype=np.float64)

    with pytest.raises(ValueError):
        layer.backward(outputGradient)


def testTanhForward() -> None:
    """
    测试 Tanh 前向传播
    """
    layer = TanhLayer()

    inputData = np.array(
        [
            [0.0, 1.0],
            [-1.0, 2.0],
        ],
        dtype=np.float64,
    )

    outputData = layer.forward(inputData)

    expectedOutput = np.tanh(inputData)

    np.testing.assert_allclose(outputData, expectedOutput)
    np.testing.assert_array_equal(layer.inputCache, inputData)
    np.testing.assert_allclose(layer.outputCache, expectedOutput)


def testTanhBackward() -> None:
    """
    测试 Tanh 反向传播
    """
    layer = TanhLayer()

    inputData = np.array(
        [
            [0.0, 1.0],
            [-1.0, 2.0],
        ],
        dtype=np.float64,
    )
    tanhOutput = layer.forward(inputData)

    outputGradient = np.array(
        [
            [1.0, 2.0],
            [3.0, 4.0],
        ],
        dtype=np.float64,
    )

    inputGradient = layer.backward(outputGradient)

    expectedGradient = outputGradient * (1.0 - tanhOutput**2)

    np.testing.assert_allclose(inputGradient, expectedGradient)


def testTanhBackwardRaisesWhenForwardNotCalled() -> None:
    """
    测试 Tanh 未先前向传播时调用反向传播会抛出异常
    """
    layer = TanhLayer()
    outputGradient = np.ones((2, 2), dtype=np.float64)

    with pytest.raises(ValueError):
        layer.backward(outputGradient)


def testActivationLayersHaveNoParameters() -> None:
    """
    测试激活层没有可训练参数
    """
    reluLayer = ReLULayer()
    sigmoidLayer = SigmoidLayer()
    tanhLayer = TanhLayer()

    assert reluLayer.getParameters() == []
    assert reluLayer.getGradients() == []
    assert reluLayer.hasParameters() is False

    assert sigmoidLayer.getParameters() == []
    assert sigmoidLayer.getGradients() == []
    assert sigmoidLayer.hasParameters() is False

    assert tanhLayer.getParameters() == []
    assert tanhLayer.getGradients() == []
    assert tanhLayer.hasParameters() is False


def testActivationLayersZeroGradDoesNothing() -> None:
    """
    测试无参数激活层调用 zeroGrad 不会报错
    """
    reluLayer = ReLULayer()
    sigmoidLayer = SigmoidLayer()
    tanhLayer = TanhLayer()

    reluLayer.zeroGrad()
    sigmoidLayer.zeroGrad()
    tanhLayer.zeroGrad()
