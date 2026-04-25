"""
线性层测试模块

功能：
1. 测试线性层初始化
2. 测试前向传播
3. 测试反向传播
4. 测试参数与梯度接口
"""

import numpy as np
import pytest

from src.nn.layers import LinearLayer


def testInitCreatesParametersWithExpectedShapes() -> None:
    """
    测试初始化后参数形状是否正确
    """
    layer = LinearLayer(inputDim=3, outputDim=2, useBias=True, randomSeed=42)

    assert layer.weights.shape == (3, 2)
    assert layer.gradWeights.shape == (3, 2)

    assert layer.bias is not None
    assert layer.gradBias is not None
    assert layer.bias.shape == (1, 2)
    assert layer.gradBias.shape == (1, 2)

    assert layer.inputDim == 3
    assert layer.outputDim == 2
    assert layer.useBias is True


def testInitWithoutBias() -> None:
    """
    测试不使用偏置时的初始化
    """
    layer = LinearLayer(inputDim=4, outputDim=3, useBias=False, randomSeed=42)

    assert layer.weights.shape == (4, 3)
    assert layer.gradWeights.shape == (4, 3)
    assert layer.bias is None
    assert layer.gradBias is None
    assert layer.useBias is False


def testInitRaisesWhenInputDimIsInvalid() -> None:
    """
    测试输入维度非法时抛出异常
    """
    with pytest.raises(ValueError):
        LinearLayer(inputDim=0, outputDim=2)


def testInitRaisesWhenOutputDimIsInvalid() -> None:
    """
    测试输出维度非法时抛出异常
    """
    with pytest.raises(ValueError):
        LinearLayer(inputDim=2, outputDim=0)


def testForwardWithBias() -> None:
    """
    测试带偏置的前向传播结果
    """
    layer = LinearLayer(inputDim=2, outputDim=3, useBias=True, randomSeed=42)

    layer.weights[...] = np.array(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
        ],
        dtype=np.float64,
    )
    layer.bias[...] = np.array([[0.1, 0.2, 0.3]], dtype=np.float64)

    inputData = np.array(
        [
            [1.0, 2.0],
            [3.0, 4.0],
        ],
        dtype=np.float64,
    )

    outputData = layer.forward(inputData)

    expectedOutput = np.array(
        [
            [9.1, 12.2, 15.3],
            [19.1, 26.2, 33.3],
        ],
        dtype=np.float64,
    )

    np.testing.assert_allclose(outputData, expectedOutput)
    np.testing.assert_array_equal(layer.inputCache, inputData)
    np.testing.assert_array_equal(layer.outputCache, expectedOutput)


def testForwardWithoutBias() -> None:
    """
    测试不带偏置的前向传播结果
    """
    layer = LinearLayer(inputDim=2, outputDim=2, useBias=False, randomSeed=42)

    layer.weights[...] = np.array(
        [
            [1.0, 2.0],
            [3.0, 4.0],
        ],
        dtype=np.float64,
    )

    inputData = np.array(
        [
            [1.0, 1.0],
            [2.0, 0.0],
        ],
        dtype=np.float64,
    )

    outputData = layer.forward(inputData)

    expectedOutput = np.array(
        [
            [4.0, 6.0],
            [2.0, 4.0],
        ],
        dtype=np.float64,
    )

    np.testing.assert_allclose(outputData, expectedOutput)


def testForwardRaisesWhenInputIsNot2D() -> None:
    """
    测试输入不是二维数组时抛出异常
    """
    layer = LinearLayer(inputDim=2, outputDim=2)

    inputData = np.array([1.0, 2.0], dtype=np.float64)

    with pytest.raises(ValueError):
        layer.forward(inputData)


def testForwardRaisesWhenInputDimMismatch() -> None:
    """
    测试输入维度不匹配时抛出异常
    """
    layer = LinearLayer(inputDim=3, outputDim=2)

    inputData = np.ones((4, 2), dtype=np.float64)

    with pytest.raises(ValueError):
        layer.forward(inputData)


def testBackwardComputesExpectedGradientsWithBias() -> None:
    """
    测试反向传播的输入梯度、权重梯度和偏置梯度
    """
    layer = LinearLayer(inputDim=2, outputDim=3, useBias=True, randomSeed=42)

    layer.weights[...] = np.array(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
        ],
        dtype=np.float64,
    )
    layer.bias[...] = np.array([[0.0, 0.0, 0.0]], dtype=np.float64)

    inputData = np.array(
        [
            [1.0, 2.0],
            [3.0, 4.0],
        ],
        dtype=np.float64,
    )
    layer.forward(inputData)

    outputGradient = np.array(
        [
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
        ],
        dtype=np.float64,
    )

    inputGradient = layer.backward(outputGradient)

    expectedInputGradient = np.array(
        [
            [4.0, 10.0],
            [5.0, 11.0],
        ],
        dtype=np.float64,
    )
    expectedGradWeights = np.array(
        [
            [1.0, 3.0, 4.0],
            [2.0, 4.0, 6.0],
        ],
        dtype=np.float64,
    )
    expectedGradBias = np.array([[1.0, 1.0, 2.0]], dtype=np.float64)

    np.testing.assert_allclose(inputGradient, expectedInputGradient)
    np.testing.assert_allclose(layer.gradWeights, expectedGradWeights)
    np.testing.assert_allclose(layer.gradBias, expectedGradBias)


def testBackwardWithoutBias() -> None:
    """
    测试不带偏置时的反向传播
    """
    layer = LinearLayer(inputDim=2, outputDim=2, useBias=False, randomSeed=42)

    layer.weights[...] = np.array(
        [
            [1.0, 2.0],
            [3.0, 4.0],
        ],
        dtype=np.float64,
    )

    inputData = np.array(
        [
            [1.0, 2.0],
            [3.0, 4.0],
        ],
        dtype=np.float64,
    )
    layer.forward(inputData)

    outputGradient = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
        ],
        dtype=np.float64,
    )

    inputGradient = layer.backward(outputGradient)

    expectedInputGradient = np.array(
        [
            [1.0, 3.0],
            [2.0, 4.0],
        ],
        dtype=np.float64,
    )
    expectedGradWeights = np.array(
        [
            [1.0, 3.0],
            [2.0, 4.0],
        ],
        dtype=np.float64,
    )

    np.testing.assert_allclose(inputGradient, expectedInputGradient)
    np.testing.assert_allclose(layer.gradWeights, expectedGradWeights)


def testBackwardRaisesWhenForwardNotCalled() -> None:
    """
    测试未先前向传播时调用反向传播会抛出异常
    """
    layer = LinearLayer(inputDim=2, outputDim=2)
    outputGradient = np.ones((2, 2), dtype=np.float64)

    with pytest.raises(ValueError):
        layer.backward(outputGradient)


def testBackwardRaisesWhenGradientIsNot2D() -> None:
    """
    测试输出梯度不是二维数组时抛出异常
    """
    layer = LinearLayer(inputDim=2, outputDim=2)
    layer.forward(np.ones((2, 2), dtype=np.float64))

    outputGradient = np.ones((2,), dtype=np.float64)

    with pytest.raises(ValueError):
        layer.backward(outputGradient)


def testBackwardRaisesWhenGradientDimMismatch() -> None:
    """
    测试输出梯度维度不匹配时抛出异常
    """
    layer = LinearLayer(inputDim=2, outputDim=3)
    layer.forward(np.ones((2, 2), dtype=np.float64))

    outputGradient = np.ones((2, 2), dtype=np.float64)

    with pytest.raises(ValueError):
        layer.backward(outputGradient)


def testGetParametersAndGetGradientsWithBias() -> None:
    """
    测试带偏置时参数和梯度接口
    """
    layer = LinearLayer(inputDim=2, outputDim=3, useBias=True)

    parameters = layer.getParameters()
    gradients = layer.getGradients()

    assert len(parameters) == 2
    assert len(gradients) == 2

    assert parameters[0] is layer.weights
    assert parameters[1] is layer.bias
    assert gradients[0] is layer.gradWeights
    assert gradients[1] is layer.gradBias


def testGetParametersAndGetGradientsWithoutBias() -> None:
    """
    测试不带偏置时参数和梯度接口
    """
    layer = LinearLayer(inputDim=2, outputDim=3, useBias=False)

    parameters = layer.getParameters()
    gradients = layer.getGradients()

    assert len(parameters) == 1
    assert len(gradients) == 1

    assert parameters[0] is layer.weights
    assert gradients[0] is layer.gradWeights


def testZeroGradClearsGradients() -> None:
    """
    测试 zeroGrad 会清空梯度
    """
    layer = LinearLayer(inputDim=2, outputDim=2, useBias=True)

    layer.gradWeights[...] = 3.0
    layer.gradBias[...] = 5.0

    layer.zeroGrad()

    np.testing.assert_array_equal(layer.gradWeights, np.zeros((2, 2)))
    np.testing.assert_array_equal(layer.gradBias, np.zeros((1, 2)))


def testHasParametersReturnsTrue() -> None:
    """
    测试线性层包含可训练参数
    """
    layer = LinearLayer(inputDim=2, outputDim=2)

    assert layer.hasParameters() is True
