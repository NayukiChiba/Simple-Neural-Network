"""
顺序模型测试模块

功能：
1. 测试顺序模型的初始化与加层
2. 测试前向传播与反向传播
3. 测试参数、梯度和模式切换
4. 测试预测接口与异常处理
"""

import numpy as np
import pytest

from src.nn.layers import BaseLayer, LinearLayer, ReLULayer
from src.nn.models.sequentialModel import SequentialModel


class ModeCheckLayer(BaseLayer):
    """
    用于测试 predict 时模型是否切换到 eval 模式的辅助层
    """

    def __init__(self) -> None:
        super().__init__()
        self.modeSeenInForward: bool | None = None

    def forward(self, inputData: np.ndarray) -> np.ndarray:
        self.modeSeenInForward = self.isTraining
        self.inputCache = inputData
        self.outputCache = inputData
        return inputData

    def backward(self, outputGradient: np.ndarray) -> np.ndarray:
        if self.inputCache is None:
            raise ValueError("调用 backward 之前必须先调用 forward")
        return outputGradient


def createTestModel() -> tuple[SequentialModel, LinearLayer, ReLULayer, LinearLayer]:
    """
    创建一个固定参数的测试模型
    """
    layer1 = LinearLayer(inputDim=2, outputDim=3, useBias=True, randomSeed=42)
    activation = ReLULayer()
    layer2 = LinearLayer(inputDim=3, outputDim=1, useBias=True, randomSeed=42)

    layer1.weights[...] = np.array(
        [
            [1.0, -1.0, 2.0],
            [0.5, 2.0, -0.5],
        ],
        dtype=np.float64,
    )
    layer2.weights[...] = np.array(
        [
            [1.0],
            [2.0],
            [3.0],
        ],
        dtype=np.float64,
    )

    assert layer1.bias is not None
    assert layer2.bias is not None

    layer1.bias[...] = np.array([[0.1, -0.2, 0.3]], dtype=np.float64)
    layer2.bias[...] = np.array([[0.5]], dtype=np.float64)

    model = SequentialModel([layer1, activation, layer2])
    return model, layer1, activation, layer2


def testInitWithLayers() -> None:
    """
    测试使用初始层列表构造模型
    """
    layer1 = LinearLayer(inputDim=2, outputDim=3)
    layer2 = ReLULayer()

    model = SequentialModel([layer1, layer2])

    assert len(model) == 2
    assert model.layers == [layer1, layer2]


def testForwardReturnsExpectedOutputAndCachesInput() -> None:
    """
    测试顺序模型前向传播结果和缓存
    """
    model, _, _, _ = createTestModel()

    inputData = np.array(
        [
            [1.0, 2.0],
            [3.0, 4.0],
        ],
        dtype=np.float64,
    )

    outputData = model.forward(inputData)

    expectedOutput = np.array(
        [
            [12.1],
            [28.1],
        ],
        dtype=np.float64,
    )

    np.testing.assert_allclose(outputData, expectedOutput)
    np.testing.assert_allclose(model.inputCache, inputData)
    np.testing.assert_allclose(model.outputCache, expectedOutput)


def testForwardRaisesWhenModelIsEmpty() -> None:
    """
    测试空模型前向传播时抛出异常
    """
    model = SequentialModel()
    inputData = np.ones((2, 2), dtype=np.float64)

    with pytest.raises(ValueError):
        model.forward(inputData)


def testBackwardReturnsExpectedGradientAndLayerGradients() -> None:
    """
    测试顺序模型反向传播结果
    """
    model, layer1, _, layer2 = createTestModel()

    inputData = np.array(
        [
            [1.0, 2.0],
            [3.0, 4.0],
        ],
        dtype=np.float64,
    )
    model.forward(inputData)

    outputGradient = np.array(
        [
            [1.0],
            [2.0],
        ],
        dtype=np.float64,
    )

    inputGradient = model.backward(outputGradient)

    expectedInputGradient = np.array(
        [
            [5.0, 3.0],
            [10.0, 6.0],
        ],
        dtype=np.float64,
    )
    expectedGradWeights1 = np.array(
        [
            [7.0, 14.0, 21.0],
            [10.0, 20.0, 30.0],
        ],
        dtype=np.float64,
    )
    expectedGradBias1 = np.array([[3.0, 6.0, 9.0]], dtype=np.float64)
    expectedGradWeights2 = np.array(
        [
            [12.3],
            [12.4],
            [9.9],
        ],
        dtype=np.float64,
    )
    expectedGradBias2 = np.array([[3.0]], dtype=np.float64)

    np.testing.assert_allclose(inputGradient, expectedInputGradient)
    np.testing.assert_allclose(layer1.gradWeights, expectedGradWeights1)
    np.testing.assert_allclose(layer2.gradWeights, expectedGradWeights2)

    assert layer1.gradBias is not None
    assert layer2.gradBias is not None

    np.testing.assert_allclose(layer1.gradBias, expectedGradBias1)
    np.testing.assert_allclose(layer2.gradBias, expectedGradBias2)


def testBackwardRaisesWhenForwardNotCalled() -> None:
    """
    测试未先调用 forward 时 backward 抛出异常
    """
    model, _, _, _ = createTestModel()
    outputGradient = np.ones((2, 1), dtype=np.float64)

    with pytest.raises(ValueError):
        model.backward(outputGradient)


def testAddRaisesWhenLayerIsInvalid() -> None:
    """
    测试添加非法层对象时抛出异常
    """
    model = SequentialModel()

    with pytest.raises(TypeError):
        model.add_layer(object())  # type: ignore[arg-type]


def testAddRaisesWhenAddingSelf() -> None:
    """
    测试不能把模型自身添加为子层
    """
    model = SequentialModel()

    with pytest.raises(ValueError):
        model.add_layer(model)


def testAddKeepsNewLayerInEvalMode() -> None:
    """
    测试模型处于 eval 模式时新加的层也会同步为 eval
    """
    model = SequentialModel()
    model.eval()

    layer = ReLULayer()
    model.add_layer(layer)

    assert model.isTraining is False
    assert layer.isTraining is False


def testGetParametersAndGradientsReturnFlattenedLists() -> None:
    """
    测试参数和梯度会按层顺序展开返回
    """
    model, layer1, _, layer2 = createTestModel()

    parameters = model.getParameters()
    gradients = model.getGradients()

    assert len(parameters) == 4
    assert len(gradients) == 4

    assert parameters[0] is layer1.weights
    assert parameters[1] is layer1.bias
    assert parameters[2] is layer2.weights
    assert parameters[3] is layer2.bias

    assert gradients[0] is layer1.gradWeights
    assert gradients[1] is layer1.gradBias
    assert gradients[2] is layer2.gradWeights
    assert gradients[3] is layer2.gradBias


def testZeroGradClearsAllGradients() -> None:
    """
    测试模型 zeroGrad 会清空所有层的梯度
    """
    model, layer1, _, layer2 = createTestModel()

    inputData = np.array(
        [
            [1.0, 2.0],
            [3.0, 4.0],
        ],
        dtype=np.float64,
    )
    outputGradient = np.array(
        [
            [1.0],
            [2.0],
        ],
        dtype=np.float64,
    )

    model.forward(inputData)
    model.backward(outputGradient)
    model.zeroGrad()

    np.testing.assert_allclose(layer1.gradWeights, np.zeros_like(layer1.gradWeights))
    np.testing.assert_allclose(layer2.gradWeights, np.zeros_like(layer2.gradWeights))

    assert layer1.gradBias is not None
    assert layer2.gradBias is not None

    np.testing.assert_allclose(layer1.gradBias, np.zeros_like(layer1.gradBias))
    np.testing.assert_allclose(layer2.gradBias, np.zeros_like(layer2.gradBias))


def testTrainAndEvalPropagateToAllLayers() -> None:
    """
    测试 train 和 eval 会传播给所有子层
    """
    model, layer1, activation, layer2 = createTestModel()

    model.eval()
    assert model.isTraining is False
    assert layer1.isTraining is False
    assert activation.isTraining is False
    assert layer2.isTraining is False

    model.train()
    assert model.isTraining is True
    assert layer1.isTraining is True
    assert activation.isTraining is True
    assert layer2.isTraining is True


def testPredictRunsInEvalModeAndRestoresTrainingMode() -> None:
    """
    测试 predict 会临时切换到 eval 模式并恢复原状态
    """
    spyLayer = ModeCheckLayer()
    model = SequentialModel([spyLayer])
    model.train()

    inputData = np.array([[1.0, 2.0]], dtype=np.float64)
    predictions = model.predict(inputData)

    np.testing.assert_array_equal(predictions, inputData)
    assert spyLayer.modeSeenInForward is False
    assert model.isTraining is True
    assert spyLayer.isTraining is True
