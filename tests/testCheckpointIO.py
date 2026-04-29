"""
模型检查点读写测试模块

功能：
1. 测试检查点文件保存
2. 测试模型参数加载恢复
3. 测试异常场景处理
"""

from pathlib import Path

import numpy as np
import pytest

from src.nn.layers import LinearLayer, ReLULayer
from src.nn.models.sequentialModel import SequentialModel
from src.nn.persistence import CheckpointIO


def createTestModel() -> SequentialModel:
    """
    创建带固定参数的测试模型

    Returns:
        SequentialModel: 测试用顺序模型
    """
    firstLayer = LinearLayer(inputDim=2, outputDim=3, useBias=True, randomSeed=42)
    activationLayer = ReLULayer()
    secondLayer = LinearLayer(inputDim=3, outputDim=1, useBias=True, randomSeed=42)

    firstLayer.weights[...] = np.array(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
        ],
        dtype=np.float64,
    )
    secondLayer.weights[...] = np.array(
        [
            [0.5],
            [1.5],
            [2.5],
        ],
        dtype=np.float64,
    )

    assert firstLayer.bias is not None
    assert secondLayer.bias is not None

    firstLayer.bias[...] = np.array([[0.1, 0.2, 0.3]], dtype=np.float64)
    secondLayer.bias[...] = np.array([[0.4]], dtype=np.float64)

    return SequentialModel([firstLayer, activationLayer, secondLayer])


def testSaveCheckpointCreatesNpzFile(tmp_path: Path) -> None:
    """
    测试保存检查点后会生成 .npz 文件
    """
    model = createTestModel()
    checkpointIO = CheckpointIO()
    checkpointPath = tmp_path / "checkpoints" / "model.npz"

    checkpointIO.saveCheckpoint(model, checkpointPath)

    assert checkpointPath.exists()

    with np.load(checkpointPath, allow_pickle=False) as checkpointData:
        assert "parameter_count" in checkpointData.files
        assert "layer_0_param_0" in checkpointData.files
        assert "layer_0_param_1" in checkpointData.files
        assert "layer_2_param_0" in checkpointData.files
        assert "layer_2_param_1" in checkpointData.files


def testSaveCheckpointRaisesWhenSuffixIsInvalid(tmp_path: Path) -> None:
    """
    测试保存到非 .npz 文件时抛出异常
    """
    model = createTestModel()
    checkpointIO = CheckpointIO()
    checkpointPath = tmp_path / "model.txt"

    with pytest.raises(ValueError):
        checkpointIO.saveCheckpoint(model, checkpointPath)


def testLoadCheckpointRestoresModelParameters(tmp_path: Path) -> None:
    """
    测试加载检查点后参数会恢复
    """
    sourceModel = createTestModel()
    targetModel = createTestModel()
    checkpointIO = CheckpointIO()
    checkpointPath = tmp_path / "model.npz"

    checkpointIO.saveCheckpoint(sourceModel, checkpointPath)

    targetFirstLayer = targetModel.layers[0]
    targetSecondLayer = targetModel.layers[2]

    assert isinstance(targetFirstLayer, LinearLayer)
    assert isinstance(targetSecondLayer, LinearLayer)
    assert targetFirstLayer.bias is not None
    assert targetSecondLayer.bias is not None

    targetFirstLayer.weights[...] = -1.0
    targetFirstLayer.bias[...] = -2.0
    targetSecondLayer.weights[...] = -3.0
    targetSecondLayer.bias[...] = -4.0

    checkpointIO.loadCheckpoint(targetModel, checkpointPath)

    sourceFirstLayer = sourceModel.layers[0]
    sourceSecondLayer = sourceModel.layers[2]

    assert isinstance(sourceFirstLayer, LinearLayer)
    assert isinstance(sourceSecondLayer, LinearLayer)
    assert sourceFirstLayer.bias is not None
    assert sourceSecondLayer.bias is not None

    np.testing.assert_allclose(targetFirstLayer.weights, sourceFirstLayer.weights)
    np.testing.assert_allclose(targetFirstLayer.bias, sourceFirstLayer.bias)
    np.testing.assert_allclose(targetSecondLayer.weights, sourceSecondLayer.weights)
    np.testing.assert_allclose(targetSecondLayer.bias, sourceSecondLayer.bias)


def testLoadCheckpointRaisesWhenFileDoesNotExist() -> None:
    """
    测试检查点文件不存在时抛出异常
    """
    model = createTestModel()
    checkpointIO = CheckpointIO()

    with pytest.raises(FileNotFoundError):
        checkpointIO.loadCheckpoint(model, "not_exists.npz")


def testLoadCheckpointRaisesWhenSuffixIsInvalid(tmp_path: Path) -> None:
    """
    测试加载非 .npz 文件时抛出异常
    """
    model = createTestModel()
    checkpointIO = CheckpointIO()
    checkpointPath = tmp_path / "model.txt"
    checkpointPath.write_text("invalid checkpoint", encoding="utf-8")

    with pytest.raises(ValueError):
        checkpointIO.loadCheckpoint(model, checkpointPath)


def testLoadCheckpointRaisesWhenParameterCountMismatch(tmp_path: Path) -> None:
    """
    测试参数数量不匹配时抛出异常
    """
    checkpointIO = CheckpointIO()
    sourceModel = createTestModel()
    checkpointPath = tmp_path / "model.npz"
    checkpointIO.saveCheckpoint(sourceModel, checkpointPath)

    targetModel = SequentialModel(
        [LinearLayer(inputDim=2, outputDim=2, useBias=True, randomSeed=42)]
    )

    with pytest.raises(ValueError):
        checkpointIO.loadCheckpoint(targetModel, checkpointPath)


def testLoadCheckpointRaisesWhenParameterFieldIsMissing(tmp_path: Path) -> None:
    """
    测试检查点缺少参数字段时抛出异常
    """
    model = createTestModel()
    checkpointIO = CheckpointIO()
    checkpointPath = tmp_path / "broken.npz"

    np.savez(
        checkpointPath,
        parameter_count=np.array([4], dtype=np.int64),
        layer_0_param_0=np.zeros((2, 3), dtype=np.float64),
    )

    with pytest.raises(ValueError):
        checkpointIO.loadCheckpoint(model, checkpointPath)


def testLoadCheckpointRaisesWhenParameterShapeMismatch(tmp_path: Path) -> None:
    """
    测试参数形状不匹配时抛出异常
    """
    model = createTestModel()
    checkpointIO = CheckpointIO()
    checkpointPath = tmp_path / "shape_mismatch.npz"

    np.savez(
        checkpointPath,
        parameter_count=np.array([4], dtype=np.int64),
        layer_0_param_0=np.zeros((1, 1), dtype=np.float64),
        layer_0_param_1=np.zeros((1, 3), dtype=np.float64),
        layer_2_param_0=np.zeros((3, 1), dtype=np.float64),
        layer_2_param_1=np.zeros((1, 1), dtype=np.float64),
    )

    with pytest.raises(ValueError):
        checkpointIO.loadCheckpoint(model, checkpointPath)
