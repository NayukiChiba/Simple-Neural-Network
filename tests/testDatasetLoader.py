"""
数据集加载器测试模块

功能：
1. 测试 xor 数据集加载
2. 测试 spiral 数据集加载
3. 测试 sine 数据集加载
4. 测试异常场景处理
"""

from pathlib import Path

import numpy as np
import pytest

import config
from src.nn.data import DataGenerator
from src.nn.data.dataLoader import DatasetLoader


def patchDatasetPaths(monkeypatch: pytest.MonkeyPatch, tmpPath: Path) -> None:
    """
    将测试中的数据集路径重定向到临时目录

    Args:
        monkeypatch (pytest.MonkeyPatch): pytest 提供的补丁工具
        tmpPath (Path): 临时目录路径
    """
    datasetsDir = tmpPath / "datasets"

    xorDir = datasetsDir / "xor"
    spiralDir = datasetsDir / "spiral"
    sineDir = datasetsDir / "sine"

    monkeypatch.setattr(config, "DATASETS_DIR", datasetsDir)

    monkeypatch.setattr(config, "XOR_DIR", xorDir)
    monkeypatch.setattr(config, "SPIRAL_DIR", spiralDir)
    monkeypatch.setattr(config, "SINE_DIR", sineDir)

    monkeypatch.setattr(config, "XOR_FILE", xorDir / "xor.npz")

    monkeypatch.setattr(config, "SPIRAL_TRAIN_FILE", spiralDir / "train.npz")
    monkeypatch.setattr(config, "SPIRAL_VALID_FILE", spiralDir / "valid.npz")
    monkeypatch.setattr(config, "SPIRAL_TEST_FILE", spiralDir / "test.npz")

    monkeypatch.setattr(config, "SINE_TRAIN_FILE", sineDir / "train.npz")
    monkeypatch.setattr(config, "SINE_VALID_FILE", sineDir / "valid.npz")
    monkeypatch.setattr(config, "SINE_TEST_FILE", sineDir / "test.npz")


def prepareDatasets(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """
    先生成一组临时数据集，供加载器测试使用

    Args:
        monkeypatch (pytest.MonkeyPatch): pytest 提供的补丁工具
        tmp_path (Path): pytest 临时目录
    """
    patchDatasetPaths(monkeypatch, tmp_path)
    generator = DataGenerator(seed=42)
    generator.generateAllDatasets()


def testLoadXorDataset(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """
    测试 XOR 数据集加载
    """
    prepareDatasets(monkeypatch, tmp_path)

    loader = DatasetLoader()
    x, y = loader.loadXorDataset()

    assert x.shape == (4, 2)
    assert y.shape == (4,)
    assert np.issubdtype(x.dtype, np.floating)
    assert np.issubdtype(y.dtype, np.integer)

    expectedY = np.array([0, 1, 1, 0], dtype=np.int64)
    np.testing.assert_array_equal(y, expectedY)


def testLoadSpiralDataset(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """
    测试 spiral 数据集加载
    """
    prepareDatasets(monkeypatch, tmp_path)

    loader = DatasetLoader()
    trainSet, validSet, testSet = loader.loadSpiralDataset()

    xTrain, yTrain = trainSet
    xValid, yValid = validSet
    xTest, yTest = testSet

    totalSampleCount = xTrain.shape[0] + xValid.shape[0] + xTest.shape[0]

    assert xTrain.shape[1] == 2
    assert xValid.shape[1] == 2
    assert xTest.shape[1] == 2

    assert xTrain.shape[0] == yTrain.shape[0]
    assert xValid.shape[0] == yValid.shape[0]
    assert xTest.shape[0] == yTest.shape[0]

    assert np.issubdtype(yTrain.dtype, np.integer)
    assert np.issubdtype(yValid.dtype, np.integer)
    assert np.issubdtype(yTest.dtype, np.integer)

    assert totalSampleCount == 3000


def testLoadSineDataset(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """
    测试 sine 数据集加载
    """
    prepareDatasets(monkeypatch, tmp_path)

    loader = DatasetLoader()
    trainSet, validSet, testSet = loader.loadSineDataset()

    xTrain, yTrain = trainSet
    xValid, yValid = validSet
    xTest, yTest = testSet

    totalSampleCount = xTrain.shape[0] + xValid.shape[0] + xTest.shape[0]

    assert xTrain.shape[1] == 1
    assert xValid.shape[1] == 1
    assert xTest.shape[1] == 1

    assert yTrain.shape[1] == 1
    assert yValid.shape[1] == 1
    assert yTest.shape[1] == 1

    assert xTrain.shape[0] == yTrain.shape[0]
    assert xValid.shape[0] == yValid.shape[0]
    assert xTest.shape[0] == yTest.shape[0]

    assert totalSampleCount == 600


def testLoadDatasetWithTaskName(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """
    测试统一任务入口加载
    """
    prepareDatasets(monkeypatch, tmp_path)

    loader = DatasetLoader()

    xorSet = loader.loadDataset("xor")
    spiralSet = loader.loadDataset("spiral")
    sineSet = loader.loadDataset("sine")

    assert len(xorSet) == 2
    assert len(spiralSet) == 3
    assert len(sineSet) == 3


def testLoadDatasetRaisesWhenTaskIsInvalid() -> None:
    """
    测试非法任务名称会抛出异常
    """
    loader = DatasetLoader()

    with pytest.raises(ValueError):
        loader.loadDataset("unknown")


def testLoadNpzFileRaisesWhenFileDoesNotExist() -> None:
    """
    测试文件不存在时抛出异常
    """
    loader = DatasetLoader()
    missingFilePath = Path("not_exists.npz")

    with pytest.raises(FileNotFoundError):
        loader.loadNpzFile(missingFilePath, "missing")


def testLoadNpzFileRaisesWhenFieldsAreMissing(tmp_path: Path) -> None:
    """
    测试 npz 文件缺少 x 或 y 字段时抛出异常
    """
    loader = DatasetLoader()
    brokenFilePath = tmp_path / "broken.npz"

    onlyX = np.array([[1.0, 2.0]], dtype=np.float64)
    np.savez(brokenFilePath, x=onlyX)

    with pytest.raises(ValueError):
        loader.loadNpzFile(brokenFilePath, "broken")


def testValidateDatasetRaisesWhenSampleCountMismatch() -> None:
    """
    测试样本数和标签数不一致时抛出异常
    """
    loader = DatasetLoader()

    x = np.zeros((5, 2), dtype=np.float64)
    y = np.zeros((4,), dtype=np.int64)

    with pytest.raises(ValueError):
        loader.validateDataset(x, y, "invalid")


def testValidateDatasetRaisesWhenDatasetIsEmpty() -> None:
    """
    测试空数据集会抛出异常
    """
    loader = DatasetLoader()

    x = np.empty((0, 2), dtype=np.float64)
    y = np.empty((0,), dtype=np.int64)

    with pytest.raises(ValueError):
        loader.validateDataset(x, y, "empty")
