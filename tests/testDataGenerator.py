"""
数据生成器测试模块

功能：
1. 测试 xor 数据集是否能正确生成
2. 测试 spiral 数据集是否能正确生成
3. 测试 sine 数据集是否能正确生成
4. 测试数据划分比例校验是否生效
"""

from pathlib import Path

import numpy as np
import pytest

import config
from src.nn.data import DataGenerator


def patchDatasetPaths(monkeypatch: pytest.MonkeyPatch, tmpPath: Path) -> None:
    """
    将测试中的数据集输出路径重定向到临时目录

    Args:
        monkeypatch (pytest.MonkeyPatch): pytest 提供的补丁工具
        tmpPath (Path): 临时目录路径
    """
    datasetsDir = tmpPath / "datasets"

    xorDir = datasetsDir / "xor"
    spiralDir = datasetsDir / "spiral"
    sineDir = datasetsDir / "sine"
    # 重定向路径
    monkeypatch.setattr(config, "DATASETS_DIR", datasetsDir)
    # 重定向子目录路径
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


def testGenerateXorDataset(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """
    测试 XOR 数据集生成
    """
    # 重定向数据集路径到临时目录
    patchDatasetPaths(monkeypatch, tmp_path)

    generator = DataGenerator(seed=42)
    generator.generateXorDataset()

    # 验证文件是否存在
    assert config.XOR_FILE.exists()

    # 验证数据内容
    with np.load(config.XOR_FILE, allow_pickle=False) as datasetFile:
        x = datasetFile["x"]
        y = datasetFile["y"]
    # 验证数据形状、类型和内容
    assert x.shape == (4, 2)
    assert y.shape == (4,)
    # 验证标签是整数类型
    assert np.issubdtype(y.dtype, np.integer)

    expectedX = np.array(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0],
        ],
        dtype=np.float64,
    )
    expectedY = np.array([0, 1, 1, 0], dtype=np.int64)

    np.testing.assert_array_equal(x, expectedX)
    np.testing.assert_array_equal(y, expectedY)


def testGenerateSpiralDataset(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """
    测试 spiral 数据集生成
    """
    patchDatasetPaths(monkeypatch, tmp_path)

    generator = DataGenerator(seed=42)
    generator.generateSpiralDataset()

    assert config.SPIRAL_TRAIN_FILE.exists()
    assert config.SPIRAL_VALID_FILE.exists()
    assert config.SPIRAL_TEST_FILE.exists()

    with np.load(config.SPIRAL_TRAIN_FILE, allow_pickle=False) as trainFile:
        xTrain = trainFile["x"]
        yTrain = trainFile["y"]

    with np.load(config.SPIRAL_VALID_FILE, allow_pickle=False) as validFile:
        xValid = validFile["x"]
        yValid = validFile["y"]

    with np.load(config.SPIRAL_TEST_FILE, allow_pickle=False) as testFile:
        xTest = testFile["x"]
        yTest = testFile["y"]

    totalSampleCount = xTrain.shape[0] + xValid.shape[0] + xTest.shape[0]

    expectedTotalSampleCount = (
        generator.SPIRAL_CLASS_COUNT * generator.SPIRAL_SAMPLES_PER_CLASS
    )

    assert xTrain.shape[1] == 2
    assert xValid.shape[1] == 2
    assert xTest.shape[1] == 2

    assert totalSampleCount == expectedTotalSampleCount

    assert xTrain.shape[0] == yTrain.shape[0]
    assert xValid.shape[0] == yValid.shape[0]
    assert xTest.shape[0] == yTest.shape[0]

    assert np.issubdtype(yTrain.dtype, np.integer)
    assert np.issubdtype(yValid.dtype, np.integer)
    assert np.issubdtype(yTest.dtype, np.integer)


def testGenerateSineDataset(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """
    测试 sine 数据集生成
    """
    patchDatasetPaths(monkeypatch, tmp_path)

    generator = DataGenerator(seed=42)
    generator.generateSineDataset()

    assert config.SINE_TRAIN_FILE.exists()
    assert config.SINE_VALID_FILE.exists()
    assert config.SINE_TEST_FILE.exists()

    with np.load(config.SINE_TRAIN_FILE, allow_pickle=False) as trainFile:
        xTrain = trainFile["x"]
        yTrain = trainFile["y"]

    with np.load(config.SINE_VALID_FILE, allow_pickle=False) as validFile:
        xValid = validFile["x"]
        yValid = validFile["y"]

    with np.load(config.SINE_TEST_FILE, allow_pickle=False) as testFile:
        xTest = testFile["x"]
        yTest = testFile["y"]

    totalSampleCount = xTrain.shape[0] + xValid.shape[0] + xTest.shape[0]

    assert totalSampleCount == generator.SINE_SAMPLE_COUNT

    assert xTrain.shape[1] == 1
    assert xValid.shape[1] == 1
    assert xTest.shape[1] == 1

    assert yTrain.shape[1] == 1
    assert yValid.shape[1] == 1
    assert yTest.shape[1] == 1

    assert xTrain.shape[0] == yTrain.shape[0]
    assert xValid.shape[0] == yValid.shape[0]
    assert xTest.shape[0] == yTest.shape[0]


def testSplitDatasetRaisesWhenRatiosAreInvalid() -> None:
    """
    测试 splitDataset 会主动校验无效的划分比例
    """
    generator = DataGenerator(seed=42)
    generator.train_ratio = 0.8
    generator.valid_ratio = 0.3
    generator.test_ratio = 0.1

    x = np.arange(20, dtype=np.float64).reshape(10, 2)
    y = np.arange(10, dtype=np.int64)

    with pytest.raises(ValueError):
        generator.splitDataset(x, y)


def testGenerateAllDatasets(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """
    测试一次性生成全部数据集
    """
    patchDatasetPaths(monkeypatch, tmp_path)

    generator = DataGenerator(seed=42)
    generator.generateAllDatasets()

    assert config.XOR_FILE.exists()
    assert config.SPIRAL_TRAIN_FILE.exists()
    assert config.SPIRAL_VALID_FILE.exists()
    assert config.SPIRAL_TEST_FILE.exists()
    assert config.SINE_TRAIN_FILE.exists()
    assert config.SINE_VALID_FILE.exists()
    assert config.SINE_TEST_FILE.exists()
