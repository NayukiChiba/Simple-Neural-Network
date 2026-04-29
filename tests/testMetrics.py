"""
评估指标测试模块

功能：
1. 测试分类准确率计算
2. 测试 one-hot 标签兼容
3. 测试回归均方误差计算
4. 测试输入校验逻辑
"""

import numpy as np
import pytest

from src.nn.training.metrics import (
    calculateAccuracy,
    calculateMeanSquaredError,
    convertLabelsToIndices,
)


def testConvertLabelsToIndicesWithIndexLabels() -> None:
    """
    测试一维索引标签保持不变
    """
    targetData = np.array([0, 2, 1], dtype=np.int64)

    convertedLabels = convertLabelsToIndices(targetData)

    np.testing.assert_array_equal(convertedLabels, targetData)


def testConvertLabelsToIndicesWithOneHotLabels() -> None:
    """
    测试二维 one-hot 标签转换为类别索引
    """
    targetData = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float64,
    )

    convertedLabels = convertLabelsToIndices(targetData)

    expectedLabels = np.array([0, 2, 1], dtype=np.int64)
    np.testing.assert_array_equal(convertedLabels, expectedLabels)


def testCalculateAccuracyWithIndexLabels() -> None:
    """
    测试使用索引标签计算准确率
    """
    predictions = np.array(
        [
            [2.0, 1.0, 0.0],
            [0.1, 0.2, 0.9],
            [0.3, 1.5, 0.2],
        ],
        dtype=np.float64,
    )
    targetData = np.array([0, 2, 1], dtype=np.int64)

    accuracy = calculateAccuracy(predictions, targetData)

    assert accuracy == 1.0


def testCalculateAccuracyWithOneHotLabels() -> None:
    """
    测试使用 one-hot 标签计算准确率
    """
    predictions = np.array(
        [
            [2.0, 1.0, 0.0],
            [0.1, 0.2, 0.9],
            [0.3, 1.5, 0.2],
        ],
        dtype=np.float64,
    )
    targetData = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float64,
    )

    accuracy = calculateAccuracy(predictions, targetData)

    assert accuracy == 1.0


def testCalculateAccuracyRaisesWhenPredictionShapeIsInvalid() -> None:
    """
    测试 predictions 不是二维数组时抛出异常
    """
    predictions = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    targetData = np.array([0], dtype=np.int64)

    with pytest.raises(ValueError):
        calculateAccuracy(predictions, targetData)


def testCalculateAccuracyRaisesWhenSampleCountMismatch() -> None:
    """
    测试预测样本数和标签数不匹配时抛出异常
    """
    predictions = np.array(
        [
            [1.0, 2.0],
            [3.0, 4.0],
        ],
        dtype=np.float64,
    )
    targetData = np.array([0], dtype=np.int64)

    with pytest.raises(ValueError):
        calculateAccuracy(predictions, targetData)


def testCalculateMeanSquaredErrorReturnsExpectedValue() -> None:
    """
    测试均方误差计算结果
    """
    predictions = np.array(
        [
            [1.0],
            [2.0],
            [3.0],
        ],
        dtype=np.float64,
    )
    targetData = np.array(
        [
            [1.0],
            [1.0],
            [5.0],
        ],
        dtype=np.float64,
    )

    mse = calculateMeanSquaredError(predictions, targetData)
    expectedMse = np.mean((predictions - targetData) ** 2)

    np.testing.assert_allclose(mse, expectedMse)


def testCalculateMeanSquaredErrorRaisesWhenShapeMismatch() -> None:
    """
    测试形状不一致时抛出异常
    """
    predictions = np.array([[1.0], [2.0]], dtype=np.float64)
    targetData = np.array([1.0, 2.0], dtype=np.float64)

    with pytest.raises(ValueError):
        calculateMeanSquaredError(predictions, targetData)
