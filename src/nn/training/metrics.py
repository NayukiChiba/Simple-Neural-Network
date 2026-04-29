"""
评估指标模块

1. 提供分类任务准确率计算
2. 提供回归任务均方误差计算
3. 统一管理训练与评估阶段的指标逻辑
"""

import numpy as np


def convertLabelsToIndices(targetData: np.ndarray) -> np.ndarray:
    """
    将标签转换为类别索引

    Args:
        targetData (np.ndarray): 标签数据

    Returns:
        np.ndarray: 一维类别索引数组
    """
    # 判断targetData的维度和形状，支持一维类别索引或二维 one-hot 编码
    if targetData.ndim == 1:
        return targetData

    if targetData.ndim == 2:
        if targetData.shape[1] == 0:
            raise ValueError("targetData 的类别维度不能为空")
        return np.argmax(targetData, axis=1)

    raise ValueError("分类标签必须是一维索引或二维 one-hot 数组")


def calculateAccuracy(predictions: np.ndarray, targetData: np.ndarray) -> float:
    """
    计算分类准确率

    Args:
        predictions (np.ndarray): 模型输出，形状为 (batchSize, classCount)
        targetData (np.ndarray): 标签数据，支持一维类别索引或二维 one-hot

    Returns:
        float: 分类准确率
    """
    # 判断predictions的维度和形状，必须是二维数组，且样本数量必须大于0，类别数量必须大于0
    if predictions.ndim != 2:
        raise ValueError("predictions 必须是二维数组")
    if predictions.shape[0] == 0:
        raise ValueError("predictions 不能为空")
    if predictions.shape[1] == 0:
        raise ValueError("predictions 的类别维度不能为空")

    targetLabels = convertLabelsToIndices(targetData)

    if targetLabels.ndim != 1:
        raise ValueError("targetLabels 必须是一维数组")
    if targetLabels.shape[0] != predictions.shape[0]:
        raise ValueError(
            f"预测样本数量和标签数量不一致, "
            f"predictions.shape[0]={predictions.shape[0]}, "
            f"targetLabels.shape[0]={targetLabels.shape[0]}"
        )

    predictedLabels = np.argmax(predictions, axis=1)
    accuracy = np.mean(predictedLabels == targetLabels)
    return float(accuracy)


def calculateMeanSquaredError(predictions: np.ndarray, targetData: np.ndarray) -> float:
    """
    计算均方误差

    Args:
        predictions (np.ndarray): 模型预测值
        targetData (np.ndarray): 真实目标值

    Returns:
        float: 均方误差
    """
    if predictions.ndim == 0:
        raise ValueError("predictions 不能是标量")
    if targetData.ndim == 0:
        raise ValueError("targetData 不能是标量")
    if predictions.shape != targetData.shape:
        raise ValueError(
            f"predictions 和 targetData 的形状必须一致, "
            f"predictions.shape={predictions.shape}, "
            f"targetData.shape={targetData.shape}"
        )
    if predictions.size == 0:
        raise ValueError("predictions 不能为空")

    mse = np.mean((predictions - targetData) ** 2)
    return float(mse)
