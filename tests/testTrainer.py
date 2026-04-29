"""
训练器测试模块

功能：
1. 测试 batch 划分
2. 测试单步训练
3. 测试评估与预测
4. 测试分类与回归训练流程
"""

import numpy as np
import pytest

from src.nn.layers import LinearLayer
from src.nn.losses import CrossEntropyLoss, MSELoss
from src.nn.models.sequentialModel import SequentialModel
from src.nn.optimizers import SGDOptimizer
from src.nn.training.trainer import Trainer


def createClassificationTrainer(
    batchSize: int = 2,
    shuffle: bool = False,
) -> Trainer:
    """
    创建一个用于分类任务的训练器
    """
    model = SequentialModel(
        [
            LinearLayer(
                inputDim=2,
                outputDim=2,
                useBias=True,
                randomSeed=42,
            )
        ]
    )
    lossFunction = CrossEntropyLoss()
    optimizer = SGDOptimizer(learning_rate=0.1)

    trainer = Trainer(
        model=model,
        lossFunction=lossFunction,
        optimizer=optimizer,
        taskType="classification",
        batchSize=batchSize,
        shuffle=shuffle,
        randomSeed=42,
    )
    return trainer


def createRegressionTrainer(
    batchSize: int = 2,
    shuffle: bool = False,
) -> Trainer:
    """
    创建一个用于回归任务的训练器
    """
    model = SequentialModel(
        [
            LinearLayer(
                inputDim=1,
                outputDim=1,
                useBias=True,
                randomSeed=42,
            )
        ]
    )
    lossFunction = MSELoss()
    optimizer = SGDOptimizer(learning_rate=0.05)

    trainer = Trainer(
        model=model,
        lossFunction=lossFunction,
        optimizer=optimizer,
        taskType="regression",
        batchSize=batchSize,
        shuffle=shuffle,
        randomSeed=42,
    )
    return trainer


def createClassificationDataset() -> tuple[np.ndarray, np.ndarray]:
    """
    创建简单分类数据集
    """
    inputData = np.array(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0],
        ],
        dtype=np.float64,
    )
    targetData = np.array([0, 0, 1, 1], dtype=np.int64)
    return inputData, targetData


def createRegressionDataset() -> tuple[np.ndarray, np.ndarray]:
    """
    创建简单回归数据集
    """
    inputData = np.array(
        [
            [0.0],
            [1.0],
            [2.0],
            [3.0],
        ],
        dtype=np.float64,
    )
    targetData = np.array(
        [
            [1.0],
            [3.0],
            [5.0],
            [7.0],
        ],
        dtype=np.float64,
    )
    return inputData, targetData


def testInitRaisesWhenBatchSizeIsInvalid() -> None:
    """
    测试 batchSize 非法时抛出异常
    """
    model = SequentialModel([LinearLayer(2, 2)])
    lossFunction = CrossEntropyLoss()
    optimizer = SGDOptimizer(learning_rate=0.1)

    with pytest.raises(ValueError):
        Trainer(
            model=model,
            lossFunction=lossFunction,
            optimizer=optimizer,
            taskType="classification",
            batchSize=0,
        )


def testValidateDatasetRaisesWhenSampleCountMismatch() -> None:
    """
    测试样本数量不一致时抛出异常
    """
    trainer = createClassificationTrainer()

    inputData = np.ones((4, 2), dtype=np.float64)
    targetData = np.array([0, 1, 0], dtype=np.int64)

    with pytest.raises(ValueError):
        trainer.validateDataset(inputData, targetData)


def testCreateBatchesWithoutShuffle() -> None:
    """
    测试不打乱时的 batch 划分
    """
    trainer = createClassificationTrainer(batchSize=2, shuffle=False)
    inputData, targetData = createClassificationDataset()

    batches = trainer.createBatches(inputData, targetData)

    assert len(batches) == 2

    firstInputs, firstTargets = batches[0]
    secondInputs, secondTargets = batches[1]

    np.testing.assert_array_equal(firstInputs, inputData[:2])
    np.testing.assert_array_equal(firstTargets, targetData[:2])

    np.testing.assert_array_equal(secondInputs, inputData[2:])
    np.testing.assert_array_equal(secondTargets, targetData[2:])


def testCreateBatchesKeepsAllSamples() -> None:
    """
    测试 batch 划分后不会丢样本
    """
    trainer = createRegressionTrainer(batchSize=3, shuffle=False)
    inputData, targetData = createRegressionDataset()

    batches = trainer.createBatches(inputData, targetData)

    totalSampleCount = sum(batchInputs.shape[0] for batchInputs, _ in batches)

    assert len(batches) == 2
    assert totalSampleCount == inputData.shape[0]


def testTrainStepUpdatesClassificationModelParameters() -> None:
    """
    测试分类任务单步训练会更新参数
    """
    trainer = createClassificationTrainer(batchSize=4, shuffle=False)
    inputData, targetData = createClassificationDataset()

    layer = trainer.model.layers[0]
    assert isinstance(layer, LinearLayer)

    oldWeights = layer.weights.copy()
    oldBias = None if layer.bias is None else layer.bias.copy()

    loss = trainer.trainStep(inputData, targetData)

    assert isinstance(loss, float)
    assert loss > 0.0
    assert not np.allclose(layer.weights, oldWeights)

    if layer.bias is not None and oldBias is not None:
        assert not np.allclose(layer.bias, oldBias)


def testTrainStepUpdatesRegressionModelParameters() -> None:
    """
    测试回归任务单步训练会更新参数
    """
    trainer = createRegressionTrainer(batchSize=4, shuffle=False)
    inputData, targetData = createRegressionDataset()

    layer = trainer.model.layers[0]
    assert isinstance(layer, LinearLayer)

    oldWeights = layer.weights.copy()
    oldBias = None if layer.bias is None else layer.bias.copy()

    loss = trainer.trainStep(inputData, targetData)

    assert isinstance(loss, float)
    assert loss >= 0.0
    assert not np.allclose(layer.weights, oldWeights)

    if layer.bias is not None and oldBias is not None:
        assert not np.allclose(layer.bias, oldBias)


def testTrainEpochReturnsFloatLoss() -> None:
    """
    测试 trainEpoch 返回浮点损失值
    """
    trainer = createClassificationTrainer(batchSize=2, shuffle=False)
    inputData, targetData = createClassificationDataset()

    epochLoss = trainer.trainEpoch(inputData, targetData)

    assert isinstance(epochLoss, float)
    assert epochLoss > 0.0


def testComputeMetricForClassification() -> None:
    """
    测试分类任务指标计算
    """
    trainer = createClassificationTrainer()

    predictions = np.array(
        [
            [2.0, 1.0],
            [3.0, 0.5],
            [0.1, 2.0],
            [0.2, 3.0],
        ],
        dtype=np.float64,
    )
    targetData = np.array([0, 0, 1, 1], dtype=np.int64)

    metricValue = trainer.computeMetric(predictions, targetData)

    assert metricValue == 1.0


def testComputeMetricForRegression() -> None:
    """
    测试回归任务指标计算
    """
    trainer = createRegressionTrainer()

    predictions = np.array(
        [
            [1.0],
            [2.0],
        ],
        dtype=np.float64,
    )
    targetData = np.array(
        [
            [0.0],
            [4.0],
        ],
        dtype=np.float64,
    )

    metricValue = trainer.computeMetric(predictions, targetData)
    expectedValue = np.mean((predictions - targetData) ** 2)

    np.testing.assert_allclose(metricValue, expectedValue)


def testEvaluateReturnsClassificationResult() -> None:
    """
    测试分类任务评估结果
    """
    trainer = createClassificationTrainer(batchSize=4, shuffle=False)
    inputData, targetData = createClassificationDataset()

    result = trainer.evaluate(inputData, targetData)

    assert "loss" in result
    assert "accuracy" in result
    assert isinstance(result["loss"], float)
    assert isinstance(result["accuracy"], float)


def testEvaluateReturnsRegressionResult() -> None:
    """
    测试回归任务评估结果
    """
    trainer = createRegressionTrainer(batchSize=4, shuffle=False)
    inputData, targetData = createRegressionDataset()

    result = trainer.evaluate(inputData, targetData)

    assert "loss" in result
    assert "mse" in result
    assert isinstance(result["loss"], float)
    assert isinstance(result["mse"], float)


def testPredictDelegatesToModel() -> None:
    """
    测试 predict 能正常返回模型输出
    """
    trainer = createRegressionTrainer(batchSize=2, shuffle=False)
    inputData, _ = createRegressionDataset()

    predictions = trainer.predict(inputData)

    assert isinstance(predictions, np.ndarray)
    assert predictions.shape == (4, 1)


def testFitRaisesWhenEpochCountIsInvalid() -> None:
    """
    测试 epochCount 非法时抛出异常
    """
    trainer = createClassificationTrainer()
    inputData, targetData = createClassificationDataset()

    with pytest.raises(ValueError):
        trainer.fit(inputData, targetData, epochCount=0, verbose=False)


def testFitReturnsClassificationHistory() -> None:
    """
    测试分类任务 fit 返回训练历史
    """
    trainer = createClassificationTrainer(batchSize=2, shuffle=False)
    inputData, targetData = createClassificationDataset()

    history = trainer.fit(
        trainInputs=inputData,
        trainTargets=targetData,
        epochCount=3,
        verbose=False,
    )

    assert "train_loss" in history
    assert "train_accuracy" in history
    assert len(history["train_loss"]) == 3
    assert len(history["train_accuracy"]) == 3


def testFitReturnsRegressionHistory() -> None:
    """
    测试回归任务 fit 返回训练历史
    """
    trainer = createRegressionTrainer(batchSize=2, shuffle=False)
    inputData, targetData = createRegressionDataset()

    history = trainer.fit(
        trainInputs=inputData,
        trainTargets=targetData,
        epochCount=3,
        verbose=False,
    )

    assert "train_loss" in history
    assert "train_mse" in history
    assert len(history["train_loss"]) == 3
    assert len(history["train_mse"]) == 3


def testFitWithValidationReturnsValidationHistory() -> None:
    """
    测试带验证集时会返回验证历史
    """
    trainer = createClassificationTrainer(batchSize=2, shuffle=False)
    inputData, targetData = createClassificationDataset()

    history = trainer.fit(
        trainInputs=inputData,
        trainTargets=targetData,
        epochCount=2,
        validInputs=inputData,
        validTargets=targetData,
        verbose=False,
    )

    assert "valid_loss" in history
    assert "valid_accuracy" in history
    assert len(history["train_loss"]) == 2
    assert len(history["valid_loss"]) == 2
    assert len(history["valid_accuracy"]) == 2


def testFitCanReduceRegressionLoss() -> None:
    """
    测试回归任务训练后损失可以下降
    """
    trainer = createRegressionTrainer(batchSize=4, shuffle=False)
    inputData, targetData = createRegressionDataset()

    beforeResult = trainer.evaluate(inputData, targetData)

    trainer.fit(
        trainInputs=inputData,
        trainTargets=targetData,
        epochCount=50,
        verbose=False,
    )

    afterResult = trainer.evaluate(inputData, targetData)

    assert afterResult["loss"] <= beforeResult["loss"]
