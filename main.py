"""
项目主入口

功能：
1. 按任务生成或加载数据集
2. 构建对应的简单神经网络模型
3. 执行分类或回归训练
4. 输出训练与测试结果

使用方法：
    python main.py --task xor
    python main.py --task spiral
    python main.py --task sine
"""

import argparse
from pathlib import Path
from typing import Literal

import numpy as np

import config
from src.nn.data import DataGenerator, DataLoader
from src.nn.layers import LinearLayer, ReLULayer, TanhLayer
from src.nn.losses import CrossEntropyLoss, MSELoss
from src.nn.models.sequentialModel import SequentialModel
from src.nn.optimizers import SGDOptimizer
from src.nn.persistence import CheckpointIO
from src.nn.training import Trainer

TaskName = Literal["xor", "spiral", "sine"]


def parseArguments() -> argparse.Namespace:
    """
    解析命令行参数

    Returns:
        argparse.Namespace: 解析结果
    """
    parser = argparse.ArgumentParser(description="简单神经网络训练入口")
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        choices=["xor", "spiral", "sine"],
        help="要执行的任务名称",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="训练轮数，不传时使用任务默认值",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="批大小，不传时使用任务默认值",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help="学习率，不传时使用任务默认值",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子",
    )
    parser.add_argument(
        "--generate-only",
        action="store_true",
        help="仅生成数据集，不执行训练",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="可选的模型保存路径，必须以 .npz 结尾",
    )
    return parser.parse_args()


def ensureDatasetExists(taskName: TaskName, seed: int) -> None:
    """
    确保任务所需数据集文件存在，不存在时自动生成

    Args:
        taskName (TaskName): 任务名称
        seed (int): 随机种子
    """
    generator = DataGenerator(seed=seed)

    if taskName == "xor":
        if not config.XOR_FILE.exists():
            generator.generateXorDataset()
        return

    if taskName == "spiral":
        requiredFiles = [
            config.SPIRAL_TRAIN_FILE,
            config.SPIRAL_VALID_FILE,
            config.SPIRAL_TEST_FILE,
        ]
        if not all(filePath.exists() for filePath in requiredFiles):
            generator.generateSpiralDataset()
        return

    requiredFiles = [
        config.SINE_TRAIN_FILE,
        config.SINE_VALID_FILE,
        config.SINE_TEST_FILE,
    ]
    if not all(filePath.exists() for filePath in requiredFiles):
        generator.generateSineDataset()


def getDefaultHyperParameters(taskName: TaskName) -> dict[str, float | int]:
    """
    获取任务默认超参数

    Args:
        taskName (TaskName): 任务名称

    Returns:
        dict[str, float | int]: 默认超参数
    """
    if taskName == "xor":
        return {
            "epochCount": 1000,
            "batchSize": 4,
            "learningRate": 0.1,
        }

    if taskName == "spiral":
        return {
            "epochCount": 200,
            "batchSize": 64,
            "learningRate": 0.05,
        }

    return {
        "epochCount": 300,
        "batchSize": 32,
        "learningRate": 0.01,
    }


def buildModel(taskName: TaskName, seed: int) -> SequentialModel:
    """
    根据任务构建模型

    Args:
        taskName (TaskName): 任务名称
        seed (int): 随机种子

    Returns:
        SequentialModel: 构建完成的模型
    """
    if taskName == "xor":
        return SequentialModel(
            [
                LinearLayer(inputDim=2, outputDim=4, randomSeed=seed),
                TanhLayer(),
                LinearLayer(inputDim=4, outputDim=2, randomSeed=seed + 1),
            ]
        )

    if taskName == "spiral":
        return SequentialModel(
            [
                LinearLayer(inputDim=2, outputDim=16, randomSeed=seed),
                ReLULayer(),
                LinearLayer(inputDim=16, outputDim=16, randomSeed=seed + 1),
                ReLULayer(),
                LinearLayer(inputDim=16, outputDim=3, randomSeed=seed + 2),
            ]
        )

    return SequentialModel(
        [
            LinearLayer(inputDim=1, outputDim=16, randomSeed=seed),
            TanhLayer(),
            LinearLayer(inputDim=16, outputDim=16, randomSeed=seed + 1),
            TanhLayer(),
            LinearLayer(inputDim=16, outputDim=1, randomSeed=seed + 2),
        ]
    )


def createTrainer(
    taskName: TaskName,
    model: SequentialModel,
    batchSize: int,
    learningRate: float,
    seed: int,
) -> Trainer:
    """
    创建训练器

    Args:
        taskName (TaskName): 任务名称
        model (SequentialModel): 模型
        batchSize (int): 批大小
        learningRate (float): 学习率
        seed (int): 随机种子

    Returns:
        Trainer: 训练器实例
    """
    if taskName in ("xor", "spiral"):
        lossFunction = CrossEntropyLoss()
        taskType: Literal["classification", "regression"] = "classification"
    else:
        lossFunction = MSELoss()
        taskType = "regression"

    optimizer = SGDOptimizer(learning_rate=learningRate)

    return Trainer(
        model=model,
        lossFunction=lossFunction,
        optimizer=optimizer,
        taskType=taskType,
        batchSize=batchSize,
        shuffle=True,
        randomSeed=seed,
    )


def loadTaskDataset(
    taskName: TaskName,
) -> tuple[
    tuple[np.ndarray, np.ndarray],
    tuple[np.ndarray, np.ndarray] | None,
    tuple[np.ndarray, np.ndarray],
]:
    """
    加载任务数据集，并统一返回训练/验证/测试结构

    Args:
        taskName (TaskName): 任务名称

    Returns:
        tuple: (训练集, 验证集, 测试集)
    """
    dataLoader = DataLoader()

    if taskName == "xor":
        xorSet = dataLoader.loadXorDataset()
        return xorSet, None, xorSet

    trainSet, validSet, testSet = dataLoader.loadDataset(taskName)
    return trainSet, validSet, testSet


def normalizeDatasetTargets(
    taskName: TaskName,
    dataset: tuple[np.ndarray, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """
    规范化数据集标签类型

    Args:
        taskName (TaskName): 任务名称
        dataset (tuple[np.ndarray, np.ndarray]): 输入数据与标签

    Returns:
        tuple[np.ndarray, np.ndarray]: 规范化后的数据集
    """
    inputData, targetData = dataset

    if taskName in ("xor", "spiral"):
        return inputData, targetData.astype(np.int64, copy=False)

    return inputData, targetData.astype(np.float64, copy=False)


def printDatasetSummary(
    taskName: TaskName,
    trainSet: tuple[np.ndarray, np.ndarray],
    validSet: tuple[np.ndarray, np.ndarray] | None,
    testSet: tuple[np.ndarray, np.ndarray],
) -> None:
    """
    打印数据集概览

    Args:
        taskName (TaskName): 任务名称
        trainSet (tuple[np.ndarray, np.ndarray]): 训练集
        validSet (tuple[np.ndarray, np.ndarray] | None): 验证集
        testSet (tuple[np.ndarray, np.ndarray]): 测试集
    """
    trainInputs, trainTargets = trainSet
    testInputs, testTargets = testSet

    print(f"任务: {taskName}")
    print(f"训练集: x={trainInputs.shape}, y={trainTargets.shape}")

    if validSet is not None:
        validInputs, validTargets = validSet
        print(f"验证集: x={validInputs.shape}, y={validTargets.shape}")

    print(f"测试集: x={testInputs.shape}, y={testTargets.shape}")


def maybeSaveCheckpoint(model: SequentialModel, checkpointPath: str | None) -> None:
    """
    按需保存模型检查点

    Args:
        model (SequentialModel): 模型
        checkpointPath (str | None): 检查点路径
    """
    if checkpointPath is None:
        return

    checkpointIO = CheckpointIO()
    checkpointIO.saveCheckpoint(model, Path(checkpointPath))
    print(f"已保存检查点: {checkpointPath}")


def main() -> None:
    """
    主函数
    """
    arguments = parseArguments()
    taskName = arguments.task
    seed = arguments.seed

    ensureDatasetExists(taskName, seed)

    if arguments.generate_only:
        print(f"已生成任务 {taskName} 所需数据集")
        return

    hyperParameters = getDefaultHyperParameters(taskName)
    epochCount = (
        arguments.epochs
        if arguments.epochs is not None
        else int(hyperParameters["epochCount"])
    )
    batchSize = (
        arguments.batch_size
        if arguments.batch_size is not None
        else int(hyperParameters["batchSize"])
    )
    learningRate = (
        arguments.learning_rate
        if arguments.learning_rate is not None
        else float(hyperParameters["learningRate"])
    )

    trainSet, validSet, testSet = loadTaskDataset(taskName)
    trainSet = normalizeDatasetTargets(taskName, trainSet)
    testSet = normalizeDatasetTargets(taskName, testSet)
    if validSet is not None:
        validSet = normalizeDatasetTargets(taskName, validSet)
    printDatasetSummary(taskName, trainSet, validSet, testSet)

    model = buildModel(taskName, seed)
    trainer = createTrainer(taskName, model, batchSize, learningRate, seed)

    trainInputs, trainTargets = trainSet

    if validSet is not None:
        validInputs, validTargets = validSet
    else:
        validInputs = None
        validTargets = None

    print(
        f"开始训练: epochs={epochCount}, batch_size={batchSize}, "
        f"learning_rate={learningRate}"
    )

    trainer.fit(
        trainInputs=trainInputs,
        trainTargets=trainTargets,
        epochCount=epochCount,
        validInputs=validInputs,
        validTargets=validTargets,
        verbose=True,
    )

    testInputs, testTargets = testSet
    testResult = trainer.evaluate(testInputs, testTargets)

    print("测试结果:")
    for metricName, metricValue in testResult.items():
        print(f"{metricName}: {metricValue:.6f}")

    maybeSaveCheckpoint(model, arguments.checkpoint)


if __name__ == "__main__":
    main()
