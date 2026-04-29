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
from typing import Literal

import numpy as np

import config
from src.nn.data import DataGenerator, DataLoader
from src.nn.layers import LinearLayer, ReLULayer, TanhLayer
from src.nn.losses import CrossEntropyLoss, MSELoss
from src.nn.models.sequentialModel import SequentialModel
from src.nn.optimizers import SGDOptimizer
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
    return parser.parse_args()


def ensureDatasetExists(taskName: TaskName) -> None:
    """
    确保任务所需数据集文件存在，不存在时自动生成

    Args:
        taskName (TaskName): 任务名称
    """
    generator = DataGenerator(seed=config.DEFAULT_RANDOM_SEED)

    if taskName == "xor":
        if not config.XOR_FILE.exists():
            print("未找到 xor 数据集，开始自动生成...")
            generator.generateXorDataset()
        return

    if taskName == "spiral":
        requiredFiles = [
            config.SPIRAL_TRAIN_FILE,
            config.SPIRAL_VALID_FILE,
            config.SPIRAL_TEST_FILE,
        ]
        if not all(filePath.exists() for filePath in requiredFiles):
            print("未找到 spiral 数据集，开始自动生成...")
            generator.generateSpiralDataset()
        return

    requiredFiles = [
        config.SINE_TRAIN_FILE,
        config.SINE_VALID_FILE,
        config.SINE_TEST_FILE,
    ]
    if not all(filePath.exists() for filePath in requiredFiles):
        print("未找到 sine 数据集，开始自动生成...")
        generator.generateSineDataset()


def getTaskType(taskName: TaskName) -> Literal["classification", "regression"]:
    """
    获取任务类型

    Args:
        taskName (TaskName): 任务名称

    Returns:
        Literal["classification", "regression"]: 任务类型
    """
    if taskName in config.CLASSIFICATION_TASK_CONFIGS:
        return "classification"

    return "regression"


def getTaskHyperParameters(taskName: TaskName) -> dict[str, object]:
    """
    获取任务超参数

    Args:
        taskName (TaskName): 任务名称

    Returns:
        dict[str, object]: 任务超参数
    """
    if taskName in config.CLASSIFICATION_TASK_CONFIGS:
        return config.CLASSIFICATION_TASK_CONFIGS[taskName]

    return config.REGRESSION_TASK_CONFIGS[taskName]


def createActivationLayer(activationName: str) -> ReLULayer | TanhLayer:
    """
    根据名称创建激活层

    Args:
        activationName (str): 激活函数名称

    Returns:
        ReLULayer | TanhLayer: 激活层实例
    """
    if activationName == "relu":
        return ReLULayer()

    if activationName == "tanh":
        return TanhLayer()

    raise ValueError(f"不支持的激活函数: {activationName}")


def buildModel(
    taskName: TaskName,
    inputDim: int,
    outputDim: int,
) -> SequentialModel:
    """
    根据任务构建模型

    Args:
        taskName (TaskName): 任务名称
        inputDim (int): 输入维度
        outputDim (int): 输出维度

    Returns:
        SequentialModel: 构建完成的模型
    """
    hyperParameters = getTaskHyperParameters(taskName)
    hiddenDims = hyperParameters["hiddenDims"]
    activationName = hyperParameters["activation"]

    assert isinstance(hiddenDims, list)
    assert isinstance(activationName, str)

    layers: list[LinearLayer | ReLULayer | TanhLayer] = []
    currentInputDim = inputDim
    currentSeed = config.DEFAULT_RANDOM_SEED

    for hiddenDim in hiddenDims:
        layers.append(
            LinearLayer(
                inputDim=currentInputDim,
                outputDim=int(hiddenDim),
                randomSeed=currentSeed,
            )
        )
        layers.append(createActivationLayer(activationName))
        currentInputDim = int(hiddenDim)
        currentSeed += 1

    layers.append(
        LinearLayer(
            inputDim=currentInputDim,
            outputDim=outputDim,
            randomSeed=currentSeed,
        )
    )

    return SequentialModel(layers)


def createTrainer(
    taskName: TaskName,
    model: SequentialModel,
) -> Trainer:
    """
    创建训练器

    Args:
        taskName (TaskName): 任务名称
        model (SequentialModel): 模型

    Returns:
        Trainer: 训练器实例
    """
    hyperParameters = getTaskHyperParameters(taskName)
    learningRate = hyperParameters["learningRate"]
    batchSize = hyperParameters["batchSize"]

    assert isinstance(learningRate, float)
    assert isinstance(batchSize, int)

    if getTaskType(taskName) == "classification":
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
        randomSeed=config.DEFAULT_RANDOM_SEED,
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


def main() -> None:
    """
    主函数
    """
    arguments = parseArguments()
    taskName = arguments.task

    ensureDatasetExists(taskName)
    hyperParameters = getTaskHyperParameters(taskName)

    epochCount = hyperParameters["epochCount"]
    batchSize = hyperParameters["batchSize"]
    learningRate = hyperParameters["learningRate"]

    assert isinstance(epochCount, int)
    assert isinstance(batchSize, int)
    assert isinstance(learningRate, float)

    trainSet, validSet, testSet = loadTaskDataset(taskName)
    trainSet = normalizeDatasetTargets(taskName, trainSet)
    testSet = normalizeDatasetTargets(taskName, testSet)
    if validSet is not None:
        validSet = normalizeDatasetTargets(taskName, validSet)
    printDatasetSummary(taskName, trainSet, validSet, testSet)

    trainInputs, trainTargets = trainSet

    if validSet is not None:
        validInputs, validTargets = validSet
    else:
        validInputs = None
        validTargets = None

    inputDim = trainInputs.shape[1]
    if getTaskType(taskName) == "classification":
        outputDim = int(np.max(trainTargets)) + 1
    else:
        outputDim = trainTargets.shape[1]

    print(
        f"开始训练: epochs={epochCount}, batch_size={batchSize}, "
        f"learning_rate={learningRate}"
    )

    model = buildModel(taskName, inputDim, outputDim)
    trainer = createTrainer(taskName, model)

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


if __name__ == "__main__":
    main()
