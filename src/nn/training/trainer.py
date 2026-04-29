"""
训练器模块

1. 管理训练循环
2. 支持分类与回归任务
3. 提供训练、评估和预测接口
"""

from typing import Literal

import numpy as np

from src.nn.losses.crossEntropyLoss import CrossEntropyLoss
from src.nn.losses.mseLoss import MSELoss
from src.nn.models.sequentialModel import SequentialModel
from src.nn.optimizers.sgdOptimizer import SGDOptimizer


class Trainer:
    """
    模型训练器

    1. 负责组织前向传播、损失计算、反向传播和参数更新
    2. 支持分类任务和回归任务
    3. 支持 mini-batch 训练

    """

    def __init__(
        self,
        model: SequentialModel,
        lossFunction: CrossEntropyLoss | MSELoss,
        optimizer: SGDOptimizer,
        taskType: Literal["classification", "regression"],
        batchSize: int = 32,
        shuffle: bool = True,
        randomSeed: int | None = None,
    ) -> None:
        """
        Args:
            model (SequentialModel): 顺序模型
            lossFunction (CrossEntropyLoss | MSELoss): 损失函数
            optimizer (SGDOptimizer): 优化器
            taskType (Literal["classification", "regression"]): 任务类型
            batchSize (int): 批大小
            shuffle (bool): 是否在每个 epoch 开始时打乱数据
            randomSeed (Optional[int]): 随机种子
        """

        if batchSize <= 0:
            raise ValueError("batchSize 必须大于 0")

        self.model = model
        self.lossFunction = lossFunction
        self.optimizer = optimizer
        self.taskType = taskType
        self.batchSize = batchSize
        self.shuffle = shuffle
        self.rng = np.random.default_rng(randomSeed)

    def validateDataset(self, inputData: np.ndarray, targetData: np.ndarray) -> None:
        """
        验证输入数据和目标数据的形状是否合法
        Args:
            inputData (np.ndarray): 输入数据，形状为 (numSamples, numFeatures)
            targetData (np.ndarray): 目标数据，形状为 (numSamples, numClasses) 或 (numSamples, 1)
        """
        if inputData.ndim == 0:
            raise ValueError("inputData 不能是标量")

        if targetData.ndim == 0:
            raise ValueError("targetData 不能是标量")

        if inputData.shape[0] == 0:
            raise ValueError("inputData 不能为空")
        if targetData.shape[0] == 0:
            raise ValueError("targetData 不能为空")

        if inputData.shape[0] != targetData.shape[0]:
            raise ValueError(
                f"输入数据和目标数据的样本数量必须相同"
                f"inputData 样本数量: {inputData.shape[0]}, targetData 样本数量: {targetData.shape[0]}"
            )

    def createBatches(
        self, inputData: np.ndarray, targetData: np.ndarray
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """
        将数据划分为 mini-batch
        Args:
            inputData (np.ndarray): 输入数据，形状为 (numSamples, numFeatures)
            targetData (np.ndarray): 目标数据，形状为 (numSamples, numClasses) 或 (numSamples, 1)
        Returns:
            list[tuple[np.ndarray, np.ndarray]]: mini-batch 列表，每个元素是一个 (inputBatch, targetBatch) 元组
        """
        self.validateDataset(inputData, targetData)

        # 获取样本数量
        sampleCount = inputData.shape[0]
        # 生成样本索引
        indices = np.arange(sampleCount)

        if self.shuffle:
            indices = self.rng.permutation(indices)

        # 打乱输入数据和目标数据
        shuffledInputs = inputData[indices]
        shuffledTargets = targetData[indices]

        batches: list[tuple[np.ndarray, np.ndarray]] = []

        # 按照 batchSize 划分数据
        for startIndex in range(0, sampleCount, self.batchSize):
            endIndex = startIndex + self.batchSize
            batchInputs = shuffledInputs[startIndex:endIndex]
            batchTargets = shuffledTargets[startIndex:endIndex]
            batches.append((batchInputs, batchTargets))

        return batches

    def trainStep(self, inputBatch: np.ndarray, targetBatch: np.ndarray) -> float:
        """
        执行一个训练步骤
        Args:
            inputBatch (np.ndarray): 输入批数据，形状为 (batchSize, numFeatures)
            targetBatch (np.ndarray): 目标批数据，形状为 (batchSize, numClasses) 或 (batchSize, 1)
        Returns:
            float: 当前批的损失值
        """
        # 设置模型为训练模式
        self.model.train()

        # 清零梯度
        self.optimizer.zeroGrad(self.model.layers)

        # 前向传播
        predictions = self.model.forward(inputBatch)

        # 计算损失
        loss = self.lossFunction.forward(predictions, targetBatch)

        # 反向传播
        outputGradient = self.lossFunction.backward()

        # 将损失函数的输出梯度传递给模型进行反向传播
        self.model.backward(outputGradient)
        # 更新模型参数
        self.optimizer.step(self.model.layers)

        return float(loss)

    def trainEpoch(self, inputData: np.ndarray, targetData: np.ndarray) -> float:
        """
        执行一个训练 epoch

        Args:
            inputData (np.ndarray): 输入数据，形状为 (numSamples, numFeatures)
            targetData (np.ndarray): 目标数据，形状为 (numSamples, numClasses) 或 (numSamples, 1)

        Returns:
            float: 当前 epoch 的平均损失值
        """
        batches = self.createBatches(inputData, targetData)

        # 记录所有批次损失的加权和。
        # 注意：trainStep 返回的是“当前 batch 的平均损失”，
        # 不是这个 batch 的总损失。
        # 因此这里需要乘以当前 batch 的样本数，
        # 先还原为该 batch 对应的总损失，再在整个 epoch 结束后
        # 除以总样本数，得到整个训练集上的平均损失。
        totalLoss = 0.0

        for inputBatch, targetBatch in batches:
            batchLoss = self.trainStep(inputBatch, targetBatch)

            # 将 batch 平均损失转换为 batch 总损失
            totalLoss += batchLoss * inputBatch.shape[0]

        # 用整个训练集的样本总数求平均，得到当前 epoch 的平均损失
        averageLoss = totalLoss / inputData.shape[0]
        return averageLoss

    def computeMetric(self, predictions: np.ndarray, targetData: np.ndarray) -> float:
        """
        计算评估指标
        Args:
            predictions (np.ndarray): 模型预测的输出，形状为 (batchSize,)
            targetData (np.ndarray): 目标批数据，形状为 (batchSize,)
        Returns:
            float: 评估指标值
        """
        if self.taskType == "classification":
            # 对于分类任务, 计算准确率
            predictedLabels = np.argmax(predictions, axis=1)
            accuracy = np.mean(predictedLabels == targetData)
            return float(accuracy)
        elif self.taskType == "regression":
            mse = np.mean((predictions - targetData) ** 2)
            return float(mse)
        else:
            raise ValueError(f"不支持的任务类型: {self.taskType}")

    def evaluate(
        self,
        inputData: np.ndarray,
        targetData: np.ndarray,
    ) -> dict[str, float]:
        """
        在给定数据集中执行评估

        Args:
            inputData (np.ndarray): 输入数据，形状为 (numSamples, numFeatures)
            targetData (np.ndarray): 目标数据，形状为 (numSamples, numClasses) 或 (numSamples, 1)

        Returns:
            dict[str, float]: 包含损失值和评估指标值的字典


        """
        # 先验证数据集的合法性
        self.validateDataset(inputData, targetData)
        # 设置模型为评估模式
        wasTraining = self.model.isTraining
        self.model.eval()

        try:
            # 先前向传播得到预测结果, 以便计算损失和评估指标
            predictions = self.model.forward(inputData)
            loss = self.lossFunction.forward(predictions, targetData)

        finally:
            if wasTraining:
                self.model.train()  # 恢复模型的训练模式
            else:
                # 如果模型原本就是评估模式, 就保持评估模式
                self.model.eval()

        result = {
            "loss": float(loss),
        }

        if self.taskType == "classification":
            metricValue = self.computeMetric(predictions, targetData)
            result["accuracy"] = float(metricValue)
        else:
            metricValue = float(loss)
            result["mse"] = metricValue

        return result

    def fit(
        self,
        trainInputs: np.ndarray,
        trainTargets: np.ndarray,
        epochCount: int,
        validInputs: np.ndarray | None = None,
        validTargets: np.ndarray | None = None,
        verbose: bool = True,
    ) -> dict[str, list[float]]:
        """
        训练模型
        Args:
            trainInputs (np.ndarray): 训练输入数据，形状为 (numSamples, numFeatures)
            trainTargets (np.ndarray): 训练目标数据，形状为 (numSamples, numClasses) 或 (numSamples, 1)
            epochCount (int): 训练的 epoch 数量
            validInputs (Optional[np.ndarray]): 验证输入数据，形状为 (numValidSamples, numFeatures)
            validTargets (Optional[np.ndarray]): 验证目标数据，形状为 (numValidSamples, numClasses) 或 (numValidSamples, 1)
            verbose (bool): 是否在每个 epoch 结束时打印训练和验证结果

        Returns:
            dict[str, list[float]]: 包含训练损失、验证损失和评估指标的字典，每个值都是一个列表，记录每个 epoch 的结果
        """

        if epochCount <= 0:
            raise ValueError("epochCount 必须大于 0")

        # 验证训练数据集的合法性
        self.validateDataset(trainInputs, trainTargets)

        # pr #22: 验证验证数据集的合法性, 但只有当 validInputs 和 validTargets 都不为 None 时才进行验证, 因为如果其中一个为 None, 另一个也必须为 None, 否则就是非法输入
        if (validInputs is None) != (validTargets is None):
            raise ValueError("validInputs 和 validTargets 必须同时提供")

        hasValidation = validInputs is not None
        if hasValidation:
            self.validateDataset(validInputs, validTargets)

        history: dict[str, list[float]] = {
            "train_loss": [],
        }

        # 初始化评估指标的历史记录列表
        if self.taskType == "classification":
            history["train_accuracy"] = []
        else:
            history["train_mse"] = []

        if hasValidation:
            history["valid_loss"] = []

            if self.taskType == "classification":
                history["valid_accuracy"] = []
            else:
                history["valid_mse"] = []

        # 训练循环
        for epochIndex in range(epochCount):
            # 执行一个训练 epoch，并复用其返回的平均训练损失
            trainLoss = self.trainEpoch(trainInputs, trainTargets)

            # 训练指标仍通过 evaluate 统一计算，避免在 fit 中重复实现逻辑
            trainResult = self.evaluate(trainInputs, trainTargets)
            history["train_loss"].append(trainLoss)

            if self.taskType == "classification":
                history["train_accuracy"].append(trainResult["accuracy"])
            else:
                history["train_mse"].append(trainResult["mse"])

            if hasValidation and validInputs is not None and validTargets is not None:
                # 评估验证集上的损失和评估指标
                validResult = self.evaluate(validInputs, validTargets)
                history["valid_loss"].append(validResult["loss"])

                if self.taskType == "classification":
                    history["valid_accuracy"].append(validResult["accuracy"])
                else:
                    history["valid_mse"].append(validResult["mse"])

            # pr #22: 只有当 verbose 为 True 时才打印训练和验证结果, 因为有时候我们可能只想获取历史记录而不需要打印结果, 这样可以避免不必要的输出, 提高效率
            if verbose:
                epochNumber = epochIndex + 1
                if self.taskType == "classification":
                    message = (
                        f"epoch {epochNumber}/{epochCount} - "
                        f"train_loss: {trainLoss:.6f} - "
                        f"train_accuracy: {trainResult['accuracy']:.6f}"
                    )

                    if (
                        hasValidation
                        and validInputs is not None
                        and validTargets is not None
                    ):
                        message += (
                            f" - valid_loss: {validResult['loss']:.6f}"
                            f" - valid_accuracy: {validResult['accuracy']:.6f}"
                        )

                else:
                    message = (
                        f"epoch {epochNumber}/{epochCount} - "
                        f"train_loss: {trainLoss:.6f} - "
                        f"train_mse: {trainResult['mse']:.6f}"
                    )
                    if (
                        hasValidation
                        and validInputs is not None
                        and validTargets is not None
                    ):
                        message += (
                            f" - valid_loss: {validResult['loss']:.6f}"
                            f" - valid_mse: {validResult['mse']:.6f}"
                        )

                print(message)
        return history

    def predict(self, inputData: np.ndarray) -> np.ndarray:
        """
        使用模型进行预测
        Args:
            inputData (np.ndarray): 输入数据，形状为 (numSamples, numFeatures)

        Returns:
            np.ndarray: 模型预测的输出，形状为 (numSamples, numClasses) 或 (numSamples, 1)

        """
        return self.model.predict(inputData)
