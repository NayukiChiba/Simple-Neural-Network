"""
src/nn/data/dataLoader.py
数据集加载器模块

1. 加载 xor 数据集
2. 加载 spiral 数据集
3. 加载 sine 数据集


"""

from pathlib import Path

# Literal 用于限制变量的取值范围，TypeAlias 用于定义类型别名
from typing import Literal, TypeAlias

import numpy as np

import config

# 数据集类型
DatasetPair: TypeAlias = tuple[np.ndarray, np.ndarray]
# 训练集、验证集、测试集
SplitDataset: TypeAlias = tuple[DatasetPair, DatasetPair, DatasetPair]


class DataLoader:
    """
    数据加载器

    """

    def loadNpzFile(self, filePath: Path, datasetName: str) -> DatasetPair:
        """
        加载单个npz数据文件

        Args:
            filePath (Path): npz文件路径
            datasetName (str): 数据集名称

        Returns:
            DatasetPair: 包含输入数据和标签的元组
        """

        # 检查文件是否存在
        if not filePath.exists():
            raise FileNotFoundError(f"{datasetName}数据文件 {filePath} 不存在")

        # 检查字段是否存在
        with np.load(filePath, allow_pickle=False) as datasetFile:
            if "x" not in datasetFile.files or "y" not in datasetFile.files:
                raise ValueError(
                    f"{datasetName}数据文件 {filePath} 中缺少 'x' 或 'y' 字段"
                )

            x = datasetFile["x"]
            y = datasetFile["y"]

        # 验证数据集的有效性
        self.validateDataset(x, y, datasetName)

        return x, y

    def validateDataset(self, x: np.ndarray, y: np.ndarray, datasetName: str) -> None:
        """
        验证数据集的有效性

        Args:
            x (np.ndarray): 输入数据
            y (np.ndarray): 标签数据
            datasetName (str): 数据集名称

        Raises:
            TypeError: 数据类型不正确
            ValueError: 数据内容不合法
        """
        # 先查数据类型
        if not isinstance(x, np.ndarray):
            raise TypeError(f"{datasetName}数据的输入数据 'x' 应该是一个numpy数组")
        if not isinstance(y, np.ndarray):
            raise TypeError(f"{datasetName}数据的标签数据 'y' 应该是一个numpy数组")

        # 再查数据内容
        if x.ndim == 0:
            raise ValueError(f"{datasetName}数据的输入数据 'x' 不能是标量")
        if y.ndim == 0:
            raise ValueError(f"{datasetName}数据的标签数据 'y' 不能是标量")

        # 检查样本数量
        if x.shape[0] == 0:
            raise ValueError(f"{datasetName}数据的输入数据 'x' 不能包含零个样本")
        if y.shape[0] == 0:
            raise ValueError(f"{datasetName}数据的标签数据 'y' 不能包含零个样本")

        # 检查输入数据和标签的样本数量是否匹配
        if x.shape[0] != y.shape[0]:
            raise ValueError(
                f"{datasetName} 的样本数和标签数不一致: "
                f"x.shape[0]={x.shape[0]}, y.shape[0]={y.shape[0]}"
            )

    def loadSplitDataset(
        self,
        trainFilePath: Path,
        validFilePath: Path,
        testFilePath: Path,
        datasetName: str,
    ) -> SplitDataset:
        """
        加载训练集、验证集和测试集

        Args:
            trainFilePath (Path): 训练集文件路径
            validFilePath (Path): 验证集文件路径
            testFilePath (Path): 测试集文件路径
            datasetName (str): 数据集名称

        Returns:
            SplitDataset: 包含训练集、验证集和测试集的元组
        """

        trainSet = self.loadNpzFile(trainFilePath, f"{datasetName}/train")
        validSet = self.loadNpzFile(validFilePath, f"{datasetName}/valid")
        testSet = self.loadNpzFile(testFilePath, f"{datasetName}/test")

        return trainSet, validSet, testSet

    def loadXorDataset(self) -> DatasetPair:
        """
        加载xor数据集

        Returns:
            DatasetPair: 包含输入数据和标签的元组
        """
        return self.loadNpzFile(config.XOR_FILE, "xor")

    def loadSpiralDataset(self) -> SplitDataset:
        """
        加载spiral数据集

        Returns:
            SplitDataset: 包含训练集、验证集和测试集的元组
        """
        return self.loadSplitDataset(
            config.SPIRAL_TRAIN_FILE,
            config.SPIRAL_VALID_FILE,
            config.SPIRAL_TEST_FILE,
            "spiral",
        )

    def loadSineDataset(self) -> SplitDataset:
        """
        加载sine数据集

        Returns:
            SplitDataset: 包含训练集、验证集和测试集的元组
        """
        return self.loadSplitDataset(
            config.SINE_TRAIN_FILE,
            config.SINE_VALID_FILE,
            config.SINE_TEST_FILE,
            "sine",
        )

    def loadDataset(
        self, datasetName: Literal["xor", "spiral", "sine"]
    ) -> DatasetPair | SplitDataset:
        """
        加载指定名称的数据集

        Args:
            datasetName (Literal["xor", "spiral", "sine"]): 数据集名称

        Returns:
            DatasetPair | SplitDataset: 包含输入数据和标签的元组，或者包含训练集、验证集和测试集的元组
        """
        if datasetName == "xor":
            return self.loadXorDataset()
        elif datasetName == "spiral":
            return self.loadSpiralDataset()
        elif datasetName == "sine":
            return self.loadSineDataset()
        else:
            raise ValueError(f"不支持的数据集名称: {datasetName}")


DatasetLoader = DataLoader
