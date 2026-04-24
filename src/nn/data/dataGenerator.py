"""
src/nn/dataGenerator.py

数据生成器模块

功能：
1. 生成 xor 数据集
2. 生成 spiral 数据集
3. 生成 sine 数据集
4. 生成前自动创建数据目录

"""

import math
from pathlib import Path

import numpy as np

import config


class DataGenerator:
    """
    数据集生成器

    """

    def __init__(self, seed: int = 42) -> None:
        """
        初始化数据集生成器

        Args:
            seed(int): 随机种子，默认为 42
        """
        randomSeed = seed
        # 使用 numpy 的默认随机数生成器
        # 这个是推荐用法, 不使用原来的np.random.seed(seed)了, 因为它会影响全局的随机状态
        # 而使用rng可以创建一个独立的随机数生成器实例, 不会干扰其他部分的随机数生成
        self.rng = np.random.default_rng(randomSeed)
        self.train_ratio = 0.7
        self.valid_ratio = 0.15
        self.test_ratio = 0.15

        # spiral数据集的参数
        self.SPIRAL_CLASS_COUNT = 3
        self.SPIRAL_SAMPLES_PER_CLASS = 1000
        self.SPIRAL_NOISE_SCALE = 0.2

        # sine数据集的参数
        self.SINE_SAMPLE_COUNT = 600
        self.SINE_X_START = -2.0 * math.pi
        self.SINE_X_END = 2.0 * math.pi
        self.SINE_NOISE_SCALE = 0.1

    def validateSplitRatios(self) -> None:
        """
        验证训练集、测试集、验证集的划分比例
        """
        total = self.train_ratio + self.valid_ratio + self.test_ratio
        if not math.isclose(total, 1.0):
            raise ValueError(
                f"训练集、测试集、验证集的划分比例之和必须为 1.0, 当前为 {total}"
            )

    def createDatasetDir(self) -> None:
        """
        创建所有数据集目录

        """
        datasetDirs = [
            config.DATASETS_DIR,
            config.XOR_DIR,
            config.SPIRAL_DIR,
            config.SINE_DIR,
        ]

        for dir in datasetDirs:
            # 创建目录, 如果目录已经存在则不会报错
            dir.mkdir(parents=True, exist_ok=True)

    def saveDataset(self, filePath: Path, x: np.ndarray, y: np.ndarray) -> None:
        """
        将数据集保存到npz文件中
        Args:
            filePath(Path): 数据集文件路径
            x(np.ndarray): 特征数据
            y(np.ndarray): 标签数据

        """
        filePath.parent.mkdir(parents=True, exist_ok=True)  # 确保目录存在
        np.savez(filePath, x=x, y=y)

    def splitDataset(
        self,
        x: np.ndarray,
        y: np.ndarray,
    ) -> tuple[
        tuple[np.ndarray, np.ndarray],  # 训练集 (x_train, y_train)
        tuple[np.ndarray, np.ndarray],  # 验证集 (x_valid, y_valid)
        tuple[np.ndarray, np.ndarray],  # 测试集 (x_test, y_test)
    ]:
        """
        将数据集划分为训练集、验证集和测试集
        Args:
            x(np.ndarray): 特征数据
            y(np.ndarray): 标签数据
        Returns:
            tuple: 包含训练集、验证集和测试集的元组


        """
        # 样本的数量为
        sampleCount = x.shape[0]
        # 生成一个随机的索引数组
        shuffledIndices = self.rng.permutation(sampleCount)

        # 先随机shuffle数据
        Xshuffled = x[shuffledIndices]
        Yshuffled = y[shuffledIndices]

        # 划分数据集
        # 获取索引划分点
        trainEndIndex = int(sampleCount * self.train_ratio)
        validEndIndex = trainEndIndex + int(sampleCount * self.valid_ratio)

        trainSet = (
            Xshuffled[:trainEndIndex],
            Yshuffled[:trainEndIndex],
        )
        validSet = (
            Xshuffled[trainEndIndex:validEndIndex],
            Yshuffled[trainEndIndex:validEndIndex],
        )
        testSet = (
            Xshuffled[validEndIndex:],
            Yshuffled[validEndIndex:],
        )

        return trainSet, validSet, testSet

    def generateXorDataset(self) -> None:
        """
        生成XOR数据集


        """
        # 创建目录
        self.createDatasetDir()

        x = np.array(
            [
                [0, 0],
                [0, 1],
                [1, 0],
                [1, 1],
            ],
            dtype=np.float64,
        )

        y = np.array([0, 1, 1, 0], dtype=np.float64)

        self.saveDataset(config.XOR_FILE, x, y)

    def generateSpiralDataset(self) -> None:
        """
        生成螺旋分类数据集


        """
        self.createDatasetDir()

        # 每个类别的样本数量
        classCount = self.SPIRAL_CLASS_COUNT
        samplesPerClass = self.SPIRAL_SAMPLES_PER_CLASS
        totalCount = classCount * samplesPerClass

        # 生成一个 totalCount x 2 的特征矩阵和一个 totalCount 的标签向量
        x = np.zeros((totalCount, 2), dtype=np.float64)
        y = np.zeros(totalCount, dtype=np.float64)

        # 为每个类别生成螺旋数据
        for classIndex in range(classCount):
            startIndex = classIndex * samplesPerClass
            endIndex = startIndex + samplesPerClass

            # 生成样本的半径和角度
            radius = np.linspace(0.0, 1.0, samplesPerClass)
            angle = np.linspace(
                classIndex * 4.0,  # 每个类别的起始角度不同
                (classIndex + 1) * 4.0,  # 每个类别的结束角度不同
                samplesPerClass,
            )

            angle += self.rng.normal(
                loc=0.0,  # 均值
                scale=self.SPIRAL_NOISE_SCALE,  # 标准差
                size=samplesPerClass,
            )

            # 根据半径和角度计算特征值
            x[startIndex:endIndex, 0] = radius * np.sin(angle)
            x[startIndex:endIndex, 1] = radius * np.cos(angle)
            y[startIndex:endIndex] = classIndex

        # 划分数据集并保存
        trainSet, validSet, testSet = self.splitDataset(x, y)

        # 保存训练集、验证集和测试集
        self.saveDataset(config.SPIRAL_TRAIN_FILE, *trainSet)
        self.saveDataset(config.SPIRAL_VALID_FILE, *validSet)
        self.saveDataset(config.SPIRAL_TEST_FILE, *testSet)

    def generateSineDataset(self) -> None:
        """
        生成正弦回归数据集
        """
        self.createDatasetDir()

        x = np.linspace(
            self.SINE_X_START,
            self.SINE_X_END,
            self.SINE_SAMPLE_COUNT,
        ).reshape(-1, 1)  # 转换为列向量

        noise = self.rng.normal(
            loc=0.0,  # 均值
            scale=self.SINE_NOISE_SCALE,  # 标准差
            size=(self.SINE_SAMPLE_COUNT, 1),
        )

        y = np.sin(x) + noise

        trainSet, validSet, testSet = self.splitDataset(x, y)

        self.saveDataset(config.SINE_TRAIN_FILE, *trainSet)
        self.saveDataset(config.SINE_VALID_FILE, *validSet)
        self.saveDataset(config.SINE_TEST_FILE, *testSet)

    def generateAllDatasets(self) -> None:
        """
        生成所有数据集
        """
        self.validateSplitRatios()
        self.generateXorDataset()
        self.generateSpiralDataset()
        self.generateSineDataset()
