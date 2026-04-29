"""
config.py
项目配置模块
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_RANDOM_SEED = 42

DATASETS_DIR = PROJECT_ROOT / "datasets"

XOR_DIR = DATASETS_DIR / "xor"
SPIRAL_DIR = DATASETS_DIR / "spiral"
SINE_DIR = DATASETS_DIR / "sine"

# xor的文件路径
XOR_FILE = XOR_DIR / "xor.npz"

# spiral的文件路径
SPIRAL_TRAIN_FILE = SPIRAL_DIR / "train.npz"
SPIRAL_TEST_FILE = SPIRAL_DIR / "test.npz"
SPIRAL_VALID_FILE = SPIRAL_DIR / "valid.npz"

# sine的文件路径
SINE_TRAIN_FILE = SINE_DIR / "train.npz"
SINE_TEST_FILE = SINE_DIR / "test.npz"
SINE_VALID_FILE = SINE_DIR / "valid.npz"


CLASSIFICATION_TASK_CONFIGS = {
    # epochCount: 训练的总轮数
    # batchSize: 每个训练批次的样本数量
    # learningRate: 学习率，控制模型权重更新的步长
    # hiddenDims: 隐藏层的维度列表，每个元素表示一个隐藏层的神经元数量
    # activation: 激活函数的类型，如 "relu"、"tanh"、"sigmoid"
    "xor": {
        "epochCount": 2000,
        "batchSize": 4,
        "learningRate": 0.1,
        "hiddenDims": [8],
        "activation": "tanh",
    },
    "spiral": {
        "epochCount": 400,
        "batchSize": 64,
        "learningRate": 0.03,
        "hiddenDims": [64, 64],
        "activation": "relu",
    },
}


REGRESSION_TASK_CONFIGS = {
    "sine": {
        "epochCount": 300,
        "batchSize": 32,
        "learningRate": 0.01,
        "hiddenDims": [16, 16],
        "activation": "tanh",
    },
}
