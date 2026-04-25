"""
数据相关子包。

功能：
1. 管理数据集生成逻辑
2. 管理数据集加载逻辑
3. 为训练流程提供统一数据入口
"""

from .dataGenerator import DataGenerator
from .dataLoader import DataLoader, DatasetLoader

__all__: list[str] = ["DataGenerator", "DataLoader", "DatasetLoader"]
