"""
网络层相关子包。

功能：
1. 定义基础层接口
2. 提供线性层与激活层实现
3. 管理前向与反向传播需要的层级能力
"""

from .baseLayer import BaseLayer
from .linearLayer import LinearLayer

__all__: list[str] = ["BaseLayer", "LinearLayer"]
