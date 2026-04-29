# XOR 数据集

## 概述

XOR（异或）是经典的二分类问题，用于验证神经网络学习**非线性关系**的能力。XOR 的输入-输出映射无法用单一线性决策边界表示，因此需要至少一个隐藏层。

## 数学定义

XOR（$\oplus$）的真值表：

| $x_1$ | $x_2$ | $x_1 \oplus x_2$ | 标签 |
|---|---|---|---|
| 0 | 0 | 0 | 0 |
| 0 | 1 | 1 | 1 |
| 1 | 0 | 1 | 1 |
| 1 | 1 | 0 | 0 |

标签定义：输入相同时为 0（同类），输入不同时为 1（异类）。

数据矩阵形式：

$$ \mathbf{X} = \begin{bmatrix} 0 & 0 \\ 0 & 1 \\ 1 & 0 \\ 1 & 1 \end{bmatrix}, \quad \mathbf{y} = [0, 1, 1, 0] $$

## 数据规格

| 属性 | 值 |
|---|---|
| 样本数 | 4 |
| 输入维度 | 2 |
| 类别数 | 2 |
| 标签格式 | 整数索引 `[0, 1]` |
| 数据类型 | X: `float64`, y: `int64` |
| 训练/验证/测试划分 | 无（仅 4 个样本，全部用于训练） |

## 生成代码

`src/nn/data/dataGenerator.py:152-171`:

```python
def generateXorDataset(self) -> None:
    x = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1],
    ], dtype=np.float64)

    y = np.array([0, 1, 1, 0], dtype=np.int64)

    self.saveDataset(config.XOR_FILE, x, y)
```

## 历史意义

1969 年，Minsky 和 Papert 在《Perceptrons》一书中证明：单层感知机无法学习 XOR 函数。这一结论一度导致神经网络研究的"寒冬"。XOR 至今仍是最小的验证非线性学习能力的标准测试案例。

## 典型训练配置

XOR 样本极少（4 个），训练时需要更多 epoch：

| 超参数 | 值 |
|---|---|
| 网络结构 | `Linear(2→8) → Tanh → Linear(8→2)` |
| 损失函数 | `CrossEntropyLoss` |
| 优化器 | `SGD(lr=0.1)` |
| Batch size | 4（全批量） |
| Epochs | 2000 |
| 预期准确率 | **1.0** |
