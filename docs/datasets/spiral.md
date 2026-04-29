# Spiral 数据集

## 概述

Spiral（螺旋）是多类二维分类数据集。三个类别的样本点以阿基米德螺旋线分布，加入高斯噪声。该数据集的决策边界高度非线性，是测试神经网络分类能力的经典问题。

## 数学定义

对于类别 $c \in \{0, 1, 2\}$，生成 $N$ 个样本（默认 $N=1000$）：

$$ r_i = \frac{i}{N}, \quad i = 0, 1, \dots, N-1 $$

$$ \theta_i^{(c)} = 4c + 4 \cdot r_i + \epsilon_i, \quad \epsilon_i \sim \mathcal{N}(0, 0.2) $$

$$ x_{i,0}^{(c)} = r_i \cdot \sin(\theta_i^{(c)}) $$

$$ x_{i,1}^{(c)} = r_i \cdot \cos(\theta_i^{(c)}) $$

$$ y_i^{(c)} = c $$

**直觉**：
- $r_i$ 从 0 到 1 线性递增，控制螺旋的半径
- $4c$ 为每个类别提供 $4$ 弧度的初始角度偏移（$c=0,1,2$），使三类错开分布
- 噪声 $\epsilon_i$ 使每个点的角度随机扰动，数据更加真实

## 数据规格

| 属性 | 值 |
|---|---|
| 总样本数 | 3000（3 类 × 1000） |
| 输入维度 | 2 |
| 类别数 | 3 |
| 标签格式 | 整数索引 `[0, 1, 2]` |
| 噪声 | $\mathcal{N}(0, 0.2)$ |
| 数据类型 | X: `float64`, y: `int64` |

### 划分比例

| 子集 | 比例 | 样本数 |
|---|---|---|
| 训练集 | 70% | 2100 |
| 验证集 | 15% | 450 |
| 测试集 | 15% | 450 |

## 生成代码

`src/nn/data/dataGenerator.py:173-220`:

```python
def generateSpiralDataset(self) -> None:
    classCount = self.SPIRAL_CLASS_COUNT       # 3
    samplesPerClass = self.SPIRAL_SAMPLES_PER_CLASS  # 1000
    totalCount = classCount * samplesPerClass    # 3000

    x = np.zeros((totalCount, 2), dtype=np.float64)
    y = np.zeros(totalCount, dtype=np.int64)

    for classIndex in range(classCount):
        startIndex = classIndex * samplesPerClass
        endIndex = startIndex + samplesPerClass

        radius = np.linspace(0.0, 1.0, samplesPerClass)
        angle = np.linspace(
            classIndex * 4.0,        # 起始角度
            (classIndex + 1) * 4.0,  # 结束角度
            samplesPerClass,
        )

        # 添加高斯噪声
        angle += self.rng.normal(0.0, self.SPIRAL_NOISE_SCALE, samplesPerClass)

        x[startIndex:endIndex, 0] = radius * np.sin(angle)
        x[startIndex:endIndex, 1] = radius * np.cos(angle)
        y[startIndex:endIndex] = classIndex

    trainSet, validSet, testSet = self.splitDataset(x, y)
    # 保存三个子集...
```

## 典型训练配置

| 超参数 | 值 |
|---|---|
| 网络结构 | `Linear(2→64) → ReLU → Linear(64→64) → ReLU → Linear(64→3)` |
| 损失函数 | `CrossEntropyLoss` |
| 优化器 | `SGD(lr=0.03)` |
| Batch size | 64 |
| Epochs | 400 |
| 预期准确率 | **~0.998** |
