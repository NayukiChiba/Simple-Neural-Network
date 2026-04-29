# Sine 数据集

## 概述

Sine（正弦）是一维回归问题。目标是从带噪声的 $\sin(x)$ 采样点中学习正弦函数。这是验证神经网络回归能力的基本测试案例。

## 数学定义

$$ y = \sin(x) + \epsilon, \quad \epsilon \sim \mathcal{N}(0, 0.1) $$

$$ x \in [-2\pi, 2\pi] $$

$x$ 在区间 $[-2\pi, 2\pi]$ 上均匀采样 600 个点。每个 $y$ 是 $\sin(x)$ 加上独立的高斯噪声。

## 数据规格

| 属性 | 值 |
|---|---|
| 总样本数 | 600 |
| 输入维度 | 1 |
| 输出维度 | 1 |
| 目标类型 | 连续值 |
| x 范围 | $[-2\pi, 2\pi]$ |
| 噪声 | $\mathcal{N}(0, 0.1)$ |
| 数据类型 | X: `float64`, y: `float64` |

### 划分比例

| 子集 | 比例 | 样本数 |
|---|---|---|
| 训练集 | 70% | 420 |
| 验证集 | 15% | 90 |
| 测试集 | 15% | 90 |

## 生成代码

`src/nn/data/dataGenerator.py:222-244`:

```python
def generateSineDataset(self) -> None:
    x = np.linspace(
        self.SINE_X_START,    # -2π
        self.SINE_X_END,      # +2π
        self.SINE_SAMPLE_COUNT,  # 600
    ).reshape(-1, 1)  # 转换为列向量

    noise = self.rng.normal(
        0.0, self.SINE_NOISE_SCALE,  # mean=0, std=0.1
        size=(self.SINE_SAMPLE_COUNT, 1),
    )

    y = np.sin(x) + noise

    trainSet, validSet, testSet = self.splitDataset(x, y)
    # 保存三个子集...
```

## 典型训练配置

| 超参数 | 值 |
|---|---|
| 网络结构 | `Linear(1→16) → Tanh → Linear(16→16) → Tanh → Linear(16→1)` |
| 损失函数 | `MSELoss` |
| 优化器 | `SGD(lr=0.01)` |
| Batch size | 32 |
| Epochs | 300 |

## 与其他数据集对比

| 特性 | XOR | Spiral | Sine |
|---|---|---|---|
| 任务类型 | 二分类 | 多分类 | 回归 |
| 样本数 | 4 | 3000 | 600 |
| 输入维度 | 2 | 2 | 1 |
| 输出维度 | 2 (类别) | 3 (类别) | 1 (连续值) |
| 有划分 | 否 | 是 | 是 |
| 损失函数 | CrossEntropyLoss | CrossEntropyLoss | MSELoss |
