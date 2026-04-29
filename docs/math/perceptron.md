# 感知机与线性变换

## 从生物神经元到数学模型

神经网络的最基本计算单元是**人工神经元**，其灵感来源于生物神经元。一个神经元接收多个输入信号，对这些信号加权求和后加上偏置，产生输出。

### 单个神经元的数学定义

设输入为 $n$ 维向量 $\mathbf{x} = [x_1, x_2, \dots, x_n]^T$，每个输入有一个对应的权重 $w_i$，外加一个偏置项 $b$。神经元的输出 $z$ 为：

$$ z = \sum_{i=1}^{n} w_i x_i + b $$

### 向量化表示

将所有权重写为行向量 $\mathbf{w} = [w_1, w_2, \dots, w_n]$，则上式可简洁表示为内积形式：

$$ z = \mathbf{w} \cdot \mathbf{x} + b = \mathbf{w}^T \mathbf{x} + b $$

### 多个神经元并行：矩阵形式

在实际网络中，一层通常包含多个神经元。设输入维度为 $d_{in}$，输出维度为 $d_{out}$（即该层有 $d_{out}$ 个神经元），则权重组织为矩阵 $\mathbf{W} \in \mathbb{R}^{d_{in} \times d_{out}}$，偏置为行向量 $\mathbf{b} \in \mathbb{R}^{1 \times d_{out}}$。

对于单个样本 $\mathbf{x} \in \mathbb{R}^{1 \times d_{in}}$：

$$ \mathbf{z} = \mathbf{x}\mathbf{W} + \mathbf{b} $$

其中 $\mathbf{z} \in \mathbb{R}^{1 \times d_{out}}$ 是该层所有神经元的输出。

### 批量处理

对于 $N$ 个样本组成的批量 $\mathbf{X} \in \mathbb{R}^{N \times d_{in}}$：

$$ \mathbf{Z} = \mathbf{X}\mathbf{W} + \mathbf{b} $$

**形状分析**:

$$ \underbrace{\mathbf{X}}_{N \times d_{in}} \cdot \underbrace{\mathbf{W}}_{d_{in} \times d_{out}} = \underbrace{\mathbf{XW}}_{N \times d_{out}} $$

偏置 $\mathbf{b}$ 的形状为 $(1, d_{out})$，通过 NumPy 的**广播机制**自动扩展到 $(N, d_{out})$ 再与 $\mathbf{XW}$ 相加。

---

## 代码实现：LinearLayer.forward()

以上数学公式在 `src/nn/layers/linearLayer.py` 中的直接对应：

```python
def forward(self, inputData: np.ndarray) -> np.ndarray:
    # inputData: (batchSize, inputDim)

    self.inputCache = inputData

    # Y = X @ W  矩阵乘法，形状: (batchSize, outputDim)
    outputData = inputData @ self.weights

    # Y = X @ W + b  加上偏置（广播）
    if self.useBias and self.bias is not None:
        outputData = outputData + self.bias

    self.outputCache = outputData
    return outputData
```

关键点：
- `@` 是 NumPy 的矩阵乘法运算符，对应公式中的矩阵乘积
- `+ self.bias` 利用了 NumPy 广播——`self.bias` 形状 $(1, d_{out})$ 自动复制到每一行
- `self.inputCache` 保存输入，供反向传播计算权重梯度使用

---

## 权重初始化：Xavier (Glorot) 均匀分布

权重初始值对训练至关重要。本项目使用 **Xavier 均匀初始化**：

$$ W_{ij} \sim \mathcal{U}\left[-\sqrt{\frac{6}{n_{in} + n_{out}}},\; \sqrt{\frac{6}{n_{in} + n_{out}}}\right] $$

其中 $n_{in} = d_{in}$ 为输入维度，$n_{out} = d_{out}$ 为输出维度。
$\mathcal{U}[a, b]$ 表示区间 $[a, b]$ 上的均匀分布。

```python
rng = np.random.default_rng(randomSeed)
limit = np.sqrt(6.0 / (inputDim + outputDim))
self.weights = rng.uniform(
    low=-limit, high=limit, size=(inputDim, outputDim),
).astype(np.float64)
```

详细的 Xavier 初始化数学推导见 [Xavier 初始化](/math/xavier-init) 章节。

---

## 为什么线性变换不够

如果神经网络只由线性变换组成，无论堆叠多少层，最终效果等价于单个线性变换：

$$ \mathbf{Y} = \mathbf{X}\mathbf{W}_1 \mathbf{W}_2 = \mathbf{X}(\mathbf{W}_1 \mathbf{W}_2) = \mathbf{X}\mathbf{W}' $$

这意味着没有隐藏层的表达能力提升。因此需要在每个线性层之后引入**非线性激活函数**，见下一章 [激活函数](/math/activations)。
