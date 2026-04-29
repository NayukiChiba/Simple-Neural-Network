# Xavier 初始化

## 为什么初始化重要

权重的初始值直接影响梯度的流动。两种经典问题：

1. **梯度消失**：权重初始化太小 → 激活值逐层缩小 → 梯度趋近于 0 → 无法学习
2. **梯度爆炸**：权重初始化太大 → 激活值逐层放大 → 梯度趋近于无穷 → 参数溢出

理想的初始化应当**保持各层激活值和梯度的方差稳定**。

---

## Xavier (Glorot) 初始化的动机

2010 年，Glorot 和 Bengio 提出：对于使用对称激活函数（如 Tanh, Sigmoid）的网络，应在初始化时保持：

$$ \text{Var}(a^{(l)}) \approx \text{Var}(a^{(l-1)}) $$

即相邻层的输出方差相等。

---

## 方差推导

### 假设条件

1. 权重 $W_{ij}$ 独立同分布，均值为 0，方差为 $\sigma^2_w$
2. 输入 $x_i$ 独立同分布，均值为 0，方差为 $\sigma^2_x$
3. 激活函数在原点附近近似线性（如 Tanh 在 $x \approx 0$ 时）

### 前向传播方差

对线性层的第 $j$ 个输出神经元：

$$ y_j = \sum_{i=1}^{n_{in}} W_{ij} x_i + b_j $$

忽略偏置（通常初始化为 0），计算方差：

$$ \begin{aligned} \text{Var}(y_j) &= \text{Var}\left(\sum_{i=1}^{n_{in}} W_{ij} x_i\right) \\ &= \sum_{i=1}^{n_{in}} \text{Var}(W_{ij} x_i) \quad \text{（独立性）} \\ &= \sum_{i=1}^{n_{in}} \text{Var}(W_{ij}) \cdot \text{Var}(x_i) \quad \text{（零均值）} \\ &= n_{in} \cdot \sigma^2_w \cdot \sigma^2_x \end{aligned} $$

要维持方差稳定（$\text{Var}(y) = \text{Var}(x)$），需要：

$$ n_{in} \cdot \sigma^2_w = 1 \quad \Rightarrow \quad \sigma^2_w = \frac{1}{n_{in}} $$

### 反向传播梯度方差

反向传播时对输入梯度进行类似分析，得到：

$$ \sigma^2_w = \frac{1}{n_{out}} $$

### Xavier 折中方案

取两者的调和平均：

$$ \sigma^2_w = \frac{2}{n_{in} + n_{out}} $$

---

## 均匀分布的边界

Xavier 使用该方差的均匀分布 $\mathcal{U}[-a, a]$。均匀分布的方差为：

$$ \text{Var}[\mathcal{U}(-a, a)] = \frac{(a - (-a))^2}{12} = \frac{a^2}{3} $$

令方差等于 $\frac{2}{n_{in} + n_{out}}$：

$$ \frac{a^2}{3} = \frac{2}{n_{in} + n_{out}} $$

$$ a = \sqrt{\frac{6}{n_{in} + n_{out}}} $$

---

## 最终公式

$$ W_{ij} \sim \mathcal{U}\left[-\sqrt{\frac{6}{n_{in} + n_{out}}},\; +\sqrt{\frac{6}{n_{in} + n_{out}}}\right] $$

其中：
- $n_{in}$：输入神经元数（`inputDim`）
- $n_{out}$：输出神经元数（`outputDim`）

---

## 代码实现

`src/nn/layers/linearLayer.py:55-62`:

```python
rng = np.random.default_rng(randomSeed)

# limit = sqrt(6 / (n_in + n_out))
limit = np.sqrt(6.0 / (inputDim + outputDim))

# 从均匀分布采样
self.weights = rng.uniform(
    low=-limit,
    high=limit,
    size=(inputDim, outputDim),
).astype(np.float64)
```

每一步对应数学推导：
1. `np.sqrt(6.0 / (inputDim + outputDim))` 计算 $a = \sqrt{6/(n_{in}+n_{out})}$
2. `rng.uniform(low=-limit, high=limit, ...)` 从 $\mathcal{U}[-a, a]$ 采样
3. `.astype(np.float64)` 确保双精度浮点

---

## 偏置初始化

偏置统一初始化为零：

$$ \mathbf{b} = \mathbf{0} \quad (\text{形状 } 1 \times d_{out}) $$

```python
self.bias = np.zeros(shape=(1, outputDim), dtype=np.float64)
```

偏置通常初始化为 0，因为：
- 零初始值不会破坏对称性（权重已随机）
- 偏置的梯度计算不依赖偏置本身，不会阻止学习

---

## 不同初始化方法对比

| 方法 | 方差 | 适用场景 |
|---|---|---|
| Xavier 均匀 | $\sigma^2 = \frac{2}{n_{in} + n_{out}}$ | Tanh / Sigmoid 激活 |
| He (Kaiming) 均匀 | $\sigma^2 = \frac{2}{n_{in}}$ | ReLU 激活 |
| LeCun 均匀 | $\sigma^2 = \frac{1}{n_{in}}$ | 线性 / SELU |

本项目使用 Xavier，因为默认激活函数是 Tanh。若切换到 ReLU，可以考虑 He 初始化（只考虑 $n_{in}$，因为 ReLU 将一半的神经元置零，方差减半）。
