# 激活函数

激活函数为神经网络引入**非线性**，使其能够学习复杂的非线性映射。没有激活函数，多层网络将退化为单层线性变换。

本项目实现了三种激活函数：ReLU、Sigmoid 和 Tanh。

---

## ReLU（Rectified Linear Unit）

### 数学定义

$$ \text{ReLU}(x) = \max(0, x) $$

即：输入大于 0 时原样输出，输入小于或等于 0 时输出 0。

### 导数

$$ \frac{d}{dx}\text{ReLU}(x) = \begin{cases} 1 & x > 0 \\ 0 & x \leq 0 \end{cases} $$

在 $x = 0$ 处导数未定义（实际实现中通常取 0）。

### 反向传播公式

给定上游梯度 $\frac{\partial \mathcal{L}}{\partial \mathbf{y}}$（$\mathbf{y}$ 是激活层的输出），输入梯度为：

$$ \frac{\partial \mathcal{L}}{\partial \mathbf{x}} = \frac{\partial \mathcal{L}}{\partial \mathbf{y}} \odot \mathbb{1}\{\mathbf{x} > 0\} $$

其中 $\odot$ 表示逐元素乘法（Hadamard 积），$\mathbb{1}\{\cdot\}$ 是指示函数。

### 代码实现

`src/nn/layers/activationLayer.py` — `ReLULayer`:

```python
def forward(self, inputData: np.ndarray) -> np.ndarray:
    self.inputCache = inputData                    # 缓存输入，backward 需要 x > 0 判断
    outputData = np.maximum(0, inputData)          # max(0, x)
    self.outputCache = outputData
    return outputData

def backward(self, outputGradient: np.ndarray) -> np.ndarray:
    reluMask = self.inputCache > 0.0               # 构建 mask: x > 0 为 True
    inputGradient = outputGradient * reluMask       # 逐元素乘
    return inputGradient
```

### 特点

- **优点**：计算简单、不饱和（正值梯度始终为 1）、引入稀疏性
- **缺点**："Dying ReLU"——若输入始终为负，梯度为 0，权重不再更新

---

## Sigmoid

### 数学定义

$$ \sigma(x) = \frac{1}{1 + e^{-x}} $$

输出范围 $(0, 1)$，呈 S 形曲线。

### 导数推导

利用商法则或链式法则。简洁推导：

$$ \begin{aligned} \sigma'(x) &= \frac{d}{dx}\left(1 + e^{-x}\right)^{-1} \\ &= -\left(1 + e^{-x}\right)^{-2} \cdot \left(-e^{-x}\right) \\ &= \frac{e^{-x}}{(1 + e^{-x})^2} \\ &= \frac{1}{1 + e^{-x}} \cdot \frac{e^{-x}}{1 + e^{-x}} \\ &= \sigma(x) \cdot \left(1 - \frac{1}{1 + e^{-x}}\right) \\ &= \sigma(x)(1 - \sigma(x)) \end{aligned} $$

### 反向传播公式

导数仅依赖输出值 $\mathbf{y} = \sigma(\mathbf{x})$，因此可利用前向传播的缓存：

$$ \frac{\partial \mathcal{L}}{\partial \mathbf{x}} = \frac{\partial \mathcal{L}}{\partial \mathbf{y}} \odot \mathbf{y} \odot (1 - \mathbf{y}) $$

### 代码实现

`src/nn/layers/activationLayer.py` — `SigmoidLayer`:

```python
def forward(self, inputData: np.ndarray) -> np.ndarray:
    self.inputCache = inputData
    outputData = 1.0 / (1.0 + np.exp(-inputData))  # sigma(x)
    self.outputCache = outputData                   # 缓存输出，backward 用 y*(1-y)
    return outputData

def backward(self, outputGradient: np.ndarray) -> np.ndarray:
    sigmoidOutput = self.outputCache
    # sigma'(x) = sigma(x) * (1 - sigma(x))
    inputGradient = outputGradient * sigmoidOutput * (1.0 - sigmoidOutput)
    return inputGradient
```

### 特点

- **优点**：输出自然解释为概率、处处可导
- **缺点**：输入绝对值大时梯度趋近于 0（饱和）、输出非零中心

---

## Tanh（双曲正切）

### 数学定义

$$ \tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}} = \frac{e^{2x} - 1}{e^{2x} + 1} $$

输出范围 $(-1, 1)$，零中心。

### 导数推导

$$ \begin{aligned} \tanh'(x) &= \frac{d}{dx} \frac{\sinh(x)}{\cosh(x)} \\ &= \frac{\cosh^2(x) - \sinh^2(x)}{\cosh^2(x)} \\ &= 1 - \tanh^2(x) \end{aligned} $$

### 反向传播公式

与 Sigmoid 类似，导数仅依赖输出值 $\mathbf{y} = \tanh(\mathbf{x})$：

$$ \frac{\partial \mathcal{L}}{\partial \mathbf{x}} = \frac{\partial \mathcal{L}}{\partial \mathbf{y}} \odot (1 - \mathbf{y}^2) $$

### 代码实现

`src/nn/layers/activationLayer.py` — `TanhLayer`:

```python
def forward(self, inputData: np.ndarray) -> np.ndarray:
    self.inputCache = inputData
    outputData = np.tanh(inputData)                # tanh(x)
    self.outputCache = outputData                  # 缓存输出，backward 用 1-y^2
    return outputData

def backward(self, outputGradient: np.ndarray) -> np.ndarray:
    tanhOutput = self.outputCache
    # tanh'(x) = 1 - tanh^2(x)
    inputGradient = outputGradient * (1.0 - tanhOutput**2)
    return inputGradient
```

### 特点

- **优点**：零中心输出（优于 Sigmoid）、梯度比 Sigmoid 更大（缓解饱和）
- **缺点**：仍有饱和问题（$|x|$ 大时梯度 $\approx 0$）

---

## 总结对比

| 函数 | 定义 | 输出范围 | 导数 | 特点 |
|---|---|---|---|---|
| ReLU | $\max(0, x)$ | $[0, \infty)$ | $\mathbb{1}\{x>0\}$ | 简单高效，稀疏 |
| Sigmoid | $\frac{1}{1+e^{-x}}$ | $(0, 1)$ | $\sigma(1-\sigma)$ | 概率解释，饱和 |
| Tanh | $\frac{e^x-e^{-x}}{e^x+e^{-x}}$ | $(-1, 1)$ | $1-\tanh^2$ | 零中心，饱和 |

**推荐实践**：
- 隐藏层优先使用 **ReLU**（现代默认选择）
- 二分类输出层使用 **Sigmoid**
- 需要零中心输出时使用 **Tanh**
