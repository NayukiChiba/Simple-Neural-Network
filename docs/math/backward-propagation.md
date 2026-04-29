# 反向传播

## 概念

反向传播（Backpropagation）是计算损失函数对神经网络中每个参数的梯度的算法。它应用微积分中的**链式法则**，从输出层向输入层逐层传递梯度。

设损失函数为 $\mathcal{L}$，我们需要计算每个参数的梯度：
- 权重梯度 $\frac{\partial \mathcal{L}}{\partial \mathbf{W}^{(l)}}$
- 偏置梯度 $\frac{\partial \mathcal{L}}{\partial \mathbf{b}^{(l)}}$

这些梯度用于优化器更新参数。

---

## 链式法则

### 标量链式法则

若 $y = f(g(x))$，则：

$$ \frac{dy}{dx} = \frac{dy}{dg} \cdot \frac{dg}{dx} $$

### 矩阵链式法则

神经网络中的操作都是矩阵/向量级别的。考虑一个线性层：

$$ \mathbf{Y} = \mathbf{X}\mathbf{W} + \mathbf{b} $$

给定损失对输出的梯度 $\frac{\partial \mathcal{L}}{\partial \mathbf{Y}}$（由后续层传入），需要计算三个梯度：

$$ \frac{\partial \mathcal{L}}{\partial \mathbf{X}} = \frac{\partial \mathcal{L}}{\partial \mathbf{Y}} \mathbf{W}^T $$

$$ \frac{\partial \mathcal{L}}{\partial \mathbf{W}} = \mathbf{X}^T \frac{\partial \mathcal{L}}{\partial \mathbf{Y}} $$

$$ \frac{\partial \mathcal{L}}{\partial \mathbf{b}} = \sum_{n=1}^{N} \left(\frac{\partial \mathcal{L}}{\partial \mathbf{Y}}\right)_{n, :} $$

---

## 线性层的梯度推导

### 1. 对输入的梯度 $\frac{\partial \mathcal{L}}{\partial \mathbf{X}}$

从元素角度展开。设 $\mathbf{Y} = \mathbf{X}\mathbf{W}$。

$$ Y_{n, j} = \sum_{i=1}^{d_{in}} X_{n, i} \cdot W_{i, j} $$

对某个 $X_{n, i}$ 求链式：

$$ \frac{\partial \mathcal{L}}{\partial X_{n, i}} = \sum_{j=1}^{d_{out}} \frac{\partial \mathcal{L}}{\partial Y_{n, j}} \cdot \frac{\partial Y_{n, j}}{\partial X_{n, i}} = \sum_{j=1}^{d_{out}} \frac{\partial \mathcal{L}}{\partial Y_{n, j}} \cdot W_{i, j} $$

即 $\frac{\partial \mathcal{L}}{\partial \mathbf{X}} = \frac{\partial \mathcal{L}}{\partial \mathbf{Y}} \mathbf{W}^T$。

### 2. 对权重的梯度 $\frac{\partial \mathcal{L}}{\partial \mathbf{W}}$

对某个 $W_{i, j}$：

$$ \frac{\partial \mathcal{L}}{\partial W_{i, j}} = \sum_{n=1}^{N} \frac{\partial \mathcal{L}}{\partial Y_{n, j}} \cdot \frac{\partial Y_{n, j}}{\partial W_{i, j}} = \sum_{n=1}^{N} \frac{\partial \mathcal{L}}{\partial Y_{n, j}} \cdot X_{n, i} $$

即 $\frac{\partial \mathcal{L}}{\partial \mathbf{W}} = \mathbf{X}^T \frac{\partial \mathcal{L}}{\partial \mathbf{Y}}$。

### 3. 对偏置的梯度 $\frac{\partial \mathcal{L}}{\partial \mathbf{b}}$

偏置对每个样本的输出贡献相同：

$$ \frac{\partial \mathcal{L}}{\partial b_j} = \sum_{n=1}^{N} \frac{\partial \mathcal{L}}{\partial Y_{n, j}} $$

即 $\frac{\partial \mathcal{L}}{\partial \mathbf{b}} = \sum_n \left(\frac{\partial \mathcal{L}}{\partial \mathbf{Y}}\right)_{n, :}$（按行求和，`keepdims=True`）。

---

## 激活层的梯度

激活函数逐元素操作 $\mathbf{y} = f(\mathbf{x})$，因此梯度也是逐元素的：

$$ \frac{\partial \mathcal{L}}{\partial \mathbf{x}} = \frac{\partial \mathcal{L}}{\partial \mathbf{y}} \odot f'(\mathbf{x}) $$

其中 $\odot$ 是 Hadamard（逐元素）积，$f'$ 是激活函数的导数：

| 激活函数 | $f'(x)$ |
|---|---|
| ReLU | $\mathbb{1}\{x > 0\}$ |
| Sigmoid | $\sigma(x)(1 - \sigma(x))$ |
| Tanh | $1 - \tanh^2(x)$ |

---

## 代码实现：LinearLayer.backward()

`src/nn/layers/linearLayer.py:107-150`:

```python
def backward(self, outputGradient: np.ndarray) -> np.ndarray:
    # dL/dX = dL/dY @ W^T
    inputGradient = outputGradient @ self.weights.T

    # dL/dW = X^T @ dL/dY
    self.gradWeights[...] = self.inputCache.T @ outputGradient

    # dL/dBias = sum(dL/dY, axis=0, keepdims=True)
    if self.useBias and self.gradBias is not None:
        self.gradBias[...] = np.sum(outputGradient, axis=0, keepdims=True)

    return inputGradient
```

注意 `[...] =` 语法是**就地覆盖**（不是累加）。这意味着每次 backward 前必须先 `zeroGrad()` 清除旧的梯度。

---

## 全局反向传播流程

`src/nn/models/sequentialModel.py:91-112` 中的 `SequentialModel.backward()` 协调全局反向传播：

```python
def backward(self, outputGradient: np.ndarray) -> np.ndarray:
    inputGradient = outputGradient
    # 从最后一层向前，逐层反向传播
    for layer in reversed(self.layers):
        inputGradient = layer.backward(inputGradient)
    return inputGradient
```

**完整训练步骤中的梯度流**：

$$ \frac{\partial \mathcal{L}}{\partial \mathbf{z}^{(L)}} \xrightarrow{\text{Layer } L \text{ .backward}} \frac{\partial \mathcal{L}}{\partial \mathbf{a}^{(L-1)}} \xrightarrow{\text{Layer } L-1 \text{ .backward}} \cdots \xrightarrow{\text{Layer } 1 \text{ .backward}} \frac{\partial \mathcal{L}}{\partial \mathbf{X}} $$

每一步的 `layer.backward()` 同时：
1. **计算**并**存储**该层参数的梯度（`self.gradWeights`, `self.gradBias`）
2. **返回**对输入的梯度（供上一层继续传播）

---

## 具体数值示例

以下是一个极简的二层网络反向传播演示。

**设定**：
- 输入 $\mathbf{X} = \begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix}$（2 个样本，2 维特征）
- $\mathbf{W}^{(1)} = \begin{bmatrix} 0.5 & 0.1 \\ -0.2 & 0.3 \end{bmatrix}$，$\mathbf{b}^{(1)} = [0, 0]$
- $\mathbf{W}^{(2)} = \begin{bmatrix} 1.0 \\ 0.5 \end{bmatrix}$，$\mathbf{b}^{(2)} = [0]$

**前向传播**：
- $\mathbf{Z}^{(1)} = \mathbf{X}\mathbf{W}^{(1)} + \mathbf{b}^{(1)} = \begin{bmatrix} 1(0.5)+2(-0.2) & 1(0.1)+2(0.3) \\ 3(0.5)+4(-0.2) & 3(0.1)+4(0.3) \end{bmatrix} = \begin{bmatrix} 0.1 & 0.7 \\ 0.7 & 1.5 \end{bmatrix}$
- $\mathbf{a}^{(1)} = \text{ReLU}(\mathbf{Z}^{(1)}) = \begin{bmatrix} 0.1 & 0.7 \\ 0.7 & 1.5 \end{bmatrix}$
- $\mathbf{z}^{(2)} = \mathbf{a}^{(1)}\mathbf{W}^{(2)} + \mathbf{b}^{(2)} = \begin{bmatrix} 0.1(1.0)+0.7(0.5) \\ 0.7(1.0)+1.5(0.5) \end{bmatrix} = \begin{bmatrix} 0.45 \\ 1.45 \end{bmatrix}$

**假设损失梯度** $\frac{\partial \mathcal{L}}{\partial \mathbf{z}^{(2)}} = \begin{bmatrix} 0.1 \\ -0.2 \end{bmatrix}$（由损失函数返回）

**反向传播第 2 层**：
- $\frac{\partial \mathcal{L}}{\partial \mathbf{a}^{(1)}} = \frac{\partial \mathcal{L}}{\partial \mathbf{z}^{(2)}} \mathbf{W}^{(2)T} = \begin{bmatrix} 0.1 & (-0.2) \end{bmatrix} \begin{bmatrix} 1.0 \\ 0.5 \end{bmatrix}^T = \begin{bmatrix} 0.1 & 0.05 \\ -0.2 & -0.1 \end{bmatrix}$ ❌

等等 — 这里让我更正。$\frac{\partial \mathcal{L}}{\partial \mathbf{z}^{(2)}}$ 形状 $(2, 1)$，$\mathbf{W}^{(2)T}$ 形状 $(1, 2)$：

$$ \frac{\partial \mathcal{L}}{\partial \mathbf{a}^{(1)}} = \frac{\partial \mathcal{L}}{\partial \mathbf{z}^{(2)}}_{(2,1)} \cdot \mathbf{W}^{(2)T}_{(1,2)} = \begin{bmatrix} 0.1(1.0) & 0.1(0.5) \\ -0.2(1.0) & -0.2(0.5) \end{bmatrix} = \begin{bmatrix} 0.1 & 0.05 \\ -0.2 & -0.1 \end{bmatrix} $$

- $\frac{\partial \mathcal{L}}{\partial \mathbf{W}^{(2)}} = \mathbf{a}^{(1)T} \frac{\partial \mathcal{L}}{\partial \mathbf{z}^{(2)}} = \begin{bmatrix} 0.1 & 0.7 \\ 0.7 & 1.5 \end{bmatrix}^T \begin{bmatrix} 0.1 \\ -0.2 \end{bmatrix} = \begin{bmatrix} 0.01 + (-0.14) \\ 0.07 + (-0.30) \end{bmatrix} = \begin{bmatrix} -0.13 \\ -0.23 \end{bmatrix}$

**反向传播第 1 层（ReLU）**：
- ReLU 的导数掩码：$\text{ReLU}'(\mathbf{Z}^{(1)}) = \begin{bmatrix} 1 & 1 \\ 1 & 1 \end{bmatrix}$（所有值 > 0）
- $\frac{\partial \mathcal{L}}{\partial \mathbf{Z}^{(1)}} = \frac{\partial \mathcal{L}}{\partial \mathbf{a}^{(1)}} \odot \text{ReLU}'(\mathbf{Z}^{(1)}) = \begin{bmatrix} 0.1 & 0.05 \\ -0.2 & -0.1 \end{bmatrix}$

**反向传播第 1 层（Linear）**：
- $\frac{\partial \mathcal{L}}{\partial \mathbf{W}^{(1)}} = \mathbf{X}^T \frac{\partial \mathcal{L}}{\partial \mathbf{Z}^{(1)}} = \begin{bmatrix} 1 & 3 \\ 2 & 4 \end{bmatrix} \begin{bmatrix} 0.1 & 0.05 \\ -0.2 & -0.1 \end{bmatrix} = \begin{bmatrix} 0.1+( -0.6) & 0.05+(-0.3) \\ 0.2+(-0.8) & 0.1+(-0.4) \end{bmatrix} = \begin{bmatrix} -0.5 & -0.25 \\ -0.6 & -0.3 \end{bmatrix}$

这些梯度随后由优化器用于参数更新。
