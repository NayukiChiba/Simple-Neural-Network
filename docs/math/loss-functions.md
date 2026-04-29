# 损失函数

损失函数衡量模型预测值与真实值之间的差距。训练的目标是最小化损失函数。

本项目实现了两个损失函数：
- **MSELoss**（均方误差）— 用于回归任务
- **CrossEntropyLoss**（交叉熵，内置 Softmax）— 用于分类任务

---

## 均方误差 (MSE)

### 数学定义

给定 $N$ 个样本，每个样本的预测值 $\hat{y}_i$ 和真实值 $y_i$：

$$ \mathcal{L}_{\text{MSE}} = \frac{1}{N} \sum_{i=1}^{N} (\hat{y}_i - y_i)^2 $$

更一般地，当预测和目标为多维张量时，$\mathcal{L}_{\text{MSE}}$ 对所有元素取平均：

$$ \mathcal{L}_{\text{MSE}} = \frac{1}{M} \sum_{i=1}^{M} (\hat{y}_i - y_i)^2 $$

其中 $M$ 是张量的总元素数。

### 梯度推导

对单个预测值 $\hat{y}_k$ 求偏导：

$$ \frac{\partial \mathcal{L}}{\partial \hat{y}_k} = \frac{\partial}{\partial \hat{y}_k} \left[\frac{1}{M} \sum_{i=1}^{M} (\hat{y}_i - y_i)^2\right] $$

只有 $i = k$ 的项参与求导：

$$ \frac{\partial \mathcal{L}}{\partial \hat{y}_k} = \frac{1}{M} \cdot 2(\hat{y}_k - y_k) = \frac{2(\hat{y}_k - y_k)}{M} $$

写成向量形式：

$$ \frac{\partial \mathcal{L}}{\partial \hat{\mathbf{y}}} = \frac{2}{M}(\hat{\mathbf{y}} - \mathbf{y}) $$

### 代码实现

`src/nn/losses/mseLoss.py`:

```python
def forward(self, predictions: np.ndarray, targets: np.ndarray) -> float:
    difference = predictions - targets
    loss = np.mean(difference**2)
    # 缓存以供 backward 使用
    self.predictions = predictions
    self.targets = targets
    self.elementCount = predictions.size
    return float(loss)

def backward(self) -> np.ndarray:
    # dL/dPred = 2 * (predictions - targets) / N
    inputGradient = 2 * (self.predictions - self.targets) / self.elementCount
    return inputGradient
```

### 使用场景

- **回归任务**：预测连续值（如 Sine 回归）
- 要求 `predictions` 和 `targets` 形状完全一致
- 当训练器中 `taskType="regression"` 时使用

---

## 交叉熵损失（Softmax + CrossEntropy）

### 为什么需要 Softmax

对于 $C$ 类分类问题，模型最后一层输出 $C$ 个 logits（未经归一化的实数）。Softmax 将 logits 转换为概率分布：

$$ \text{softmax}(z_i) = \frac{e^{z_i}}{\sum_{j=1}^{C} e^{z_j}} $$

性质：
- $\sum_{j=1}^{C} \text{softmax}(z_j) = 1$（总和为 1）
- $\text{softmax}(z_j) > 0$（始终为正）
- 保持相对大小关系

### 交叉熵损失定义

$$ \mathcal{L}_{\text{CE}} = -\frac{1}{N} \sum_{n=1}^{N} \log \left(p_{n, y_n}\right) $$

其中 $p_{n, y_n}$ 是第 $n$ 个样本在真实类别 $y_n$ 上的 softmax 概率。

**直觉**：若模型对真实类别赋予了高概率（$p_{n, y_n} \approx 1$），则 $\log(1) = 0$，损失趋近于 0；若赋予了低概率（$p_{n, y_n} \approx 0$），则 $\log(0) \to -\infty$，损失很大。

### 数值稳定性技巧

直接计算 $e^{z_i}$ 可能因 $z_i$ 过大导致溢出。一个关键技巧是**减去每行的最大值**——这不会改变 softmax 结果：

$$ \text{softmax}(z_i) = \frac{e^{z_i - \max(\mathbf{z})}}{\sum_{j=1}^{C} e^{z_j - \max(\mathbf{z})}} $$

**证明**：分子分母同时乘以 $e^{-\max(\mathbf{z})}$ 不改变比值。
**效果**：平移后的最大 logit 为 0，$e^0 = 1$，所有指数值 $\leq 1$，避免数值溢出。

### 最优雅的梯度

Softmax + 交叉熵的组合有一个令人惊讶的简洁梯度：

$$ \frac{\partial \mathcal{L}}{\partial z_{n, c}} = \frac{1}{N}\left(p_{n, c} - \delta_{c, y_n}\right) $$

其中 $\delta_{c, y_n}$ 是 Kronecker delta（当 $c = y_n$ 时取 1，否则取 0）。

**推导概要**：

$$ \begin{aligned} \mathcal{L} &= -\frac{1}{N} \sum_n \log \left( \frac{e^{z_{n, y_n}}}{\sum_j e^{z_{n, j}}} \right) \\ &= -\frac{1}{N} \sum_n \left[ z_{n, y_n} - \log\left(\sum_j e^{z_{n, j}}\right) \right] \end{aligned} $$

对 $z_{n, c}$ 求偏导（分情况 $c = y_n$ 和 $c \neq y_n$）：

$$ \frac{\partial \mathcal{L}}{\partial z_{n, c}} = -\frac{1}{N} \left[ \delta_{c, y_n} - \frac{e^{z_{n, c}}}{\sum_j e^{z_{n, j}}} \right] = \frac{1}{N}\left(p_{n, c} - \delta_{c, y_n}\right) $$

### 代码实现

`src/nn/losses/crossEntropyLoss.py`:

```python
def forward(self, logits: np.ndarray, targetLabels: np.ndarray) -> float:
    # 数值稳定：减去每行最大值
    shiftedLogits = logits - np.max(logits, axis=1, keepdims=True)

    # Softmax
    expLogits = np.exp(shiftedLogits)
    probabilities = expLogits / np.sum(expLogits, axis=1, keepdims=True)

    # 取真实类别的概率
    selectedProbs = probabilities[np.arange(batchSize), targetLabels]

    # 裁剪避免 log(0)
    clippedProbs = np.clip(selectedProbs, self.epsilon, 1.0)

    # 交叉熵损失
    loss = -np.mean(np.log(clippedProbs))

    # 缓存 softmax 概率用于 backward
    self.probabilities = probabilities
    self.targetLabels = targetLabels
    self.batchSize = batchSize
    return float(loss)

def backward(self) -> np.ndarray:
    # dL/dz = (p - one_hot(y)) / N
    inputGradient = self.probabilities.copy()
    inputGradient[np.arange(self.batchSize), self.targetLabels] -= 1.0
    inputGradient /= self.batchSize
    return inputGradient
```

关键实现细节：
- `shiftedLogits` 是数值稳定技巧的核心
- `probabilities` 缓存了整个 softmax 输出矩阵，供 `backward` 使用
- `backward` 中的 `-= 1.0` 实现了 $p - \delta$，其中 `np.arange(batchSize)` 索引每个样本的真实类别位置
- `epsilon` 参数（默认 `1e-12`）防止 $\log(0)$

### 使用场景

- **分类任务**（二分类或多分类）
- `targetLabels` 必须是整数索引（不是 one-hot）
- 当训练器中 `taskType="classification"` 时使用

---

## 损失函数对比

| 特性 | MSELoss | CrossEntropyLoss |
|---|---|---|
| 任务类型 | 回归 | 分类 |
| 输入 | 预测值（任意形状） | logits（2D） |
| 目标 | 真实值（同形状） | 整数类别索引（1D） |
| 数值技巧 | 无 | logit-shift + epsilon clipping |
| 梯度与输入同形状 | 是 | 是 |
