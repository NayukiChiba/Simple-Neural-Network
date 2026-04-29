# Losses — 损失函数 API

## MSELoss

`src/nn/losses/mseLoss.py`

均方误差损失，用于回归任务。

### 构造函数

```python
MSELoss()
```

无参数。

### forward()

```python
def forward(self, predictions: np.ndarray, targets: np.ndarray) -> float
```

| 参数 | 类型 | 形状 | 说明 |
|---|---|---|---|
| `predictions` | `np.ndarray` | 任意 | 模型预测值 |
| `targets` | `np.ndarray` | 与 predictions 一致 | 真实目标值 |
| **返回** | `float` | — | 平均均方误差 |

**数学公式**：

$$ \mathcal{L} = \frac{1}{M} \sum_{i=1}^{M} (\hat{y}_i - y_i)^2 $$

其中 $M$ 是张量的总元素数。

**异常**：
- `ValueError` — `predictions` 或 `targets` 是标量
- `ValueError` — 两者形状不一致
- `ValueError` — 张量为空

### backward()

```python
def backward(self) -> np.ndarray
```

| 返回 | 类型 | 形状 | 说明 |
|---|---|---|---|
| `gradient` | `np.ndarray` | 与 predictions 一致 | $\frac{\partial \mathcal{L}}{\partial \hat{\mathbf{y}}}$ |

**数学公式**：

$$ \frac{\partial \mathcal{L}}{\partial \hat{\mathbf{y}}} = \frac{2}{M}(\hat{\mathbf{y}} - \mathbf{y}) $$

**前置条件**：必须先调用 `forward()`。

---

## CrossEntropyLoss

`src/nn/losses/crossEntropyLoss.py`

交叉熵损失，内置 Softmax，用于分类任务。

### 构造函数

```python
CrossEntropyLoss(epsilon: float = 1e-12)
```

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `epsilon` | `float` | `1e-12` | 防止 $\log(0)$ 的裁剪下界 |

### forward()

```python
def forward(self, logits: np.ndarray, targetLabels: np.ndarray) -> float
```

| 参数 | 类型 | 形状 | 说明 |
|---|---|---|---|
| `logits` | `np.ndarray` | $(N, C)$ | 模型输出的原始 logits（未经 Softmax） |
| `targetLabels` | `np.ndarray` | $(N,)$ | 真实类别索引（整数，范围 $[0, C)$） |
| **返回** | `float` | — | 平均交叉熵损失 |

**内部计算步骤**：
1. 数值稳定：`shifted = logits - max(logits, axis=1)`
2. Softmax：$p_i = e^{\text{shifted}_i} / \sum_j e^{\text{shifted}_j}$
3. 取真实类别概率：`p_true = p[np.arange(N), targetLabels]`
4. 裁剪到 `[epsilon, 1.0]`
5. 交叉熵：$-\frac{1}{N} \sum \log(p_{\text{true}})$

**异常**：
- `ValueError` — `logits` 不是二维数组
- `ValueError` — `targetLabels` 不是一维数组
- `ValueError` — batch size 为 0
- `ValueError` — `targetLabels` 不是整数类型
- `ValueError` — 存在超出 $[0, C)$ 范围的标签

### backward()

```python
def backward(self) -> np.ndarray
```

| 返回 | 类型 | 形状 | 说明 |
|---|---|---|---|
| `gradient` | `np.ndarray` | $(N, C)$ | $\frac{\partial \mathcal{L}}{\partial \mathbf{z}}$ |

**数学公式**：

$$ \frac{\partial \mathcal{L}}{\partial z_{n,c}} = \frac{1}{N}\left(p_{n,c} - \delta_{c, y_n}\right) $$

其中 $p_{n,c}$ 是 softmax 概率，$\delta_{c, y_n}$ 仅在真实类别位置为 1。

**实现**：

```python
inputGradient = self.probabilities.copy()
inputGradient[np.arange(batchSize), targetLabels] -= 1.0
inputGradient /= batchSize
```

---

## 对比

| 特性 | MSELoss | CrossEntropyLoss |
|---|---|---|
| 任务类型 | 回归 | 分类 |
| 输入要求 | 与目标同形状 | 二维 logits |
| 目标格式 | 连续值 | 整数类别索引 |
| 内部是否有 softmax | 否 | 是 |
| 数值稳定技巧 | 无 | logit-shift + epsilon clip |
| 梯度形状 | 与输入一致 | $(N, C)$ |
