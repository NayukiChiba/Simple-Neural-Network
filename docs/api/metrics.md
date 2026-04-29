# Metrics — 评估指标 API

`src/nn/training/metrics.py`

## calculateAccuracy()

```python
def calculateAccuracy(predictions: np.ndarray, targetData: np.ndarray) -> float
```

计算分类准确率。

| 参数 | 类型 | 形状 | 说明 |
|---|---|---|---|
| `predictions` | `np.ndarray` | $(N, C)$ | 模型预测输出（logits 或概率） |
| `targetData` | `np.ndarray` | $(N,)$ 或 $(N, C)$ | 一维类别索引或二维 one-hot 标签 |
| **返回** | `float` | — | 准确率，范围 $[0, 1]$ |

**数学公式**：

$$ \text{Accuracy} = \frac{1}{N} \sum_{n=1}^{N} \mathbb{1}\{\arg\max_c(\hat{y}_{n,c}) = y_n\} $$

**实现**：
1. 对 predictions 取 `np.argmax(predictions, axis=1)` 得到预测类别
2. 对 targetData：一维直接使用，二维取 `np.argmax(targetData, axis=1)`
3. 比较并取平均

**异常**：
- `ValueError` — `predictions` 不是二维数组
- `ValueError` — 样本数为 0 或类别数为 0
- `ValueError` — 预测样本数与标签数不一致

---

## calculateMeanSquaredError()

```python
def calculateMeanSquaredError(predictions: np.ndarray, targetData: np.ndarray) -> float
```

计算均方误差（作为评估指标，与 `MSELoss.forward()` 计算方式相同）。

| 参数 | 类型 | 形状 | 说明 |
|---|---|---|---|
| `predictions` | `np.ndarray` | 任意 | 模型预测值 |
| `targetData` | `np.ndarray` | 与 predictions 一致 | 真实值 |
| **返回** | `float` | — | 均方误差 |

**数学公式**：

$$ \text{MSE} = \frac{1}{M} \sum_{i=1}^{M} (\hat{y}_i - y_i)^2 $$

**异常**：
- `ValueError` — 存在标量输入
- `ValueError` — 形状不一致
- `ValueError` — 空张量

---

## convertLabelsToIndices()

```python
def convertLabelsToIndices(targetData: np.ndarray) -> np.ndarray
```

将标签转换为类别索引（一维）。

| 输入格式 | 处理 |
|---|---|
| `(N,)` 一维整数 | 直接返回 |
| `(N, C)` 二维 one-hot | `np.argmax(axis=1)` |
| 其他 | `ValueError` |

主要用于 `calculateAccuracy()` 内部，统一处理两种标签格式。
