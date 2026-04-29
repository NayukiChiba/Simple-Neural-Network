# Training — 训练器 API

## Trainer

`src/nn/training/trainer.py`

模型训练器，负责编排前向传播、损失计算、反向传播、参数更新和评估的完整流程。

### 构造函数

```python
Trainer(
    model: SequentialModel,
    lossFunction: CrossEntropyLoss | MSELoss,
    optimizer: SGDOptimizer,
    taskType: Literal["classification", "regression"],
    batchSize: int = 32,
    shuffle: bool = True,
    randomSeed: int | None = None,
)
```

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `model` | `SequentialModel` | — | 顺序模型 |
| `lossFunction` | `CrossEntropyLoss \| MSELoss` | — | 损失函数 |
| `optimizer` | `SGDOptimizer` | — | 优化器 |
| `taskType` | `"classification" \| "regression"` | — | 任务类型 |
| `batchSize` | `int` | `32` | Mini-batch 大小 |
| `shuffle` | `bool` | `True` | 每个 epoch 是否打乱数据 |
| `randomSeed` | `int \| None` | `None` | 控制 shuffle 的随机种子 |

**异常**：
- `ValueError` — `batchSize <= 0`

---

### validateDataset()

```python
def validateDataset(self, inputData: np.ndarray, targetData: np.ndarray) -> None
```

验证数据集的合法性。检查：
- 两个参数都不是标量（`ndim == 0`）
- 样本数非零
- 样本数一致

---

### createBatches()

```python
def createBatches(
    self, inputData: np.ndarray, targetData: np.ndarray
) -> list[tuple[np.ndarray, np.ndarray]]
```

将数据划分为 mini-batch。若 `self.shuffle = True`，先打乱再分。

| 参数 | 类型 | 形状 | 说明 |
|---|---|---|---|
| `inputData` | `np.ndarray` | $(N, d_{in})$ | 输入数据 |
| `targetData` | `np.ndarray` | $(N, \ldots)$ | 目标数据 |
| **返回** | `list[tuple]` | — | `[(batchX, batchY), ...]` |

**说明**：最后一个 batch 的大小可能小于 `batchSize`。

---

### trainStep()

```python
def trainStep(self, inputBatch: np.ndarray, targetBatch: np.ndarray) -> float
```

执行**一个** mini-batch 的训练步骤。这是训练的最小原子操作。

**内部 7 步**：
1. `model.train()` — 训练模式
2. `optimizer.zeroGrad(model.layers)` — 清零梯度
3. `model.forward(inputBatch)` — 前向传播
4. `lossFunction.forward(predictions, targetBatch)` — 计算损失
5. `lossFunction.backward()` — 损失梯度
6. `model.backward(outputGradient)` — 反向传播
7. `optimizer.step(model.layers)` — 参数更新

| 返回 | 类型 | 说明 |
|---|---|---|
| `loss` | `float` | 当前 batch 的平均损失 |

---

### trainEpoch()

```python
def trainEpoch(self, inputData: np.ndarray, targetData: np.ndarray) -> float
```

执行一个完整的训练 epoch（遍历所有 batch）。

| 返回 | 类型 | 说明 |
|---|---|---|
| `avgLoss` | `float` | 加权平均损失 $\frac{\sum |B_b| \cdot \mathcal{L}_b}{N}$ |

---

### evaluate()

```python
def evaluate(
    self, inputData: np.ndarray, targetData: np.ndarray
) -> dict[str, float]
```

在评估模式下对整个数据集进行评估。

**返回格式**：

```python
# 分类任务
{"loss": float, "accuracy": float}

# 回归任务
{"loss": float, "mse": float}
```

**说明**：自动临时切换到 eval 模式，完成后恢复原模式。

---

### fit()

```python
def fit(
    self,
    trainInputs: np.ndarray,
    trainTargets: np.ndarray,
    epochCount: int,
    validInputs: np.ndarray | None = None,
    validTargets: np.ndarray | None = None,
    verbose: bool = True,
) -> dict[str, list[float]]
```

完整训练循环。

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `trainInputs` | `np.ndarray` | — | 训练输入，形状 $(N, d_{in})$ |
| `trainTargets` | `np.ndarray` | — | 训练目标 |
| `epochCount` | `int` | — | 训练 epoch 数，必须 > 0 |
| `validInputs` | `np.ndarray \| None` | `None` | 可选验证输入 |
| `validTargets` | `np.ndarray \| None` | `None` | 可选验证目标 |
| `verbose` | `bool` | `True` | 是否每个 epoch 打印日志 |
| **返回** | `dict[str, list[float]]` | — | 训练历史 |

**返回格式（分类 + 验证）**：

```python
{
    "train_loss": [0.693, 0.521, ...],       # epochCount 个值
    "train_accuracy": [0.35, 0.67, ...],
    "valid_loss": [0.688, 0.512, ...],
    "valid_accuracy": [0.37, 0.69, ...],
}
```

**异常**：
- `ValueError` — `epochCount <= 0`
- `ValueError` — `validInputs` 和 `validTargets` 只给了一个

---

### predict()

```python
def predict(self, inputData: np.ndarray) -> np.ndarray
```

对输入数据进行预测（委托给 `self.model.predict()`）。
