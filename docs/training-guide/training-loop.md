# 训练循环详解

## 训练层次结构

训练过程有三个层次，形成嵌套调用：

```text
Trainer.fit()          ← 最外层：epoch 循环
  ├── for epoch in range(epochCount):
  │   ├── Trainer.trainEpoch()   ← 中层：遍历所有 batch
  │   │   ├── for batch in batches:
  │   │   │   └── Trainer.trainStep()  ← 最内层：单个 batch 训练
  │   ├── Trainer.evaluate(trainSet)   ← 训练集评估
  │   └── Trainer.evaluate(validSet)   ← 验证集评估（可选）
```

## trainStep() — 原子训练操作

这是最小训练单元，对单个 mini-batch 执行完整的计算图。

`src/nn/training/trainer.py:122-151`:

```python
def trainStep(self, inputBatch: np.ndarray, targetBatch: np.ndarray) -> float:
    # 1. 训练模式
    self.model.train()

    # 2. 清零梯度（防止跨 batch 累积）
    self.optimizer.zeroGrad(self.model.layers)

    # 3. 前向传播
    predictions = self.model.forward(inputBatch)

    # 4. 计算损失
    loss = self.lossFunction.forward(predictions, targetBatch)

    # 5. 损失对输出求梯度
    outputGradient = self.lossFunction.backward()

    # 6. 模型反向传播（逐层求参数梯度）
    self.model.backward(outputGradient)

    # 7. 参数更新
    self.optimizer.step(self.model.layers)

    return float(loss)
```

**为什么需要 zeroGrad？**

`LinearLayer.backward()` 使用 `self.gradWeights[...] =` **覆盖**写入（而非累加 `+=`）。如果不先清零，新值会覆盖旧值，但如果某些层没有收到新的 backward（比如跳过），旧梯度残留就可能污染后续的参数更新。

即使使用 `+=` 累加，每个新的 batch 也需要从零开始独立计算梯度。因此 `zeroGrad()` 是训练的标准步骤。

## trainEpoch() — Epoch 级聚合

`src/nn/training/trainer.py:153-182`:

```python
def trainEpoch(self, inputData, targetData) -> float:
    batches = self.createBatches(inputData, targetData)
    totalLoss = 0.0

    for inputBatch, targetBatch in batches:
        batchLoss = self.trainStep(inputBatch, targetBatch)
        totalLoss += batchLoss * inputBatch.shape[0]  # 加权累加

    averageLoss = totalLoss / inputData.shape[0]       # 总平均
    return averageLoss
```

### 加权平均公式

`trainStep()` 返回的是 **batch 内部的平均损失**。但不同 batch 可能大小不同（最后一个 batch 通常取不满 `batchSize`），因此需要加权平均：

$$ \mathcal{L}_{\text{epoch}} = \frac{\sum_{b=1}^{B} |B_b| \cdot \mathcal{L}_b}{\sum_{b=1}^{B} |B_b|} = \frac{\sum_{b=1}^{B} |B_b| \cdot \mathcal{L}_b}{N_{\text{total}}} $$

**不加权的问题**：如果直接对 batch 损失求平均，小 batch 的损失权重会与大 batch 相同，导致 epoch 损失并非真实的全体本平均损失。

## fit() — 完整训练循环

`src/nn/training/trainer.py:248-364`:

```python
def fit(
    self,
    trainInputs: np.ndarray,
    trainTargets: np.ndarray,
    epochCount: int,
    validInputs: np.ndarray | None = None,
    validTargets: np.ndarray | None = None,
    verbose: bool = True,
) -> dict[str, list[float]]:
```

### 历史记录结构

返回的 `history` 字典格式：

```python
# 分类 + 验证
{
    "train_loss": [0.693, 0.521, ...],
    "train_accuracy": [0.35, 0.67, ...],
    "valid_loss": [0.688, 0.512, ...],
    "valid_accuracy": [0.37, 0.69, ...],
}

# 回归（无验证）
{
    "train_loss": [0.523, 0.312, ...],
    "train_mse": [0.523, 0.312, ...],
}
```

### verbose 输出格式

当 `verbose=True` 时，每个 epoch 打印一行：

```
epoch 1/400 - train_loss: 0.693147 - train_accuracy: 0.350000 - valid_loss: 0.688123 - valid_accuracy: 0.370000
epoch 2/400 - train_loss: 0.521234 - train_accuracy: 0.675000 - valid_loss: 0.512345 - valid_accuracy: 0.690000
...
```

## 数据打乱与 Mini-batch 创建

`src/nn/training/trainer.py:86-120`:

```python
def createBatches(self, inputData, targetData) -> list[tuple]:
    sampleCount = inputData.shape[0]
    indices = np.arange(sampleCount)

    if self.shuffle:
        indices = self.rng.permutation(indices)  # 随机排列

    shuffledInputs = inputData[indices]
    shuffledTargets = targetData[indices]

    batches = []
    for startIndex in range(0, sampleCount, self.batchSize):
        endIndex = startIndex + self.batchSize
        batchInputs = shuffledInputs[startIndex:endIndex]
        batchTargets = shuffledTargets[startIndex:endIndex]
        batches.append((batchInputs, batchTargets))

    return batches
```

**关键细节**：
- 使用 `self.rng`（独立的 `np.random.Generator` 实例），不影响全局随机状态
- `shuffle=False` 时保持原始顺序（用于确定性测试或全批量训练）
- `self.batchSize=4` 时 XOR 的 4 个样本恰好形成一个 batch
