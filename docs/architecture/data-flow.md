# 训练数据流

本文追踪一次完整的训练步骤（单个 mini-batch）中的数据流动，包括维度变化和梯度传递。

## 总体流程

```text
    ┌─────────────────────────────────────────────────────────┐
    │                    Trainer.trainStep()                   │
    │                                                         │
    │  ① model.train()             设置训练模式               │
    │  ② optimizer.zeroGrad()      清除旧梯度                │
    │                                                         │
    │         ③ model.forward(inputBatch)                     │
    │         ┌─────────────────────────────────────┐         │
    │         │  Input → Linear → ReLU → Linear →  │         │
    │         │  cache   cache   cache   cache     │         │
    │         └─────────────────┬───────────────────┘         │
    │                           │ predictions                 │
    │  ┌────────────────────────▼──────────────────────┐      │
    │  │  ④ lossFunction.forward(predictions, targets) │      │
    │  │     → loss (float)                            │      │
    │  │  ⑤ outputGradient = lossFunction.backward()   │      │
    │  └────────────────────────┬──────────────────────┘      │
    │                           │ outputGradient               │
    │         ⑥ model.backward(outputGradient)                │
    │         ┌─────────────────────────────────────┐         │
    │         │  Linear ← ReLU ← Linear ← Grad      │         │
    │         │  dW,db    mask   dW,db              │         │
    │         └─────────────────────────────────────┘         │
    │                                                         │
    │  ⑦ optimizer.step(model.layers)  参数更新               │
    └─────────────────────────────────────────────────────────┘
```

## 维度追踪示例

以 Spiral 分类任务为例。模型结构：

$$ \text{Input}(32, 2) \rightarrow \text{Linear}^{(1)}(2 \rightarrow 64) \rightarrow \text{ReLU} \rightarrow \text{Linear}^{(2)}(64 \rightarrow 3) $$

Batch size = 32。

### 前向传播

```
步骤                        维度变化
────────────────────────────────────────
输入 X                      (32, 2)
    ↓ Linear(1).forward
  Z1 = X @ W1 + b1         (32, 2) @ (2, 64) + (1, 64)
                            = (32, 64)
    ↓ ReLU.forward
  A1 = max(0, Z1)          (32, 64)
    ↓ Linear(2).forward
  Z2 = A1 @ W2 + b2        (32, 64) @ (64, 3) + (1, 3)
                            = (32, 3)    ← logits（3类）
```

### 损失计算

```
步骤                             维度
────────────────────────────────────────
CrossEntropyLoss.forward(Z2, targets)
  shifted = Z2 - max(Z2)         (32, 3)
  probs = softmax(shifted)       (32, 3)    每个样本 3 个概率
  loss = -mean(log(probs[true])) scalar
```

### 反向传播

```
步骤                               维度
────────────────────────────────────────
CrossEntropyLoss.backward()
  gradient = (probs - onehot) / 32  (32, 3)    ← ∂L/∂Z2

SequentialModel.backward(gradient)
  ↓ Linear(2).backward(gradient)
    inputGrad = gradient @ W2.T    (32, 3) @ (3, 64) = (32, 64)
    gradW2 = A1.T @ gradient       (64, 32) @ (32, 3) = (64, 3)
    gradB2 = sum(gradient, 0)     (1, 3)

  ↓ ReLU.backward(inputGrad)
    mask = Z1 > 0                  (32, 64)
    grad = inputGrad * mask        (32, 64)

  ↓ Linear(1).backward(grad)
    inputGrad = grad @ W1.T        (32, 64) @ (64, 2) = (32, 2)
    gradW1 = X.T @ grad            (2, 32) @ (32, 64) = (2, 64)
    gradB1 = sum(grad, 0)          (1, 64)
```

### 参数更新

```
SGD(lr=0.03).step([layer1, layer2])
  W1 -= 0.03 * gradW1             (2, 64)
  b1 -= 0.03 * gradB1             (1, 64)
  W2 -= 0.03 * gradW2             (64, 3)
  b2 -= 0.03 * gradB2             (1, 3)
```

---

## 批量加权平均

在 `trainEpoch()` 中，多个 batch 的损失需要聚合为 epoch 级别的平均损失。不能直接取各 batch 损失的平均，因为最后一个 batch 的大小可能小于 `batchSize`。

$$ \mathcal{L}_{\text{epoch}} = \frac{\sum_{b=1}^{B} |B_b| \cdot \mathcal{L}_b}{\sum_{b=1}^{B} |B_b|} $$

代码实现（`src/nn/training/trainer.py:164-182`）：

```python
def trainEpoch(self, inputData, targetData) -> float:
    batches = self.createBatches(inputData, targetData)
    totalLoss = 0.0

    for inputBatch, targetBatch in batches:
        batchLoss = self.trainStep(inputBatch, targetBatch)
        totalLoss += batchLoss * inputBatch.shape[0]  # ← 加权

    averageLoss = totalLoss / inputData.shape[0]       # ← 除以总样本数
    return averageLoss
```

---

## 训练与评估模式切换

`Trainer.evaluate()` 在执行评估时临时切换到 eval 模式：

```python
def evaluate(self, inputData, targetData):
    wasTraining = self.model.isTraining
    self.model.eval()
    try:
        predictions = self.model.forward(inputData)
        loss = self.lossFunction.forward(predictions, targetData)
    finally:
        if wasTraining:
            self.model.train()     # ← 恢复原模式
    return {"loss": float(loss), ...}
```

`finally` 块确保即使评估过程出错，模型模式也能正确恢复。

---

## 数据流中的缓存依赖关系

```
forward 路径:                   backward 路径:
X ──→ Linear(1).inputCache      Linear(1).gradWeights ←── inputCache
   │                                    ↑
   └─→ ReLU.inputCache           ReLU.backward ←── inputCache (>0?)
           │                            ↑
           └─→ Linear(2).inputCache      Linear(2).gradWeights ←── inputCache
```

每条缓存线在 forward 时写入，在 backward 中读取。若 `backward()` 在 `forward()` 之前被调用，会触发 `ValueError("调用 backward 之前必须先调用 forward")`。
