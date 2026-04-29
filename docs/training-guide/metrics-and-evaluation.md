# 评估指标

## 分类准确率

### 定义

$$ \text{Accuracy} = \frac{\text{正确预测数}}{\text{总样本数}} = \frac{1}{N} \sum_{n=1}^{N} \mathbb{1}\{\hat{y}_n = y_n\} $$

其中 $\hat{y}_n = \arg\max_c \text{output}_{n,c}$ 是第 $n$ 个样本的预测类别。

### 标签格式兼容

`calculateAccuracy()` 通过 `convertLabelsToIndices()` 兼容两种标签格式：

| 格式 | 形状 | 示例 | 处理 |
|---|---|---|---|
| 整数索引 | $(N,)$ | `[0, 2, 1]` | 直接使用 |
| One-hot | $(N, C)$ | `[[1,0,0],[0,0,1],[0,1,0]]` | `np.argmax(axis=1)` |

### 实现

`src/nn/training/metrics.py:34-66`:

```python
def calculateAccuracy(predictions, targetData) -> float:
    # 预测类别：取每行最大值的索引
    predictedLabels = np.argmax(predictions, axis=1)

    # 真实类别：一维直接使用，二维取 argmax
    targetLabels = convertLabelsToIndices(targetData)

    # 比较并求平均
    accuracy = np.mean(predictedLabels == targetLabels)
    return float(accuracy)
```

---

## 回归均方误差

### 定义

$$ \text{MSE} = \frac{1}{M} \sum_{i=1}^{M} (\hat{y}_i - y_i)^2 $$

这与 `MSELoss.forward()` 的计算结果一致。在回归任务中，`evaluate()` 返回的 `"mse"` 值等于 `"loss"` 值。

### 实现

`src/nn/training/metrics.py:69-94`:

```python
def calculateMeanSquaredError(predictions, targetData) -> float:
    if predictions.shape != targetData.shape:
        raise ValueError("形状必须一致")
    mse = np.mean((predictions - targetData) ** 2)
    return float(mse)
```

---

## 评估流程

`Trainer.evaluate()` 在计算损失和指标时**临时切换到 eval 模式**：

```python
def evaluate(self, inputData, targetData) -> dict[str, float]:
    wasTraining = self.model.isTraining
    self.model.eval()
    try:
        predictions = self.model.forward(inputData)
        loss = self.lossFunction.forward(predictions, targetData)
    finally:
        if wasTraining:
            self.model.train()   # 恢复训练模式

    result = {"loss": float(loss)}

    if self.taskType == "classification":
        result["accuracy"] = self.computeMetric(predictions, targetData)
    else:
        result["mse"] = float(loss)

    return result
```

---

## 训练历史分析

`fit()` 返回的训练历史可用于检测：

| 现象 | 训练损失 | 验证损失 | 说明 |
|---|---|---|---|
| 正常训练 | 持续下降 | 持续下降 | 模型正在学习 |
| 过拟合 | 持续下降 | 停止下降或上升 | 模型记忆了训练数据，泛化能力开始退化 |
| 欠拟合 | 收敛但较高 | 收敛但较高 | 模型容量不足或训练不充分 |
| 发散 | 上升 | 上升 | 学习率可能太大 |

### 使用历史数据

```python
history = trainer.fit(xTrain, yTrain, epochCount=400, validInputs=xValid, validTargets=yValid)

# 检查最终指标
print(f"最终训练准确率: {history['train_accuracy'][-1]:.4f}")
print(f"最终验证准确率: {history['valid_accuracy'][-1]:.4f}")

# 找出最佳验证 epoch
bestEpoch = np.argmax(history['valid_accuracy'])
print(f"最佳验证 epoch: {bestEpoch + 1}")
```
