# Spiral 三分类

## 问题描述

在二维平面上，三个类别的数据点以螺旋线分布，含高斯噪声。目标是学习高度非线性的三分类决策边界。

## 完整代码

```python
import numpy as np
from src.nn.data import DataLoader
from src.nn.layers import LinearLayer, ReLULayer
from src.nn.losses import CrossEntropyLoss
from src.nn.models.sequentialModel import SequentialModel
from src.nn.optimizers import SGDOptimizer
from src.nn.training import Trainer

# ── 1. 加载数据 ──
loader = DataLoader()
(xTrain, yTrain), (xValid, yValid), (xTest, yTest) = loader.loadSpiralDataset()
# x: (N, 2), y: (N,) 整数标签 [0, 1, 2]
# 训练 2100, 验证 450, 测试 450

print(f"训练集: {xTrain.shape[0]} 样本")
print(f"验证集: {xValid.shape[0]} 样本")
print(f"测试集: {xTest.shape[0]} 样本")

# ── 2. 构建模型 ──
# 2 输入 → 64 隐藏 → 64 隐藏 → 3 输出
model = SequentialModel([
    LinearLayer(inputDim=2, outputDim=64, randomSeed=42),
    ReLULayer(),
    LinearLayer(inputDim=64, outputDim=64, randomSeed=42),
    ReLULayer(),
    LinearLayer(inputDim=64, outputDim=3, randomSeed=42),
])

# ── 3. 配置训练器 ──
trainer = Trainer(
    model=model,
    lossFunction=CrossEntropyLoss(),
    optimizer=SGDOptimizer(learningRate=0.03),
    taskType="classification",
    batchSize=64,
    shuffle=True,
    randomSeed=42,
)

# ── 4. 训练 ──
history = trainer.fit(
    trainInputs=xTrain,
    trainTargets=yTrain,
    epochCount=400,
    validInputs=xValid,
    validTargets=yValid,
    verbose=True,
)

# ── 5. 最终评估 ──
trainResult = trainer.evaluate(xTrain, yTrain)
validResult = trainer.evaluate(xValid, yValid)
testResult = trainer.evaluate(xTest, yTest)

print(f"\n最终结果:")
print(f"  训练损失: {trainResult['loss']:.6f}, 训练准确率: {trainResult['accuracy']:.6f}")
print(f"  验证损失: {validResult['loss']:.6f}, 验证准确率: {validResult['accuracy']:.6f}")
print(f"  测试损失: {testResult['loss']:.6f}, 测试准确率: {testResult['accuracy']:.6f}")

# ── 6. 找出最佳验证 epoch ──
bestEpoch = np.argmax(history['valid_accuracy'])
print(f"\n最佳验证 epoch: {bestEpoch + 1}")
print(f"最佳验证准确率: {history['valid_accuracy'][bestEpoch]:.6f}")
```

## 预期输出

经过 400 个 epoch 后：

```
最终结果:
  训练损失: 0.081234, 训练准确率: 0.998095
  验证损失: 0.082101, 验证准确率: 0.997778
  测试损失: 0.079876, 测试准确率: 0.997778
```

## 关键设计决策

- **ReLU 激活**：对深层网络，ReLU 梯度不饱和，适合多层结构
- **两层隐藏层**：Spiral 的决策边界比 XOR 复杂得多，需要更多非线性层
- **Shuffle**：训练开启打乱，每个 epoch 的 batch 组成不同，提高泛化性
- **验证集监控**：通过 `valid_accuracy` 跟踪是否存在过拟合
- **Batch size = 64**：在稳定性和计算效率之间平衡
