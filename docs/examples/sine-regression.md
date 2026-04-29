# Sine 回归

## 问题描述

从带噪声的正弦函数采样数据中学习映射 $x \mapsto \sin(x)$。这是标准的一维回归任务，验证网络能否拟合光滑的非线性函数。

## 完整代码

```python
import numpy as np
from src.nn.data import DataLoader
from src.nn.layers import LinearLayer, TanhLayer
from src.nn.losses import MSELoss
from src.nn.models.sequentialModel import SequentialModel
from src.nn.optimizers import SGDOptimizer
from src.nn.training import Trainer

# ── 1. 加载数据 ──
loader = DataLoader()
(xTrain, yTrain), (xValid, yValid), (xTest, yTest) = loader.loadSineDataset()
# x: (N, 1), y: (N, 1)
# 训练 420, 验证 90, 测试 90

print(f"训练集: {xTrain.shape[0]} 样本")
print(f"验证集: {xValid.shape[0]} 样本")
print(f"测试集: {xTest.shape[0]} 样本")

# ── 2. 构建模型 ──
# 1 输入 → 16 隐藏 → 16 隐藏 → 1 输出
model = SequentialModel([
    LinearLayer(inputDim=1, outputDim=16, randomSeed=42),
    TanhLayer(),
    LinearLayer(inputDim=16, outputDim=16, randomSeed=42),
    TanhLayer(),
    LinearLayer(inputDim=16, outputDim=1, randomSeed=42),
])

# ── 3. 配置训练器 ──
trainer = Trainer(
    model=model,
    lossFunction=MSELoss(),
    optimizer=SGDOptimizer(learningRate=0.01),
    taskType="regression",
    batchSize=32,
    shuffle=True,
    randomSeed=42,
)

# ── 4. 训练 ──
history = trainer.fit(
    trainInputs=xTrain,
    trainTargets=yTrain,
    epochCount=300,
    validInputs=xValid,
    validTargets=yValid,
    verbose=True,
)

# ── 5. 最终评估 ──
trainResult = trainer.evaluate(xTrain, yTrain)
validResult = trainer.evaluate(xValid, yValid)
testResult = trainer.evaluate(xTest, yTest)

print(f"\n最终结果 (MSE):")
print(f"  训练: {trainResult['loss']:.6f}")
print(f"  验证: {validResult['loss']:.6f}")
print(f"  测试: {testResult['loss']:.6f}")

# ── 6. 在测试集上对比真实值 ──
predictions = trainer.predict(xTest)
# 取前 5 个样本对比
print(f"\n前 5 个测试样本对比:")
for i in range(5):
    print(f"  x={xTest[i,0]:.4f}, 预测={predictions[i,0]:.4f}, 真实={yTest[i,0]:.4f}")
```

## 预期输出

经过 300 个 epoch 后，测试 MSE 应降低到合理水平（如 0.01~0.05）。$\epsilon \sim \mathcal{N}(0, 0.1)$ 的噪声设定了可达到的 MSE 下限约为 0.01。

```
最终结果 (MSE):
  训练: 0.009234
  验证: 0.009876
  测试: 0.010123
```

## 关键设计决策

- **MSELoss**：回归任务使用均方误差
- **Tanh 激活**：正弦函数输出范围 $[-1, 1]$，与 Tanh 的输出范围一致，更自然
- **小学习率（0.01）**：回归对学习率更敏感，过大容易震荡
- **最后一层无激活**：回归输出需要连续的标量值，不应被激活函数压缩

## 与分类任务的区别

| 特性 | 分类（Spiral） | 回归（Sine） |
|---|---|---|
| 损失函数 | `CrossEntropyLoss` | `MSELoss` |
| 最后一层 | 输出 logits | 输出标量 |
| 评估指标 | `accuracy` | `mse` |
| 标签格式 | 整数索引 | 连续值 |
| 输出维度 | 类别数 | 1 |
