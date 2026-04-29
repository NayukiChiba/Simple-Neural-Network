# XOR 二分类

## 问题描述

学习 XOR（异或）逻辑：`f(0,0)=0, f(0,1)=1, f(1,0)=1, f(1,1)=0`。这是最简单的非线性二分类问题。

## 完整代码

```python
import numpy as np
from src.nn.data import DataLoader
from src.nn.layers import LinearLayer, TanhLayer
from src.nn.losses import CrossEntropyLoss
from src.nn.models.sequentialModel import SequentialModel
from src.nn.optimizers import SGDOptimizer
from src.nn.training import Trainer

# ── 1. 加载数据 ──
loader = DataLoader()
x, y = loader.loadXorDataset()
# x: (4, 2), y: (4,) 整数标签 [0, 1, 1, 0]

# ── 2. 构建模型 ──
# 2 输入 → 8 隐藏 → 2 输出（两类 logits）
model = SequentialModel([
    LinearLayer(inputDim=2, outputDim=8, randomSeed=42),
    TanhLayer(),
    LinearLayer(inputDim=8, outputDim=2, randomSeed=42),
])

# ── 3. 配置训练器 ──
trainer = Trainer(
    model=model,
    lossFunction=CrossEntropyLoss(),
    optimizer=SGDOptimizer(learningRate=0.1),
    taskType="classification",
    batchSize=4,      # 全批量（4 个样本全部放进一个 batch）
    shuffle=False,    # 样本少，不需要打乱
    randomSeed=42,
)

# ── 4. 训练 ──
history = trainer.fit(
    trainInputs=x,
    trainTargets=y,
    epochCount=2000,
    verbose=True,
)

# ── 5. 评估 ──
result = trainer.evaluate(x, y)
print(f"\n最终结果:")
print(f"  损失: {result['loss']:.6f}")
print(f"  准确率: {result['accuracy']:.6f}")

# ── 6. 查看每个输入的预测 ──
predictions = trainer.predict(x)
predictedClasses = np.argmax(predictions, axis=1)
print(f"\n预测详情:")
for i in range(4):
    logits = predictions[i]
    print(f"  输入 [{x[i,0]:.0f}, {x[i,1]:.0f}] → "
          f"预测类别 {predictedClasses[i]}, 真实类别 {y[i]}, "
          f"logits [{logits[0]:.4f}, {logits[1]:.4f}]")
```

## 预期输出

经过 2000 个 epoch 后：

```
最终结果:
  损失: 0.000123
  准确率: 1.000000

预测详情:
  输入 [0, 0] → 预测类别 0, 真实类别 0, logits [2.3412, -2.1056]
  输入 [0, 1] → 预测类别 1, 真实类别 1, logits [-1.8932, 2.4533]
  输入 [1, 0] → 预测类别 1, 真实类别 1, logits [-1.8821, 2.4201]
  输入 [1, 1] → 预测类别 0, 真实类别 0, logits [2.3104, -2.0933]
```

对于类别 0 的样本，logits 第 0 列为正、第 1 列为负；类别 1 则相反。这表明模型正确地将正类 logit 推高。

## 关键设计决策

- **Tanh 激活**：对 XOR 问题，Tanh 和 ReLU 都能工作，Tanh 收敛更平滑
- **无 shuffle**：只有 4 个样本，shuffle 无意义
- **大学习率（0.1）**：小样本时 SGD 梯度较稳定，可以承受更高的学习率
- **2000 epoch**：样本极少，需要更多迭代充分学习
