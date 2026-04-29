# 快速示例

本文展示两个端到端的训练示例：XOR 二分类和 Sine 回归。

## XOR 二分类

XOR 是一个经典的非线性二分类问题，需要至少一个隐藏层才能解决。

```python
from src.nn.data import DataLoader
from src.nn.layers import LinearLayer, TanhLayer
from src.nn.losses import CrossEntropyLoss
from src.nn.models.sequentialModel import SequentialModel
from src.nn.optimizers import SGDOptimizer
from src.nn.training import Trainer

# 1. 加载数据
loader = DataLoader()
x, y = loader.loadXorDataset()

# 2. 构建模型：2 输入 → 8 隐藏 → 2 输出
model = SequentialModel([
    LinearLayer(inputDim=2, outputDim=8, randomSeed=42),
    TanhLayer(),
    LinearLayer(inputDim=8, outputDim=2, randomSeed=42),
])

# 3. 配置训练器
trainer = Trainer(
    model=model,
    lossFunction=CrossEntropyLoss(),
    optimizer=SGDOptimizer(learningRate=0.1),
    taskType="classification",
    batchSize=4,
    shuffle=False,
    randomSeed=42,
)

# 4. 训练
history = trainer.fit(
    trainInputs=x,
    trainTargets=y,
    epochCount=2000,
    verbose=True,
)

# 5. 评估
result = trainer.evaluate(x, y)
print(f"最终损失: {result['loss']:.6f}")
print(f"最终准确率: {result['accuracy']:.6f}")
```

**预期输出**：经过 2000 个 epoch 后，准确率应达到 **1.0**。

---

## Sine 回归

Sine 是一个一维回归任务，目标是从带噪声的 sin(x) 采样点中学习正弦函数。

```python
from src.nn.data import DataLoader
from src.nn.layers import LinearLayer, TanhLayer
from src.nn.losses import MSELoss
from src.nn.models.sequentialModel import SequentialModel
from src.nn.optimizers import SGDOptimizer
from src.nn.training import Trainer

# 1. 加载数据
loader = DataLoader()
(xTrain, yTrain), (xValid, yValid), (xTest, yTest) = loader.loadSineDataset()

# 2. 构建模型：1 输入 → 16 隐藏 → 16 隐藏 → 1 输出
model = SequentialModel([
    LinearLayer(inputDim=1, outputDim=16, randomSeed=42),
    TanhLayer(),
    LinearLayer(inputDim=16, outputDim=16, randomSeed=42),
    TanhLayer(),
    LinearLayer(inputDim=16, outputDim=1, randomSeed=42),
])

# 3. 配置训练器
trainer = Trainer(
    model=model,
    lossFunction=MSELoss(),
    optimizer=SGDOptimizer(learningRate=0.01),
    taskType="regression",
    batchSize=32,
    randomSeed=42,
)

# 4. 训练
history = trainer.fit(
    trainInputs=xTrain,
    trainTargets=yTrain,
    epochCount=300,
    validInputs=xValid,
    validTargets=yValid,
    verbose=True,
)

# 5. 评估
result = trainer.evaluate(xTest, yTest)
print(f"测试损失 (MSE): {result['loss']:.6f}")
```

**预期输出**：经过 300 个 epoch 后，测试 MSE 应显著降低。

---

## 模型保存与加载

```python
from src.nn.persistence import CheckpointIO

# 保存
io = CheckpointIO()
io.saveCheckpoint(model, "checkpoints/model.npz")

# 加载到新模型
newModel = SequentialModel([...])  # 结构必须一致
io.loadCheckpoint(newModel, "checkpoints/model.npz")

# 使用加载的模型直接预测
predictions = newModel.predict(xTest)
```

---

## 架构说明

上面两个示例中，模型结构如下：

$$ \mathbf{X} \rightarrow \text{Linear} \rightarrow \text{Activation} \rightarrow \cdots \rightarrow \text{Linear} \rightarrow \text{Loss} $$

- 分类任务最后一层输出 logits（未经 softmax），由 `CrossEntropyLoss` 内部完成 softmax
- 回归任务最后一层输出标量预测值，由 `MSELoss` 计算均方误差
- 隐藏层激活函数：Tanh（XOR 和 Sine）或 ReLU（Spiral）

详细的数学推导见以下章节。
