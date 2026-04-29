# 模型持久化

## 为什么需要持久化

训练好的模型如果不保存，进程退出后模型参数即丢失。持久化解决以下需求：

1. **推理部署**：训练一次，反复用于预测
2. **训练检查点**：长时间训练中定期保存，防止训练意外中断
3. **模型分享**：将训练好的参数分发给其他人使用
4. **实验复现**：保存特定 epoch 的参数，便于对比分析

## 保存格式

使用 NumPy 的 `.npz` 格式（压缩的 NPZ 归档）。

### 参数命名规则

`CheckpointIO.saveCheckpoint()` 为每个参数生成唯一的键名：

```text
layer_{layerIndex}_param_{parameterIndex}
```

例如，模型 `[Linear(2,8), ReLU, Linear(8,3)]` 保存后的键名：

| 键名 | 对应参数 | 形状 |
|---|---|---|
| `layer_0_param_0` | 第 1 个 Linear 的 weights | `(2, 8)` |
| `layer_0_param_1` | 第 1 个 Linear 的 bias | `(1, 8)` |
| `layer_2_param_0` | 第 2 个 Linear 的 weights | `(8, 3)` |
| `layer_2_param_1` | 第 2 个 Linear 的 bias | `(1, 3)` |
| `parameter_count` | 元数据 | `[4]` |

注意 `layer_1`（ReLU 激活层）被跳过，因为它没有参数。

## 保存

```python
from src.nn.persistence import CheckpointIO

io = CheckpointIO()
io.saveCheckpoint(model, "checkpoints/best_model.npz")
```

**内部流程**：

```python
def saveCheckpoint(self, model, filePath):
    # 1. 确保目录存在
    Path(filePath).parent.mkdir(parents=True, exist_ok=True)

    # 2. 遍历所有层，收集参数
    paramDict = {}
    paramCount = 0
    for layerIdx, layer in enumerate(model.layers):
        for paramIdx, param in enumerate(layer.getParameters()):
            key = f"layer_{layerIdx}_param_{paramIdx}"
            paramDict[key] = param.copy()  # .copy() 防止引用
            paramCount += 1

    # 3. 添加元数据
    paramDict["parameter_count"] = np.array([paramCount])

    # 4. 写入文件
    np.savez(filePath, **paramDict)
```

## 加载

```python
# 构建与保存时结构一致的模型
newModel = SequentialModel([
    LinearLayer(inputDim=2, outputDim=8, randomSeed=42),
    ReLULayer(),
    LinearLayer(inputDim=8, outputDim=3, randomSeed=42),
])

io.loadCheckpoint(newModel, "checkpoints/best_model.npz")
```

### 加载校验

`loadCheckpoint()` 执行多层校验：

1. **文件存在性** — `FileNotFoundError`
2. **后缀检查** — 必须是 `.npz`
3. **参数计数** — 检查点中的 `parameter_count` 必须等于模型期望的参数总数
4. **形状匹配** — 每个参数在检查点中的形状必须与模型对应参数的形状一致
5. **完整性** — 实际加载的参数数量必须等于声明的数量

### 就地加载

关键实现细节——加载使用就地赋值而非替换引用：

```python
parameter[...] = savedParameter  # 就地写入，保留原数组引用
```

这很重要，因为模型中的 `self.weights` 等引用已被优化器（`SGDOptimizer`）和其他组件持有。直接替换引用（`parameter = savedParameter`）会断开这些引用链，导致后续训练无法正确更新参数。

## 完整工作流

```python
# === 训练阶段 ===
model = SequentialModel([...])
trainer = Trainer(model, ...)
history = trainer.fit(xTrain, yTrain, epochCount=400)

# 训练后保存
io = CheckpointIO()
io.saveCheckpoint(model, "checkpoints/trained_model.npz")

# === 推理阶段 ===
# 构建相同结构的模型
inferenceModel = SequentialModel([...])
io.loadCheckpoint(inferenceModel, "checkpoints/trained_model.npz")

# 直接预测
predictions = inferenceModel.predict(xNew)
```

## 最佳实践

- **保存前确保目录存在**：`saveCheckpoint` 会自动创建父目录
- **模型结构一致性**：加载时模型结构必须与保存时完全一致（层数、每层的输入/输出维度、是否有偏置）
- **命名版本化**：使用有意义的文件名如 `model_epoch100.npz`、`best_valid_acc.npz`
- **定期保存检查点**：长时间训练中每 N 个 epoch 保存一次，避免训练中断导致全部丢失
