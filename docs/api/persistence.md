# Persistence — 持久化 API

## CheckpointIO

`src/nn/persistence/checkpointIO.py`

模型检查点读写器，支持将模型参数保存到 `.npz` 文件并加载回来。

### 构造函数

```python
CheckpointIO()
```

无参数。

---

### saveCheckpoint()

```python
def saveCheckpoint(self, model: SequentialModel, filePath: str | Path) -> None
```

将模型参数保存到 `.npz` 文件。

| 参数 | 类型 | 说明 |
|---|---|---|
| `model` | `SequentialModel` | 要保存的模型 |
| `filePath` | `str \| Path` | 检查点文件路径（必须以 `.npz` 结尾） |

**保存格式**：

```text
checkpoint.npz
├── layer_0_param_0    # 第 0 层第 0 个参数（如 weights）
├── layer_0_param_1    # 第 0 层第 1 个参数（如 bias）
├── layer_2_param_0    # 第 2 层第 0 个参数（跳过激活层）
├── layer_2_param_1    # 第 2 层第 1 个参数
└── parameter_count    # 参数总数（元数据，用于校验）
```

**关键行为**：
- 跳过无参数层（如 ReLU/Sigmoid/Tanh）
- 使用 `.copy()` 避免引用问题
- 自动创建父目录

**异常**：
- `ValueError` — 文件后缀不是 `.npz`

---

### loadCheckpoint()

```python
def loadCheckpoint(self, model: SequentialModel, filePath: str | Path) -> None
```

从 `.npz` 文件加载参数到模型。

| 参数 | 类型 | 说明 |
|---|---|---|
| `model` | `SequentialModel` | 要写入参数的模型（结构需与保存时一致） |
| `filePath` | `str \| Path` | 检查点文件路径 |

**校验步骤**：
1. 文件存在性检查
2. 文件后缀检查（必须 `.npz`）
3. 参数数量匹配检查（`parameter_count` vs 模型期望数）
4. 逐参数形状匹配检查
5. 实际加载数量 vs 声明数量一致性检查

**加载方式**：就地写入（`parameter[...] = savedParameter`），保留数组引用不变。

**异常**：
- `FileNotFoundError` — 文件不存在
- `ValueError` — 后缀不是 `.npz`
- `ValueError` — 缺少 `parameter_count` 字段
- `ValueError` — 参数数量不匹配
- `ValueError` — 参数形状不匹配
- `ValueError` — 实际加载数量与声明不一致

---

### 使用示例

```python
from src.nn.persistence import CheckpointIO
from src.nn.models.sequentialModel import SequentialModel

io = CheckpointIO()

# 保存
io.saveCheckpoint(model, "checkpoints/best_model.npz")

# 加载（需构建相同结构的模型）
newModel = SequentialModel([...])  # 结构必须与保存时一致
io.loadCheckpoint(newModel, "checkpoints/best_model.npz")

# 加载后可直接预测
predictions = newModel.predict(xTest)
```
