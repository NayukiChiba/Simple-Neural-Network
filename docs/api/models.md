# Models — 模型 API

## SequentialModel

`src/nn/models/sequentialModel.py`

顺序模型容器，继承自 `BaseLayer`，按顺序组织多个网络层。

### 构造函数

```python
SequentialModel(layers: Iterable[BaseLayer] | None = None)
```

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `layers` | `Iterable[BaseLayer] \| None` | `None` | 可选的层列表 |

**示例**：

```python
model = SequentialModel([
    LinearLayer(inputDim=2, outputDim=16, randomSeed=42),
    ReLULayer(),
    LinearLayer(inputDim=16, outputDim=3, randomSeed=42),
])
```

### add_layer()

```python
def add_layer(self, layer: BaseLayer) -> None
```

| 参数 | 类型 | 说明 |
|---|---|---|
| `layer` | `BaseLayer` | 要添加的层实例 |

向模型末尾追加一个层。新层自动与模型当前的训练/评估模式对齐。

**异常**：
- `TypeError` — `layer` 不是 `BaseLayer` 的子类
- `ValueError` — `layer is self`（防止自引用）

### forward()

```python
def forward(self, inputData: np.ndarray) -> np.ndarray
```

| 参数 | 类型 | 形状 | 说明 |
|---|---|---|---|
| `inputData` | `np.ndarray` | $(N, d_{in})$ | 批量输入 |
| **返回** | `np.ndarray` | $(N, d_{out})$ | 经过所有层后的输出 |

依次调用各层的 `forward()`，将每层的输出作为下一层的输入。

**异常**：
- `ValueError` — 模型为空（无层）

### backward()

```python
def backward(self, outputGradient: np.ndarray) -> np.ndarray
```

| 参数 | 类型 | 形状 | 说明 |
|---|---|---|---|
| `outputGradient` | `np.ndarray` | $(N, d_{out})$ | $\frac{\partial \mathcal{L}}{\partial \mathbf{y}_{\text{final}}}$ |
| **返回** | `np.ndarray` | $(N, d_{in})$ | $\frac{\partial \mathcal{L}}{\partial \mathbf{x}}$ |

**逆序**调用各层的 `backward()`，将梯度从输出层反向传播到输入层。

**异常**：
- `ValueError` — 模型为空
- `ValueError` — 未先调用 `forward()`（`outputCache is None`）

### getParameters()

```python
def getParameters(self) -> list[np.ndarray]
```

按层顺序**扁平化**聚合所有层的参数。

**示例**（两层 Linear + 一个 ReLU）：

```python
model = SequentialModel([
    LinearLayer(2, 8),   # params: [W1, b1]
    ReLULayer(),         # params: []
    LinearLayer(8, 3),   # params: [W2, b2]
])
params = model.getParameters()
# 返回: [W1, b1, W2, b2]  (4 个 np.ndarray)
```

### getGradients()

```python
def getGradients(self) -> list[np.ndarray]
```

与 `getParameters()` 对应，返回所有层的梯度列表，顺序一致。

### train() / eval()

```python
def train(self) -> None
def eval(self) -> None
```

递归传播到所有子层。`SequentialModel` 重写了这两个方法以确保子层状态同步。

### predict()

```python
def predict(self, inputData: np.ndarray) -> np.ndarray
```

在评估模式下执行前向传播，**自动恢复**原先的训练/评估状态。

| 参数 | 类型 | 说明 |
|---|---|---|
| `inputData` | `np.ndarray` | 输入数据 |
| **返回** | `np.ndarray` | 预测输出 |

**关键**：使用 `try/finally` 确保模式恢复：

```python
wasTraining = self.isTraining
self.eval()
try:
    return self.forward(inputData)
finally:
    if wasTraining:
        self.train()
```

### \_\_len\_\_()

```python
def __len__(self) -> int
```

返回模型中的层数：`len(model.layers)`。
