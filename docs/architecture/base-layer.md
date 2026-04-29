# BaseLayer 接口设计

`BaseLayer` 是所有网络层的抽象基类，定义了一套统一的接口契约。任何网络层（无论有无参数）都通过实现这套契约参与训练流程。

## 完整接口

`src/nn/layers/baseLayer.py`:

```python
class BaseLayer(ABC):
    # ---- 生命周期 ----
    def __init__(self) -> None
    def train(self) -> None        # 切换到训练模式
    def eval(self) -> None         # 切换到评估模式

    # ---- 核心计算 ----
    @abstractmethod
    def forward(self, inputData: np.ndarray) -> np.ndarray
    @abstractmethod
    def backward(self, outputGradient: np.ndarray) -> np.ndarray

    # ---- 参数管理 ----
    def getParameters(self) -> list[np.ndarray]   # 默认返回 []
    def getGradients(self) -> list[np.ndarray]    # 默认返回 []
    def zeroGrad(self) -> None
    def hasParameters(self) -> bool               # len(getParameters()) > 0
```

## 状态管理

### `isTraining` 标志

```python
self.isTraining: bool = True        # 初始为训练模式
```

- `train()` → `isTraining = True`
- `eval()` → `isTraining = False`
- 在 `SequentialModel` 中，`train()` / `eval()` 会递归传播到所有子层
- `add_layer()` 新加入的层自动与模型模式对齐

### 输入/输出缓存

```python
self.inputCache: Optional[np.ndarray] = None   # 最近一次 forward 的输入
self.outputCache: Optional[np.ndarray] = None  # 最近一次 forward 的输出
```

| 缓存 | 写入时机 | 读取时机 | 用途 |
|---|---|---|---|
| `inputCache` | `forward()` 开始时 | `backward()` 中 | 计算权重梯度（$X^T \cdot dY$）、ReLU 导数 mask |
| `outputCache` | `forward()` 结束时 | `backward()` 中 | 计算激活函数导数（Sigmoid/Tanh）、forward 校验 |

## forward() 契约

```python
@abstractmethod
def forward(self, inputData: np.ndarray) -> np.ndarray:
    """
    Args:
        inputData: 输入张量（形状取决于具体层）
    Returns:
        outputData: 输出张量
    Side-effect:
        1. self.inputCache ← inputData
        2. self.outputCache ← outputData
    """
```

**调用者保证**：在调用 `backward()` 之前先调用 `forward()`。
**实现者保证**：必须缓存 `inputCache` 和 `outputCache`。

## backward() 契约

```python
@abstractmethod
def backward(self, outputGradient: np.ndarray) -> np.ndarray:
    """
    Args:
        outputGradient: ∂L/∂y，损失对该层输出的梯度
    Returns:
        inputGradient: ∂L/∂x，损失对该层输入的梯度
    Side-effect:
        对于参数层：写入 self.gradWeights, self.gradBias
    """
```

**梯度流**：

$$ \text{outputGradient} = \frac{\partial \mathcal{L}}{\partial \mathbf{y}} \quad \longrightarrow \quad \text{backward()} \quad \longrightarrow \quad \text{inputGradient} = \frac{\partial \mathcal{L}}{\partial \mathbf{x}} $$

## getParameters() / getGradients() 契约

这两个方法构成了**参数发现协议**，使优化器能遍历所有层的参数而不需要知道层的具体类型：

```python
def getParameters(self) -> list[np.ndarray]:
    # LinearLayer: [weights, bias]  (或仅 [weights] 如果 useBias=False)
    # 激活层: [] (继承自基类)
```

```python
def getGradients(self) -> list[np.ndarray]:
    # LinearLayer: [gradWeights, gradBias]  (或仅 [gradWeights])
    # 激活层: [] (继承自基类)
```

**契约**：
- 对于同一层，`getParameters()` 和 `getGradients()` 返回的列表长度必须相等
- 对应位置的参数和梯度形状必须一致

## zeroGrad() 契约

```python
def zeroGrad(self) -> None:
    for gradient in self.getGradients():
        gradient.fill(0.0)
```

将本层所有梯度数组**就地清零**。在每次 `trainStep` 开始时调用，防止梯度跨 batch 累积。

## hasParameters() 契约

```python
def hasParameters(self) -> bool:
    return len(self.getParameters()) > 0
```

用于判断层是否有可训练参数。激活层返回 `False`，线性层返回 `True`。

## 正确的子类实现模式

以 `LinearLayer` 为例：

```python
class LinearLayer(BaseLayer):
    def __init__(self, inputDim, outputDim, useBias=True, randomSeed=None):
        super().__init__()                    # ← 必须调用，初始化 isTraining 和 caches
        # ... 初始化自己的参数
        self.weights = ...
        self.gradWeights = np.zeros_like(self.weights)  # ← 梯度必须初始化为零

    def forward(self, inputData):
        self.inputCache = inputData           # ← 缓存输入
        outputData = inputData @ self.weights  # ← 计算
        self.outputCache = outputData          # ← 缓存输出
        return outputData

    def backward(self, outputGradient):
        # 1. 计算输入梯度（返回给上一层）
        inputGradient = outputGradient @ self.weights.T
        # 2. 计算并存储参数梯度（覆盖写入）
        self.gradWeights[...] = self.inputCache.T @ outputGradient
        return inputGradient

    def getParameters(self):
        return [self.weights, self.bias]  # 顺序任意，但必须与 getGradients 一致

    def getGradients(self):
        return [self.gradWeights, self.gradBias]  # 必须与 getParameters 顺序一致
```
