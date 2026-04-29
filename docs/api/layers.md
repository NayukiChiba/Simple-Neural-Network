# Layers — 网络层 API

## BaseLayer

`src/nn/layers/baseLayer.py`

所有网络层的抽象基类，定义了统一的接口契约。

```python
class BaseLayer(ABC):
    isTraining: bool
    inputCache: np.ndarray | None
    outputCache: np.ndarray | None

    @abstractmethod
    def forward(self, inputData: np.ndarray) -> np.ndarray
    @abstractmethod
    def backward(self, outputGradient: np.ndarray) -> np.ndarray
    def getParameters(self) -> list[np.ndarray]
    def getGradients(self) -> list[np.ndarray]
    def zeroGrad(self) -> None
    def train(self) -> None
    def eval(self) -> None
    def hasParameters(self) -> bool
```

### 方法说明

| 方法 | 返回 | 说明 |
|---|---|---|
| `forward(inputData)` | `np.ndarray` | 抽象方法。前向传播，**必须**缓存 `inputCache` 和 `outputCache` |
| `backward(outputGradient)` | `np.ndarray` | 抽象方法。反向传播，参数层须写入梯度数组 |
| `getParameters()` | `list[np.ndarray]` | 返回层的可训练参数列表（基类返回 `[]`） |
| `getGradients()` | `list[np.ndarray]` | 返回层的梯度列表，顺序须与 `getParameters()` 一致 |
| `zeroGrad()` | `None` | 将所有梯度数组就地清零 (`fill(0.0)`) |
| `train()` | `None` | 设置 `isTraining = True` |
| `eval()` | `None` | 设置 `isTraining = False` |
| `hasParameters()` | `bool` | 是否有可训练参数 |

---

## LinearLayer

`src/nn/layers/linearLayer.py`

全连接线性层，实现 $\mathbf{Y} = \mathbf{X}\mathbf{W} + \mathbf{b}$。

### 构造函数

```python
LinearLayer(
    inputDim: int,            # 输入维度（必须 > 0）
    outputDim: int,           # 输出维度（必须 > 0）
    useBias: bool = True,     # 是否使用偏置
    randomSeed: int | None = None  # 随机种子
)
```

**初始化行为**：
- 权重使用 **Xavier 均匀初始化**：$W \sim \mathcal{U}\left[-\sqrt{\frac{6}{n_{in}+n_{out}}},\; \sqrt{\frac{6}{n_{in}+n_{out}}}\right]$
- 偏置初始化为 $\mathbf{0}$（形状 $(1, \text{outputDim})$）
- 梯度数组初始化为同形状零矩阵

### forward()

```python
def forward(self, inputData: np.ndarray) -> np.ndarray
```

| 参数 | 类型 | 形状 | 说明 |
|---|---|---|---|
| `inputData` | `np.ndarray` | $(N, d_{in})$ | 批量输入 |
| **返回** | `np.ndarray` | $(N, d_{out})$ | $\mathbf{X}\mathbf{W} + \mathbf{b}$ |

**异常**：
- `ValueError` — `inputData` 不是二维数组
- `ValueError` — `inputData.shape[1] != inputDim`

### backward()

```python
def backward(self, outputGradient: np.ndarray) -> np.ndarray
```

| 参数 | 类型 | 形状 | 说明 |
|---|---|---|---|
| `outputGradient` | `np.ndarray` | $(N, d_{out})$ | $\frac{\partial \mathcal{L}}{\partial \mathbf{Y}}$ |
| **返回** | `np.ndarray` | $(N, d_{in})$ | $\frac{\partial \mathcal{L}}{\partial \mathbf{X}}$ |

**副作用**：
- `self.gradWeights[...] = inputCache.T @ outputGradient`（形状 $(d_{in}, d_{out})$）
- `self.gradBias[...] = np.sum(outputGradient, axis=0, keepdims=True)`（形状 $(1, d_{out})$）

**数学公式**：
$$ \frac{\partial \mathcal{L}}{\partial \mathbf{X}} = \frac{\partial \mathcal{L}}{\partial \mathbf{Y}} \mathbf{W}^T $$
$$ \frac{\partial \mathcal{L}}{\partial \mathbf{W}} = \mathbf{X}^T \frac{\partial \mathcal{L}}{\partial \mathbf{Y}} $$
$$ \frac{\partial \mathcal{L}}{\partial \mathbf{b}} = \sum_n \left(\frac{\partial \mathcal{L}}{\partial \mathbf{Y}}\right)_{n,:} $$

---

## ReLULayer

`src/nn/layers/activationLayer.py`

ReLU 激活层，无参数。

### forward()

$$ \text{ReLU}(x) = \max(0, x) $$

```python
def forward(self, inputData: np.ndarray) -> np.ndarray
```

| 参数 | 类型 | 形状 | 说明 |
|---|---|---|---|
| `inputData` | `np.ndarray` | 任意 | 输入张量 |
| **返回** | `np.ndarray` | 同输入 | $\max(0, \mathbf{x})$ |

### backward()

$$ \frac{\partial \mathcal{L}}{\partial \mathbf{x}} = \frac{\partial \mathcal{L}}{\partial \mathbf{y}} \odot \mathbb{1}\{\mathbf{x} > 0\} $$

```python
def backward(self, outputGradient: np.ndarray) -> np.ndarray
```

**异常**：
- `ValueError` — 未先调用 `forward()`

---

## SigmoidLayer

`src/nn/layers/activationLayer.py`

Sigmoid 激活层，无参数。输出范围 $(0, 1)$。

### forward()

$$ \sigma(x) = \frac{1}{1 + e^{-x}} $$

### backward()

$$ \frac{\partial \mathcal{L}}{\partial \mathbf{x}} = \frac{\partial \mathcal{L}}{\partial \mathbf{y}} \odot \mathbf{y} \odot (1 - \mathbf{y}) $$

---

## TanhLayer

`src/nn/layers/activationLayer.py`

Tanh 激活层，无参数。输出范围 $(-1, 1)$。

### forward()

$$ \tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}} $$

### backward()

$$ \frac{\partial \mathcal{L}}{\partial \mathbf{x}} = \frac{\partial \mathcal{L}}{\partial \mathbf{y}} \odot (1 - \mathbf{y}^2) $$

---

## 激活层共同特性

- 三类激活层**均无参数和梯度**：`getParameters()` → `[]`，`getGradients()` → `[]`
- `zeroGrad()` 为空操作
- `hasParameters()` 返回 `False`
- 输入/输出形状不变
