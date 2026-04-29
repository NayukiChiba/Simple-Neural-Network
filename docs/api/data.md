# Data — 数据模块 API

## DataGenerator

`src/nn/data/dataGenerator.py`

数据集生成器，生成 XOR、Spiral、Sine 三类数据集并保存为 `.npz` 文件。

### 构造函数

```python
DataGenerator(seed: int = 42)
```

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `seed` | `int` | `42` | 随机种子 |

**实例属性**：

| 属性 | 值 | 说明 |
|---|---|---|
| `train_ratio` | `0.7` | 训练集比例 |
| `valid_ratio` | `0.15` | 验证集比例 |
| `test_ratio` | `0.15` | 测试集比例 |
| `SPIRAL_CLASS_COUNT` | `3` | 螺旋数据集类别数 |
| `SPIRAL_SAMPLES_PER_CLASS` | `1000` | 每类样本数 |
| `SINE_SAMPLE_COUNT` | `600` | 正弦数据集样本数 |

### generateAllDatasets()

```python
def generateAllDatasets() -> None
```

生成全部三类数据集。依次调用：
- `generateXorDataset()`
- `generateSpiralDataset()`
- `generateSineDataset()`

### generateXorDataset()

生成 XOR 二分类数据：

$$ \mathbf{X} = \begin{bmatrix} 0 & 0 \\ 0 & 1 \\ 1 & 0 \\ 1 & 1 \end{bmatrix}, \quad \mathbf{y} = [0, 1, 1, 0] $$

4 个样本，无训练/验证/测试划分。保存到 `datasets/xor/xor.npz`。

### generateSpiralDataset()

生成三分类螺旋数据，共 $3 \times 1000 = 3000$ 个样本。添加高斯噪声 $\epsilon \sim \mathcal{N}(0, 0.2)$。

按 70/15/15 划分后保存到 `datasets/spiral/{train,valid,test}.npz`。

### generateSineDataset()

生成一维正弦回归数据：$y = \sin(x) + \epsilon, \; x \in [-2\pi, 2\pi], \; \epsilon \sim \mathcal{N}(0, 0.1)$。

600 个样本，按 70/15/15 划分。保存到 `datasets/sine/{train,valid,test}.npz`。

### splitDataset()

```python
def splitDataset(
    self, x: np.ndarray, y: np.ndarray
) -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]
```

随机打乱后按比例划分数据集。先调用 `validateSplitRatios()` 验证比例和是否为 1.0。

---

## DataLoader

`src/nn/data/dataLoader.py`

从 `.npz` 文件加载数据集。

**类型别名**：
```python
DatasetPair = tuple[np.ndarray, np.ndarray]          # (x, y)
SplitDataset = tuple[DatasetPair, DatasetPair, DatasetPair]  # (train, valid, test)
DatasetLoader = DataLoader  # 别名
```

### 构造函数

```python
DataLoader()
```

无参数。

### loadDataset()

```python
def loadDataset(self, datasetName: Literal["xor", "spiral", "sine"]) -> DatasetPair | SplitDataset
```

按名称加载数据集。

| `datasetName` | 返回类型 |
|---|---|
| `"xor"` | `DatasetPair` |
| `"spiral"` | `SplitDataset` |
| `"sine"` | `SplitDataset` |

### loadXorDataset()

```python
def loadXorDataset() -> DatasetPair
```

返回 `(x, y)`：输入形状 `(4, 2)`，标签形状 `(4,)`。

### loadSpiralDataset()

```python
def loadSpiralDataset() -> SplitDataset
```

返回 `((x_train, y_train), (x_valid, y_valid), (x_test, y_test))`。输入形状 $(N, 2)$，标签形状 $(N,)$。

### loadSineDataset()

```python
def loadSineDataset() -> SplitDataset
```

返回 `((x_train, y_train), (x_valid, y_valid), (x_test, y_test))`。输入形状 $(N, 1)$，标签形状 $(N, 1)$。

### validateDataset()

```python
def validateDataset(self, x: np.ndarray, y: np.ndarray, datasetName: str) -> None
```

验证数据合法性：
- 类型必须是 `np.ndarray`
- 不能是标量
- 样本数 > 0
- 样本数一致

**异常**：
- `TypeError` — 类型不是 `np.ndarray`
- `ValueError` — 内容不合法
