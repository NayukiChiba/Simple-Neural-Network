# 代码规范

## Ruff 配置

本项目使用 [Ruff](https://docs.astral.sh/ruff/) 进行代码格式化和检查。配置位于 `pyproject.toml`。

```toml
[tool.ruff]
target-version = "py311"
line-length = 88

[tool.ruff.lint]
select = ["E", "F", "I"]
ignore = ["E501", "F841"]

[tool.ruff.format]
quote-style = "double"
indent-style = "space"
line-ending = "auto"
```

**规则说明**：

| 规则 | 含义 |
|---|---|
| `E` | pycodestyle 错误检查 |
| `F` | Pyflakes 未使用变量/导入检查 |
| `I` | isort 导入排序 |
| `E501` (忽略) | 行过长（由自动换行处理） |
| `F841` (忽略) | 未使用的变量（某些场景有意为之） |

**运行命令**：

```bash
ruff check .          # 检查
ruff check . --fix    # 自动修复
ruff format .         # 格式化
```

## 命名规范

| 命名对象 | 风格 | 示例 |
|---|---|---|
| 文件名 | 驼峰命名法 | `linearLayer.py`, `crossEntropyLoss.py` |
| 类名 | 大驼峰 (PascalCase) | `LinearLayer`, `CrossEntropyLoss` |
| 函数/方法名 | 小驼峰 (camelCase) | `calculateAccuracy()`, `loadDataset()` |
| 变量名 | 小驼峰 | `inputData`, `outputGradient`, `batchSize` |
| 常量 | 全大写 + 下划线 | `PROJECT_ROOT`, `DATASETS_DIR` |

## 类型标注

所有公共方法都包含完整的类型标注：

```python
from typing import Optional, Iterable, Literal

def forward(self, inputData: np.ndarray) -> np.ndarray: ...

def fit(
    self,
    trainInputs: np.ndarray,
    trainTargets: np.ndarray,
    epochCount: int,
    validInputs: np.ndarray | None = None,
    validTargets: np.ndarray | None = None,
    verbose: bool = True,
) -> dict[str, list[float]]: ...
```

## 文档字符串

使用中文，格式为：简要说明 + Args/Returns/Raises。

```python
def linearLayerForward(inputData: np.ndarray) -> np.ndarray:
    """
    前向传播

    Args:
        inputData (np.ndarray): 输入数据, 形状为 (batchSize, inputDim)

    Returns:
        np.ndarray: 输出数据, 形状为 (batchSize, outputDim)
    """
```

## 路径处理

统一使用 `pathlib.Path`，不进行字符串拼接：

```python
from pathlib import Path

# ✅ 推荐
DATASETS_DIR = PROJECT_ROOT / "datasets"
checkpointPath = Path(filePath)
checkpointPath.parent.mkdir(parents=True, exist_ok=True)

# ❌ 避免
filepath = "datasets/" + name + "/file.npz"
```

## 随机数生成

使用 `np.random.default_rng(seed)` 创建独立生成器，不修改全局随机状态：

```python
# ✅ 推荐
self.rng = np.random.default_rng(randomSeed)

# ❌ 避免
np.random.seed(randomSeed)
```

## 异常处理

显式捕获具体异常类型，错误信息清楚指明期望值和实际值：

```python
if inputData.shape[1] != self.inputDim:
    raise ValueError(
        f"输入的维度不匹配, 期望 {self.inputDim}, 实际为 {inputData.shape[1]}"
    )
```
