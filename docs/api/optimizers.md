# Optimizers — 优化器 API

## SGDOptimizer

`src/nn/optimizers/sgdOptimizer.py`

随机梯度下降（Stochastic Gradient Descent）优化器。

### 构造函数

```python
SGDOptimizer(learningRate: float = 0.01)
```

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `learningRate` | `float` | `0.01` | 学习率 $\eta$，必须为正数 |

**异常**：
- `ValueError` — `learningRate <= 0.0`

### step()

```python
def step(self, layers: Iterable[BaseLayer]) -> None
```

对所有层执行参数更新。

**更新规则**：

$$ \boldsymbol{\theta} \leftarrow \boldsymbol{\theta} - \eta \cdot \nabla_{\boldsymbol{\theta}} \mathcal{L} $$

即对每个参数-梯度对：

```python
parameter -= self.learningRate * gradient
```

**参数**：

| 参数 | 类型 | 说明 |
|---|---|---|
| `layers` | `Iterable[BaseLayer]` | 需要更新的层列表（通常为 `model.layers`） |

**内部流程**：
1. 遍历每个 layer
2. 获取 `layer.getParameters()` 和 `layer.getGradients()`
3. 逐对验证形状一致
4. 就地更新参数

**异常**：
- `ValueError` — 某层的参数与梯度数量不一致
- `ValueError` — 某对参数与梯度形状不一致

### zeroGrad()

```python
def zeroGrad(self, layers: Iterable[BaseLayer]) -> None
```

清空所有层的梯度。内部委托给各层的 `layer.zeroGrad()`。

```python
for layer in layers:
    layer.zeroGrad()
```

### 使用示例

```python
optimizer = SGDOptimizer(learningRate=0.03)

# 训练步骤
optimizer.zeroGrad(model.layers)   # 清零梯度
model.forward(batchX)               # 前向
loss = lossFunc.forward(...)        # 计算损失
model.backward(lossFunc.backward()) # 反向（写入梯度）
optimizer.step(model.layers)        # 更新参数
```
