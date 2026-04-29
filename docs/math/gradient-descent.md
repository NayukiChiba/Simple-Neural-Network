# 梯度下降优化

## 优化问题

神经网络的训练本质上是一个优化问题：

$$ \boldsymbol{\theta}^* = \arg\min_{\boldsymbol{\theta}} \mathcal{L}(\boldsymbol{\theta}) $$

其中 $\boldsymbol{\theta}$ 表示网络中的所有可训练参数（所有权重和偏置的集合），$\mathcal{L}$ 是损失函数。

由于神经网络的高度非线性和大规模参数，无法得到解析解，必须使用迭代优化方法。

---

## 梯度下降法（Gradient Descent）

### 核心思想

梯度指向函数值增长最快的方向。要最小化函数，应向**负梯度方向**移动。

### 更新规则

$$ \boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \eta \nabla_{\boldsymbol{\theta}} \mathcal{L}(\boldsymbol{\theta}_t) $$

其中：
- $\boldsymbol{\theta}_t$：第 $t$ 步的参数
- $\eta > 0$：学习率（learning rate），控制步长
- $\nabla_{\boldsymbol{\theta}} \mathcal{L}$：损失对参数的梯度

### 学习率 $\eta$ 的影响

- **太小**：收敛极慢，可能在局部最小值附近振荡
- **太大**：可能越过最优解，甚至发散
- **恰好**：在有限的迭代次数内接近最优解

### 三种梯度下降变体

| 类型 | 梯度计算所用数据 | 特点 |
|---|---|---|
| 批量梯度下降（BGD） | 全部训练集 | 稳定但计算开销大 |
| 随机梯度下降（SGD） | 单个样本 | 更新快但噪声大 |
| **Mini-batch SGD** | 小批量样本 | **最常用**，平衡稳定性与效率 |

本项目使用 **Mini-batch SGD**。

---

## Mini-batch SGD

每次迭代从训练集中随机抽取 $B$ 个样本，用这些小批量样本计算梯度的近似：

$$ \nabla_{\boldsymbol{\theta}} \mathcal{L} \approx \frac{1}{|B|} \sum_{(\mathbf{x}, \mathbf{y}) \in B} \nabla_{\boldsymbol{\theta}} \mathcal{L}(\mathbf{x}, \mathbf{y}; \boldsymbol{\theta}) $$

**本项目的实现路径**：
1. `Trainer.createBatches()` 将数据划分为 mini-batch
2. 对每个 batch 调用 `Trainer.trainStep()`
3. `trainStep` 内部：forward → loss → backward → `optimizer.step()`
4. 损失函数在 `forward` 中已经做了平均（除以 $|B|$），因此梯度中也包含这个除法

---

## 代码实现：SGDOptimizer

`src/nn/optimizers/sgdOptimizer.py`:

```python
class SGDOptimizer:
    def __init__(self, learningRate: float = 0.01) -> None:
        if learningRate <= 0.0:
            raise ValueError("学习率必须为正数")
        self.learningRate = learningRate

    def step(self, layers: Iterable[BaseLayer]) -> None:
        """对所有层执行参数更新"""
        for layer in layers:
            parameters = layer.getParameters()
            gradients = layer.getGradients()

            for parameter, gradient in zip(parameters, gradients):
                # θ ← θ - η · ∇L
                parameter -= self.learningRate * gradient

    def zeroGrad(self, layers: Iterable[BaseLayer]) -> None:
        """清空所有层的梯度"""
        for layer in layers:
            layer.zeroGrad()
```

### step() 的工作流程

`step()` 遍历每个层，对每个（参数，梯度）对执行：

$$ \mathbf{W} \leftarrow \mathbf{W} - \eta \cdot \frac{\partial \mathcal{L}}{\partial \mathbf{W}} $$
$$ \mathbf{b} \leftarrow \mathbf{b} - \eta \cdot \frac{\partial \mathcal{L}}{\partial \mathbf{b}} $$

其中 `-=` 是 NumPy 的就地减法（直接修改参数数组）。

### zeroGrad() 的必要性

线性层的 `backward()` 使用 `self.gradWeights[...] =` 直接**覆盖**梯度，而不是累加。因此：

- 每次 `trainStep` 开始时必须调用 `zeroGrad()` 清除旧梯度
- 如果不清零，之前的梯度残留会导致错误的参数更新

`zeroGrad()` 的实现（在 `BaseLayer` 中）：

```python
def zeroGrad(self) -> None:
    for gradient in self.getGradients():
        gradient.fill(0.0)
```

---

## 完整训练步骤

`src/nn/training/trainer.py` 中的 `trainStep()` 将一切串联起来：

```python
def trainStep(self, inputBatch, targetBatch) -> float:
    # 1. 训练模式
    self.model.train()

    # 2. 清零梯度
    self.optimizer.zeroGrad(self.model.layers)

    # 3. 前向传播
    predictions = self.model.forward(inputBatch)

    # 4. 计算损失
    loss = self.lossFunction.forward(predictions, targetBatch)

    # 5. 损失梯度
    outputGradient = self.lossFunction.backward()

    # 6. 反向传播（逐层）
    self.model.backward(outputGradient)

    # 7. 参数更新
    self.optimizer.step(self.model.layers)

    return float(loss)
```

这 7 步构成了神经网络训练的原子操作。整个训练循环（`fit()`）就是对所有 epoch 和所有 batch 重复执行这 7 步。

---

## 训练收敛的条件

训练收敛通常表现为：
- 训练损失持续下降并趋于稳定
- 验证损失不再下降（早期停止的时机）
- 梯度范数趋近于零

在本项目的 Spiral 分类任务中，400 个 epoch 后训练准确率可达 $\sim 0.998$。
