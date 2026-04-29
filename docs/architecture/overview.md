# 项目架构总览

## 设计哲学

本项目遵循以下核心原则：

- **显式优于隐式** — 每个数学操作在代码中一目了然
- **简单优于复杂** — 模块职责单一，不引入不必要的抽象
- **零外部深度学习框架依赖** — 仅使用 NumPy 完成矩阵运算
- **代码即文档** — 命名清晰，注释充分，类型标注完整

## 模块依赖关系

```text
                main.py
                   │
              ┌────┴────┐
              │  Trainer │ ←── 顶层编排器
              └────┬────┘
                   │
    ┌──────────────┼──────────────┐
    │              │              │
┌───▼───┐   ┌─────▼─────┐  ┌────▼─────┐
│ Model │   │   Loss    │  │Optimizer │
│Sequen-│   │ MSE / CE  │  │   SGD    │
│ tial  │   └─────┬─────┘  └──────────┘
└───┬───┘         │
    │             │
┌───▼───┐         │
│Layers │         │
│Linear │         │
│ReLU   │         │
│Sigmoid│         │
│Tanh   │         │
└───┬───┘         │
    │             │
┌───▼─────────────▼──┐
│   CheckpointIO     │ ←── 持久化
└────────────────────┘
```

依赖方向：`Trainer` 依赖 `Model`、`Loss`、`Optimizer`，而这三者彼此独立，仅通过 `Trainer` 协调。

## 设计模式

### 1. 模板方法模式（Template Method）— BaseLayer

`BaseLayer` 定义了神经网络层的通用接口，子类实现具体的 `forward()` 和 `backward()`：

```python
class BaseLayer(ABC):
    @abstractmethod
    def forward(self, inputData: np.ndarray) -> np.ndarray: ...
    @abstractmethod
    def backward(self, outputGradient: np.ndarray) -> np.ndarray: ...
    def getParameters(self) -> list[np.ndarray]: return []
    def getGradients(self) -> list[np.ndarray]: return []
    def zeroGrad(self): ...
    def train(self): ...
    def eval(self): ...
```

参数层（`LinearLayer`）覆盖 `getParameters()` / `getGradients()`，无参数层（激活函数）沿用基类的空列表实现。

### 2. 组合模式（Composite）— SequentialModel

`SequentialModel` 本身继承自 `BaseLayer`，同时容纳多个 `BaseLayer` 实例。这使得模型在接口上等同于单个层——可以嵌套、可以作为其他模型的子组件（虽然实践中不常用）。

```python
class SequentialModel(BaseLayer):
    def add_layer(self, layer: BaseLayer): ...
    def forward(self, inputData): ...
    def backward(self, outputGradient): ...
```

### 3. 策略模式（Strategy）— Loss 和 Optimizer

- **损失函数**：`MSELoss` 和 `CrossEntropyLoss` 是可互换的策略
- **优化器**：`SGDOptimizer` 是可替换的策略（未来可扩展 Momentum、Adam 等）

`Trainer` 通过构造函数接受这些策略对象，不关心具体实现。

## 数据所有权

- **权重和偏置** 由 `LinearLayer` 持有和管理
- **梯度** 由 `LinearLayer` 持有，`backward()` 写入，`zeroGrad()` 清空
- **中间缓存**（`inputCache` / `outputCache`）由各层在 `forward()` 中写入，`backward()` 中读取
- **训练历史** 由 `Trainer.fit()` 返回，格式为 `dict[str, list[float]]`

## 扩展点

当前架构预留了清晰的扩展点：

| 扩展 | 需要做什么 |
|---|---|
| 新激活函数 | 继承 `BaseLayer`，实现 `forward()` 和 `backward()` |
| 新损失函数 | 实现 `forward(predictions, targets)` 和 `backward()` 方法 |
| 新优化器 | 实现 `step(layers)` 和 `zeroGrad(layers)` 方法 |
| 新网络层 | 继承 `BaseLayer`，实现前向/反向传播，可选地覆盖参数接口 |
| Dropout / BatchNorm | 继承 `BaseLayer`，利用 `isTraining` 标志区分训练/评估行为 |
