# 前向传播

## 概念

前向传播（Forward Propagation）是神经网络从输入到输出逐层计算的过程。给定输入 $\mathbf{X}$，网络逐层应用线性变换和激活函数，最终得到预测输出 $\hat{\mathbf{Y}}$。

## 逐层递推公式

设网络共有 $L$ 层。第 $l$ 层的前向传播为：

$$ \mathbf{a}^{(0)} = \mathbf{X} \quad \text{（输入）} $$

$$ \mathbf{z}^{(l)} = \mathbf{a}^{(l-1)} \mathbf{W}^{(l)} + \mathbf{b}^{(l)} \quad \text{（线性变换）} $$

$$ \mathbf{a}^{(l)} = f^{(l)}\left(\mathbf{z}^{(l)}\right) \quad \text{（激活函数）} $$

其中：
- $\mathbf{a}^{(l-1)}$ 是第 $l-1$ 层的输出（也是第 $l$ 层的输入）
- $\mathbf{W}^{(l)}$ 是第 $l$ 层的权重矩阵
- $\mathbf{b}^{(l)}$ 是第 $l$ 层的偏置
- $f^{(l)}$ 是第 $l$ 层的激活函数

## 具体示例：两层隐藏网络

考虑一个完整的网络结构：

$$ \mathbf{X} \rightarrow \text{Linear}^{(1)} \rightarrow \text{ReLU} \rightarrow \text{Linear}^{(2)} \rightarrow \text{Softmax} + \text{CrossEntropy} $$

**设具体维度**：
- 输入 $\mathbf{X}$：形状 $(N, 2)$ — 2 维特征
- $\mathbf{W}^{(1)}$：形状 $(2, 16)$ — 第 1 层 16 个神经元
- $\mathbf{W}^{(2)}$：形状 $(16, 3)$ — 第 2 层 3 个神经元（3 类输出）

**第 1 层（Linear + ReLU）**:

$$ \mathbf{z}^{(1)} = \mathbf{X}_{(32, 2)} \cdot \mathbf{W}^{(1)}_{(2, 16)} + \mathbf{b}^{(1)}_{(1, 16)} = \mathbf{Z}^{(1)}_{(32, 16)} $$

$$ \mathbf{a}^{(1)} = \text{ReLU}(\mathbf{z}^{(1)}) = \max(0, \mathbf{z}^{(1)}) $$

**第 2 层（Linear）**:

$$ \mathbf{z}^{(2)} = \mathbf{a}^{(1)}_{(32, 16)} \cdot \mathbf{W}^{(2)}_{(16, 3)} + \mathbf{b}^{(2)}_{(1, 3)} = \mathbf{Z}^{(2)}_{(32, 3)} $$

$32$ 是 batch size，$(2, 16, 3)$ 分别对应每层的神经元数量。

最终 $\mathbf{z}^{(2)}$（logits）输入到损失函数。

---

## 代码实现：SequentialModel.forward()

`src/nn/models/sequentialModel.py:69-89`:

```python
def forward(self, inputData: np.ndarray) -> np.ndarray:
    # 缓存输入，供 backward 使用
    self.inputCache = inputData

    outputData = inputData
    # 按顺序通过每一层
    for layer in self.layers:
        outputData = layer.forward(outputData)

    # 缓存最终输出，供 backward 校验
    self.outputCache = outputData
    return outputData
```

- `self.layers` 是一个有序列表（如 `[LinearLayer, ReLULayer, LinearLayer]`）
- 每层的 `forward()` 内部会缓存自己的 `inputCache` 和 `outputCache`
- 模型的 `inputCache` 保存网络的原始输入，`outputCache` 保存最终输出

---

## 模式：训练 vs 评估

`BaseLayer` 维护一个 `isTraining` 标志，通过 `train()` 和 `eval()` 切换。

```python
def train(self):
    self.isTraining = True

def eval(self):
    self.isTraining = False
```

当前实现中，`isTraining` 标志主要影响：
- `Dropout`（尚未实现，但接口已预留）
- `BatchNorm`（尚未实现，但接口已预留）
- `SequentialModel.add_layer()` — 新添加的层会继承模型的训练/评估模式

训练时通常调用 `model.train()`，评估和预测时调用 `model.eval()`。

---

## 前向传播中的缓存机制

前向传播的关键副作用是**缓存中间结果**，因为反向传播需要用到这些值：

| 缓存变量 | 所在类 | 用途 |
|---|---|---|
| `inputCache` | `BaseLayer` | 存储层的输入，用于计算权重梯度（如 `X^T @ dY`） |
| `outputCache` | `BaseLayer` | 存储层的输出，用于计算激活函数的导数（如 `y(1-y)`） |

对于 `LinearLayer.backward()`，需要 `inputCache` 来计算 $\mathbf{X}^T \frac{\partial L}{\partial \mathbf{Y}}$。

对于 `SigmoidLayer.backward()`，需要 `outputCache` 来计算 $\frac{\partial L}{\partial \mathbf{Y}} \odot \mathbf{y}(1-\mathbf{y})$。

这便是为什么在前向传播中必须保存这些中间值——它们构成了反向传播所需计算图的一半信息。
