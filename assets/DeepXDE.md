# PINN



使用PINN算法解微分方程的一般步骤:
Step 1 构造一个输入为 $\mathscr{x}$，参数为 $\mathscr{\theta}$ 的神经网络 $\hat{u}(\mathscr{x};\mathscr{\theta})$。
Step 2 为微分方程、边界/初始条件指定两个训练集 $\mathscr{T}_{f}$ 和 $\mathscr{T}_{b}$。(有时候，还需要额外的观察点数据集 $\mathscr{T}_{d}$，比如在求微分方程参数的逆问题中)
Step 3 通过对微分方程和边界条件残差(同样还有观察点数据集的残差)的加权 $L^2$ 范数求和来指定损失函数 $\mathscr{L}(\mathscr{\theta}, \mathscr{T})$。
Step 4 指定优化算法、学习率、迭代轮数、残差的权重系数，通过最小化损失函数 $\mathscr{L}(\mathscr{\theta}, \mathscr{T})$ 来训练神经网络以找到最佳参数  $\theta^*$。



使用DeepXDE求解微分方程的步骤：

Step 1 使用几何模块指定计算域。
Step 2 按照DeepXDE后端的深度学习框架，使用TensorFlow、PyTorch等的语法指定微分方程。
Step 3 指定边界和初始条件。
Step 4 将几何、偏微分方程和边界/初始条件一起组合到 data.PDE 或 data.TimePDE 中，分别用于时间无关问题或时间相关问题。要指定训练数据，可以设置特定的点位置，或者只设置点的数量，然后 DeepXDE将在网格上或随机采样所需数量的点。
Step 5 使用maps模块构建神经网络。
Step 6 通过结合第 4 步中的 PDE 问题和第 5 步中的神经网络来定义模型。
Step 7 调用Model.compile设置优化超参数，例如优化器、学习率和损失权重。
Step 8 调用Model.train从随机初始化或使用参数模型恢复路径的预训练模型训练网络。使用回调来监控和修改训练行为非常灵活。
Step 9 调用Model.predict预测不同位置的PDE解。

























# DeepXDE

[DeepXDE](https://github.com/lululxvi/deepxde)是一个用于科学机器学习的库。深度学习库DeepXDE可以

- 通过物理信息神经网络 (PINN) 求解正向和逆偏微分方程 (PDE)，
- 通过 PINN 求解正向和逆积分微分方程 (IDE)，
- 通过分数 PINN (fPINN) 求解正向和逆分数偏微分方程 (fPDE)，
- 通过深度算子网络（DeepONet、MIONet、DeepM&Mnet）逼近算子，
- 通过多保真神经网络 (MFNN) 从多保真数据中逼近函数。

[DeepXDE](https://github.com/lululxvi/deepxde) is a library for scientific machine learning. Use DeepXDE if you need a deep learning library that

- solves forward and inverse partial differential equations (PDEs) via physics-informed neural network (PINN),
- solves forward and inverse integro-differential equations (IDEs) via PINN,
- solves forward and inverse fractional partial differential equations (fPDEs) via fractional PINN (fPINN),
- approximates operators via deep operator network (DeepONet, MIONet, DeepM&Mnet),
- approximates functions from multi-fidelity data via multi-fidelity NN (MFNN).



**算法论文**

- 通过 PINN [ [SIAM Rev.](https://doi.org/10.1137/19M1274067) ]、梯度增强 PINN (gPINN) [ [arXiv](https://arxiv.org/abs/2111.02801) ]求解 PDE 和 IDE
- 通过 fPINN 求解 fPDE [ [SIAM J. Sci. 计算。](https://epubs.siam.org/doi/abs/10.1137/18M1229845)]
- 通过 NN 任意多项式混沌 (NN-aPC) 求解随机偏微分方程 [ [J. Comput. 物理。](https://www.sciencedirect.com/science/article/pii/S0021999119305340)]
- 通过具有硬约束 (hPINN) 的 PINN 求解逆设计/拓扑优化 [ [SIAM J. Sci. 计算。](https://doi.org/10.1137/21M1397908)]
- 通过 DeepONet [ [Nat. 学习算子。马赫 英特尔。](https://doi.org/10.1038/s42256-021-00302-5), [arXiv](https://arxiv.org/abs/2111.05512) ], MIONet [ [arXiv](https://arxiv.org/abs/2202.06137) ], DeepM&Mnet [ [J. Comput. 物理。](https://doi.org/10.1016/j.jcp.2021.110296), [J. 计算机。物理。](https://doi.org/10.1016/j.jcp.2021.110698)]
- 通过 MFNN 从多保真数据中学习 [ [J. Comput. 物理。](https://doi.org/10.1016/j.jcp.2019.109020),[美国国家科学院院刊](https://www.pnas.org/content/117/13/7052)]

**Papers on algorithms**

- Solving PDEs and IDEs via PINN [[SIAM Rev.](https://doi.org/10.1137/19M1274067)], gradient-enhanced PINN (gPINN) [[arXiv](https://arxiv.org/abs/2111.02801)]
- Solving fPDEs via fPINN [[SIAM J. Sci. Comput.](https://epubs.siam.org/doi/abs/10.1137/18M1229845)]
- Solving stochastic PDEs via NN-arbitrary polynomial chaos (NN-aPC) [[J. Comput. Phys.](https://www.sciencedirect.com/science/article/pii/S0021999119305340)]
- Solving inverse design/topology optimization via PINN with hard constraints (hPINN) [[SIAM J. Sci. Comput.](https://doi.org/10.1137/21M1397908)]
- Learning operators via DeepONet [[Nat. Mach. Intell.](https://doi.org/10.1038/s42256-021-00302-5), [arXiv](https://arxiv.org/abs/2111.05512)], MIONet [[arXiv](https://arxiv.org/abs/2202.06137)], DeepM&Mnet [[J. Comput. Phys.](https://doi.org/10.1016/j.jcp.2021.110296), [J. Comput. Phys.](https://doi.org/10.1016/j.jcp.2021.110698)]
- Learning from multi-fidelity data via MFNN [[J. Comput. Phys.](https://doi.org/10.1016/j.jcp.2019.109020), [PNAS](https://www.pnas.org/content/117/13/7052)]

![_images/pinn.png](https://deepxde.readthedocs.io/en/latest/_images/pinn.png)

![_images/deeponet.png](https://deepxde.readthedocs.io/en/latest/_images/deeponet.png)







# 一个简单的 ODE 系统

## 问题设置

我们将求解一个简单的 ODE 系统：

$$\large \frac{dy_1}{dt} = y ,\,\,\,\, \frac{dy_2}{dt} = -y_1,\,\,\,\, where \,\,\,\,t∈[0,10],$$



与初始条件 $\large y_1(0)=0,y_2(0)=1$  。

参考解决方案是 $$ \large  y_1 = \sin(x) ,\,\,\,\, y_2 = \cos(x)$$。

## 执行

本描述逐步实现了上述 ODE 系统的求解器。

首先，导入 DeepXDE 和 NumPy ( `np`) 模块：

```python
import deepxde as dde
import numpy as np
```

我们首先定义一个计算几何。我们可以使用一个内置类`TimeDomain`来定义一个时域，如下所示

```python
geom = dde.geometry.TimeDomain(0, 10)
```

接下来，我们表达 ODE 系统：

```python
def ode_system(x, y):
    y1, y2 = y[:, 0:1], y[:, 1:]
    dy1_x = dde.grad.jacobian(y, x, i=0)
    dy2_x = dde.grad.jacobian(y, x, i=1)
    return [dy1_x - y2, dy2_x + y1]
```

第一个参数`ode_system`是网络输入，即tt-坐标，这里我们将其表示为`x`。的第二个参数`ode_system`是网络输出，它是一个二维向量，其中第一个分量 ( ) 是`y[:, 0:1]`y1y1第二个分量 ( ) 是`y[:, 1:]`y2y2.

接下来，我们考虑初始条件。我们需要实现一个函数，它应该返回`True`子域内`False`的点和外部的点。在我们的例子中，重点tt的初始条件是t=0t=0. （请注意，由于舍入误差，通常明智的做法是`np.isclose`测试两个浮点值是否相等。）

```python
def boundary(x, on_initial):
    return np.isclose(x[0], 0)
```

参数`x`是`boundary`网络输入并且是dd-dim 向量，其中dd是维度和d=1d=1在这种情况下。为了便于实现`boundary`，使用布尔值`on_initial`作为第二个参数。如果点t=0t=0，`on_initial`则为真，否则`on_initial`为假。因此，我们也可以`boundary`用更简单的方式定义：

```python
def boundary(_, on_initial):
    return on_initial
```

然后使用计算域、初始函数和边界指定初始条件。参数`component`是指该 IC 是用于第一个组件还是第二个组件。

```python
ic1 = dde.icbc.IC(geom, np.sin, boundary, component=0)
ic2 = dde.icbc.IC(geom, np.cos, boundary, component=1)
```

现在，我们已经指定了几何、ODE 和初始条件。由于`PDE`也是 ODE 求解器，因此我们将 ODE 问题定义为

```python
data = dde.data.PDE(geom, ode_system, [ic1, ic2], 35, 2, solution=func, num_test=100)
```

数字 35 是在域内采样的训练残差点数，数字 2 是在边界（本例中为时域的左右端点）上采样的训练点数。我们使用 100 个点来测试 ODE 残差。参数 `solution=func`是计算我们解决方案误差的参考解决方案，我们将其定义如下：

```python
def func(x):
    return np.hstack((np.sin(x), np.cos(x)))
```

接下来，我们选择网络。在这里，我们使用深度为 4（即 3 个隐藏层）和宽度为 50 的全连接神经网络：

```python
layer_size = [1] + [50] * 3 + [2]
activation = "tanh"
initializer = "Glorot uniform"
net = dde.nn.FNN(layer_size, activation, initializer)
```

现在，我们有 ODE 问题和网络。我们建立一个`Model`并选择优化器和学习率：

```python
model = dde.Model(data, net)
model.compile("adam", lr=0.001, metrics=["l2 relative error"])
```

然后我们训练模型进行 20000 次迭代：

```python
losshistory, train_state = model.train(epochs=20000)
```

我们还保存和绘制最好的训练结果和损失历史。

```python
dde.saveplot(losshistory, train_state, issave=True, isplot=True)
```

## 完整代码

```python
"""Backend supported: tensorflow.compat.v1, tensorflow, pytorch"""
import deepxde as dde
import numpy as np


def ode_system(x, y):
    """ODE system.
    dy1/dx = y2
    dy2/dx = -y1
    """
    y1, y2 = y[:, 0:1], y[:, 1:]
    dy1_x = dde.grad.jacobian(y, x, i=0)
    dy2_x = dde.grad.jacobian(y, x, i=1)
    return [dy1_x - y2, dy2_x + y1]


def boundary(_, on_initial):
    return on_initial


def func(x):
    """
    y1 = sin(x)
    y2 = cos(x)
    """
    return np.hstack((np.sin(x), np.cos(x)))


geom = dde.geometry.TimeDomain(0, 10)
ic1 = dde.icbc.IC(geom, np.sin, boundary, component=0)
ic2 = dde.icbc.IC(geom, np.cos, boundary, component=1)
data = dde.data.PDE(geom, ode_system, [ic1, ic2], 35, 2, solution=func, num_test=100)

layer_size = [1] + [50] * 3 + [2]
activation = "tanh"
initializer = "Glorot uniform"
net = dde.nn.FNN(layer_size, activation, initializer)

model = dde.Model(data, net)
model.compile("adam", lr=0.001, metrics=["l2 relative error"])
losshistory, train_state = model.train(epochs=20000)

dde.saveplot(losshistory, train_state, issave=True, isplot=True)
```





















