## [物理信息神经网络](https://maziarraissi.github.io/PINNs/) 

--------Physics-Informed Neural Networks

**摘要**：

​		我们介绍了**物理信息神经网络**(physics informed neural networks)——在遵守由一般**非线性[偏微分方程]((https://en.wikipedia.org/wiki/Partial_differential_equation))**(general nonlinear partial differential equations)描述的任何给定物理定律的同时，训练用于解决**监督学习**(supervised learning)任务的神经网络。我们在解决两类主要问题的背景下介绍了我们的贡献：偏微分方程的**[数据驱动解决方案](https://arxiv.org/abs/1711.10561)和[数据驱动发现](https://arxiv.org/abs/1711.10566)**(data-driven solution and data-driven discovery)。根据可用数据的性质和排列，我们设计了两类不同的算法，即**连续时间模型和离散时间模型**(continuous time and discrete time models)。第一类模型形成了一个新的**数据高效的通用函数逼近器族**(family of data-efficient spatio-temporal function approximators)，它自然地将任何**潜在的物理定律**(underlying physical laws)编码为**先验信息**(prior information)；而后一类模型允许使用任意精确的隐式Runge-Kutta时间步进格式，具有无限的级数。通过流体、量子力学、反应扩散系统和非线性浅水波传播中的一系列经典问题，证明了该框架的有效性。在第一部分中，我们将演示如何使用这些网络来[推断偏微分方程的解](https://epubs.siam.org/doi/abs/10.1137/17M1120762)，并获得关于所有输入坐标和自由参数完全可微的物理信息代理模型。在第二部分中，我们关注[数据驱动发现的偏微分方程](https://www.sciencedirect.com/science/article/pii/S0021999117309014)问题。

**关键词**：数据驱动的科学计算(Data-driven scientific computing)、机器学习(Machine learning)、预测建模(Predictive modeling)、

​				龙格库塔方法( Runge-Kutta methods)、非线性动力学(Nonlinear dynamics)

For more information, please refer to the following: (https://maziarraissi.github.io/PINNs/)

- Raissi, Maziar, Paris Perdikaris, and George E. Karniadakis. "[Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations](https://www.sciencedirect.com/science/article/pii/S0021999118307125)." Journal of Computational Physics 378 (2019): 686-707.
- Raissi, Maziar, Paris Perdikaris, and George Em Karniadakis. "[Physics Informed Deep Learning (Part I): Data-driven Solutions of Nonlinear Partial Differential Equations](https://arxiv.org/abs/1711.10561)." arXiv preprint arXiv:1711.10561 (2017).
- Raissi, Maziar, Paris Perdikaris, and George Em Karniadakis. "[Physics Informed Deep Learning (Part II): Data-driven Discovery of Nonlinear Partial Differential Equations](https://arxiv.org/abs/1711.10566)." arXiv preprint arXiv:1711.10566 (2017).

### 非线性偏微分方程的数据驱动解

--------Data-driven Solutions of Nonlinear Partial Differential Equations

在这篇分为两部分的论文的[第一部分](https://arxiv.org/abs/1711.10561)中，我们将重点讨论计算**一般形式偏微分方程**(partial differential equations of the general form)的数据驱动解
$$
\large u_t + \cal N[u] = 0, x\in Ω, t \in [0,T],
$$
其中 $u(t,x)$ 表示潜在（隐藏）解，$\cal N[⋅]$ 是一个**非线性微分算子**(nonlinear differential operator)，$Ω$ 是 $R^D$ 的子集。接下来，我们提出了两类不同的算法，即**连续时间模型和离散时间模型**(continuous and discrete time models)，并通过不同的基准问题来突出它们的性质和性能。[这里](https://github.com/maziarraissi/PINNs)提供了所有代码和数据集。

#### 连续时间模型

--------Continuous Time Models

我们定义 $f(t,x)$ 为
$$
\large f := u_t + \cal N[u],
$$

然后用深度神经网络逼近 $u(t,x)$ 。这个假设产生了一个[物理信息神经网络](https://arxiv.org/abs/1711.10561) $f(t,x)$ 。这个网络可以通过计算图上的演算得到：[反向传播](http://colah.github.io/posts/2015-08-Backprop/)。

##### 示例1(Burgers方程)

作为一个例子，让我们考虑[Burgers方程](https://en.wikipedia.org/wiki/Burgers'_equation)。在一维空间中，Burger方程的Dirichlet边界条件如下
$$
\begin{align}
& \large u_t + uu_x - (0.01/\pi)u_{xx} = 0, x \in [-1,1], t \in [0,1],\\
& \large u(0,x) = -sin(\pi x),\\
& \large u(t,-1) = u(t,1) = 0.\\
\end{align}
$$



让我们定义 $f(t,x)$ 为
$$
\large f := u_t + uu_x - (0.01/\pi)u_{xx},
$$

然后用深度神经网络逼近 $u(t,x)$ 。为了强调这个想法的简单性，让我们用Python来实现它，只需要一点点[Tensorflow](https://www.tensorflow.org/)。为此， $u(t,x)$ 可以简单地定义为

```python
def u(t, x):
    u = neural_net(tf.concat([t,x],1), weights, biases)
    return u
```

相应地，[物理信息神经网络](https://arxiv.org/abs/1711.10561)$f(t,x)$ 采用以下形式

```python
def f(t, x):
    u = u(t, x)
    u_t = tf.gradients(u, t)[0]
    u_x = tf.gradients(u, x)[0]
    u_xx = tf.gradients(u_x, x)[0]
    f = u_t + u*u_x - (0.01/tf.pi)*u_xx
    return f
```

通过最小化均方误差损失，可以学习神经网络 $u(t,x)$ 和 $f(t,x)$ 之间的共享参数
$$
\begin{align}
& MSE = MSE_u + MSF_f,\\
& MSE_u = \frac 1 {N_u} \sum^{N_u}_{i=1} |u(t^i_u, x^i_u) - u^i|^2,\\
& MSE_f = \frac 1 {N_f} \sum^{N_f}_{i=1} |f(t^i_f, x^i_f)|^2.\\
\end{align}
$$
这里，$\{ {t^i_u, x^i_u, u^i} \}^{N_u}_{i=1}$ 表示 $u(t,x)$ 上的初始和边界训练数据，$\{{t^i_f,x^i_f}\}^{N_f}_{i=1}$指定 $f(t,x)$ 的配置点。损失 $MSE_u$ 对应于初始和边界数据，而 $MSE_f$ 在有限的配置点集上强制执行Burgers方程施加的结构。

下图总结了我们对Burgers方程的数据驱动解决方案的结果。

![img](https://maziarraissi.github.io/assets/img/Burgers_CT_inference.png)



> *Burgers方程*：**顶部**：预测解以及初始和边界训练数据。此外，我们使用10000个搭配点，这些搭配点是使用**拉丁超立方体采样策略**( Latin Hypercube Sampling strategy)生成的。**底部**：与顶部面板中白色垂直线描绘的三个时间快照相对应的预测解和精确解的比较。在一张NVIDIA Titan X GPU卡上进行模型培训大约需要60秒。

------

##### 示例2(薛定谔方程)

本例旨在强调我们的方法处理周期边界条件、复值解以及控制偏微分方程中不同类型非线性的能力。**非线性薛定谔方程**([nonlinear Schrödinger equation](https://en.wikipedia.org/wiki/Nonlinear_Schrödinger_equation))和周期边界条件由下式给出：
$$
\begin{align}
& \large ih_t + 0.5h_{xx} + |h|^2h = 0, x \in [-5,5], t \in [0,\pi/2],\\
& \large h(0,x) = 2 sech(x),\\
& \large h(t,-5) = h(t,5),\\
& \large h_x(t,-5) = h_x(t,5),\\
\end{align}
$$
其中 $h(t,x)$ 是复数解。让我们定义 $f(t,x)$ 为
$$
\large f := ih_t + 0.5h_{xx} + |h|^2h ,
$$

然后在 $h(t,x)$ 上放置一个复值神经网络(complex-valued neural network)。事实上，如果 $u$ 表示 $h$ 的实部，$v$ 是虚部，我们在 $h(t,x) = [u(t,x)\ \ \ \ \ \ v(t,x)]$上放置一个多输出神经网络。这将产生复数（多输出）物理信息神经网络 $f(t,x)$。通过最小化均方误差损失，可以学习神经网络 $h(t,x)$ 和 $f(t,x)$ 的共享参数
$$
\begin{align}
& MSE = MSE_0 + MSE_b + MSF_f,\\
& MSE_0 = \frac 1 {N_0} \sum^{N_0}_{i=1} |h(0,x^i_0) - h^i_0|^2,\\
& MSE_b = \frac 1 {N_b} \sum^{N_b}_{i=1} \left ( |h^i(t^i_b,-5) - h^i(t^i_b,5)|^2 + 
                              |h^i_x(t^i_b,-5) - h^i_x(t^i_b,5)|^2 \right ),\\
& MSE_f = \frac 1 {N_f} \sum^{N_f}_{i=1} |f(t^i_f, x^i_f)|^2.\\
\end{align}
$$
这里，$\{ {x^i_0, h^i_0} \}^{N_0}_{i=1}$ 表示初始数据，$\{ {t^i_b} \}^{N_b}_{i=1}$ 对应于边界上的配置点，$\{{t^i_f,x^i_f}\}^{N_f}_{i=1}$ 表示 $f(t,x)$ 上的配置点。因此，$MSE_0$ 对应于初始数据的损失， $MSE_b$ 强制执行周期性边界条件， $MSE_f$ 惩罚配置点上不满足的薛定谔方程。
下图总结了我们的实验结果。

![img](https://maziarraissi.github.io/assets/img/NLS.png)

> *Shrödinger方程*：**顶部**：预测解以及初始和边界训练数据。此外，我们使用20000个搭配点，这些搭配点是使用拉丁超立方体采样策略生成的。**底部**：与顶部面板中垂直虚线描绘的三个时间快照对应的预测和精确解的比较。

到目前为止，连续时间神经网络模型的一个潜在局限性在于需要使用大量的配置点 $N_f$，以便在整个时空域中实施物理信息约束。尽管这对一个或两个空间维度的问题没有重大影响，但它可能会在高维问题中引入严重的瓶颈，因为全局实施物理约束（即，在我们的例子中，偏微分方程）所需的配置点总数将以指数方式增加。在下一节中，我们提出了一种不同的方法，通过引入更结构化的神经网络表示，利用经典的龙格-库塔([Runge-Kutta](https://en.wikipedia.org/wiki/Runge–Kutta_methods))时间步方案，绕过配置点的需要。



#### 离散时间模型

--------Discrete Time Models

让我们利用一般形式的q阶Runge-Kutta方法，并得到
$$
\begin{align}
& \large u^{n+c_i} = u^n - \Delta t \sum^{q}_{j=1} a_{ij} \cal N[u^{n+c_j}], i=1,\dots,q, \\
& \large u^{n+1} = u^n - \Delta t \sum^{q}_{j=1} b_j \cal N[u^{n+c_j}]. \\
\end{align}
$$
这里， $u^{n+c_j}(x) = u(t^n+c_j \Delta t, x)$，对于 $j=1,\dots,q$。根据参数$\{a_{ij}, b_j, c_j\}$的选择，这种一般形式封装了隐式和显式时间步进方案。上述方程式可等效表示为
$$
\begin{align}
& \large u^n = u^n_i, i=1,\dots,q, \\
& \large u^n = u^n_{q+1}, \\
& \large u^n_i = u^{n+c_i} + \Delta t \sum^{q}_{j=1} a_{ij} \cal N[u^{n+c_j}], i=1,\dots,q, \\
& \large u^n_{q+1} = u^{n+1} + \Delta t \sum^{q}_{j=1} b_j \cal N[u^{n+c_j}]. \\
\end{align}
$$
我们将一个多输出神经网络置于 $[u^{n+c_1}(x), \dots, u^{n+c_q}(x), u^{n+1}(x)]$。
这一先验假设与上述方程一起产生了一个以 $x$ 为输入和输出的物理信息神经网络 $[u^n_1(x), \dots, u^n_q(x), u^n_{q+1}(x)]$ 。

##### 示例3(Allen Cahn方程)

本例旨在强调所提出的离散时间模型处理控制偏微分方程中不同类型非线性的能力。为此，让我们考虑[Allen-Cahn equation](https://en.wikipedia.org/wiki/Allen–Cahn_equation) 和周期边界条件。
$$
\begin{align}
& \large u_t - 0.0001 u_{xx} + 5u^3 - 5u = 0, x \in [-1,1], t \in [0,1], \\
& \large u(0,x) = x^2 cos(\pi x), \\
& \large u(t,-1) = u(t,1), \\
& \large u_x(t,-1) = u_x(t,1). \\
\end{align}
$$
Allen-Cahn方程是反应扩散系统领域的一个著名方程。它描述了多组分合金系统中的相分离过程，包括有序-无序转变。对于Allen-Cahn方程，非线性算子由下式给出：
$$
\large \cal N[u^{n+c_j}] = - 0.0001 u^{n+c_j}_{xx} + 5(u^{n+c_j})^3 - 5u^{n+c_j},
$$
通过最小化误差平方和，可以学习神经网络的共享参数
$$
\begin{align}
 SSE  &= SSE_n + SSE_b,\\
SSE_n &= \sum^{q+1}_{j=1} \sum^{N_n}_{i=1} |u^n_j(x^{n,i}) - u^{n,i}|^2,\\
\end{align}
$$

$$
\begin{equation}
\begin{aligned}
SSE_b &= \sum^q_{i=1} |u^{n+c_i}(-1) - u^{n+c_i}(1)|^2 + |u^{n+1}(-1) - u^{n+1}(1)|^2 \\
        & + \sum^q_{i=1} |u^{n+c_i}(-1) - u^{n+c_i}(1)|^2 + |u^{n+1}(-1) - u^{n+1}(1)|^2 ,\\
\end{aligned}
\end{equation}
$$

这里，$\{x^{n,i}, u^{n,i}\}^{N_n}_{i=1}$ 对应于时刻 $t_n$ 的数据。
下图总结了我们使用上述损失函数对网络进行训练后的预测。

![img](https://maziarraissi.github.io/assets/img/AC.png)

> *Allen-Cahn方程*：**顶部**：初始训练快照在t=0.1时的位置以及最终预测快照在t=0.9时的位置。**底部**：顶部面板中白色垂直线所示快照处的初始训练数据和最终预测。
> 
> 

------

### 非线性偏微分方程的数据驱动发现

--------Data-driven Discovery of Nonlinear Partial Differential Equations

在我们研究的[第二部分](https://arxiv.org/abs/1711.10566)中，我们将注意力转移到偏微分方程的数据驱动发现问题上。为此，让我们考虑一般形式的参数化和非线性偏微分方程。
$$
\large u_t + \cal N[u, \lambda] = 0, x\in Ω, t \in [0,T],
$$
其中 $u(t,x)$ 表示潜在（隐藏）解，$\cal N[⋅;\lambda]$ 是一个由 $\lambda$ 参数化的非线性算子，$Ω$ 是 $R^D$ 的子集。现在，偏微分方程的数据驱动发现问题提出了以下问题：给定一个系统的隐藏状态 $u(t,x)$ 的一小组分散且可能有噪声的观测值，最能描述观测数据的参数 $\lambda$ 是什么？
在下文中，我们将概述我们解决这一问题的两种主要方法，即连续时间模型和离散时间模型，以及针对各种基准的一系列结果和系统研究。在第一种方法中，我们将假设在整个时空域中存在分散的和潜在的噪声测量。在后者中，我们将尝试仅从在不同时间点拍摄的两个数据快照推断未知参数λ。本手稿中使用的所有数据和代码都可以在[GitHub](https://github.com/maziarraissi/PINNs)上公开获取。

#### 连续时间模型

--------Continuous Time Models

我们定义 $f(t,x)$ 为
$$
\large f := u_t + \cal N[u;\lambda],
$$

然后用深度神经网络逼近 $u(t,x)$ 。这个假设产生了一个[物理信息神经网络](https://arxiv.org/abs/1711.10561) $f(t,x)$ 。这个网络可以通过计算图上的演算得到：[反向传播](http://colah.github.io/posts/2015-08-Backprop/)。值得强调的是，微分算子 $\lambda$ 的参数变成了物理信息神经网络 $f(t,x)$ 的参数。

##### 示例4(Navier-Stokes方程)

我们的下一个例子涉及不可压缩流体流动的真实场景，如普遍存在的[Navier-Stokes方程](https://en.wikipedia.org/wiki/Navier–Stokes_existence_and_smoothness)所描述。Navier-Stokes方程描述了许多具有科学和工程意义的物理现象。它们可以用来模拟天气、洋流、管道中的水流和机翼周围的气流。完整和简化形式的Navier-Stokes方程有助于飞机和汽车的设计、血液流动的研究、发电站的设计、污染物扩散的分析以及许多其他应用。让我们考虑二维(2D)给出的Navier-斯托克斯方程。
$$
\begin{align}
\large u_t + \lambda_1(uu_x + vu_y) &= -p_x + \lambda_2(u_{xx} + u_{yy}),\\
\large v_t + \lambda_1(uv_x + vv_y) &= -p_y + \lambda_2(v_{xx} + v_{yy}),\\
\end{align}
$$
其中 $u(t,x,y)$ 表示速度场的 $x$ 分量， $v(t,x,y)$ 表示 $y$ 分量， $p(t,x,y)$ 表示压力。这里，$λ=(λ_1,λ_2)$是未知参数。在无散度函数集中搜索Navier-Stokes方程的解；即，
$$
\large u_x + v_y = 0.
$$
这个额外的方程是描述流体质量守恒的不可压缩流体的连续性方程。我们假设
$$
\large u = ψ_y,   v = −ψ_x,
$$

对于某些**潜函数**(latent function) $ψ(t, x, y)$。在此假设下，连续性方程将自动满足。给定速度场的噪声测量值$$\{t^i, x^i, y^i, u^i, v^i\}^N_{i=1}$$，我们感兴趣的是学习参数 $λ$ 以及压力 $p(t,x,y)$ 。我们定义 $f(t,x,y)$ 和 $g(t,x,y)$ 为
$$
\begin{align}
\large f &:= u_t + \lambda_1(uu_x + vu_y) + p_x - \lambda_2(u_{xx} + u_{yy}),\\
\large g &:= v_t + \lambda_1(uv_x + vv_y) + p_y - \lambda_2(v_{xx} + v_{yy}),\\
\end{align}
$$
并使用具有两个输出的单个神经网络联合逼近 $[ψ(t,x,y)\ \ \ p(t,x,y)]$ 。这个先验假设产生了一个[物理信息神经网络](https://arxiv.org/abs/1711.10566)$[f(t,x,y)\ \ \ g(t,x,y)]$。Navier-Stokes算子的参数 $λ$ 以及神经网络的参数 $[ψ(t,x,y)\ \ \ p(t,x,y)]$ 和 $[f(t,x,y)\ \ \ g(t,x,y)]$ 可以通过最小化均方误差损失来训练:
$$
\begin{equation}
\begin{aligned}
MSE &:= \frac 1 N \sum^N_{i=1} \left( |u(t^i, x^i, y^i) - u^i|^2 + |v(t^i, x^i, y^i)-v^i|^2\right)\\
    &+ \frac 1 N \sum^N_{i=1}  \left( |f(t^i, x^i, y^i)|^2 + |g(t^i, x^i, y^i)|^2\right).\\

\end{aligned}
\end{equation}
$$
下图显示了本例的结果摘要。

![img](https://maziarraissi.github.io/assets/img/NavierStokes_data.png)

> *Navier-Stokes方程*：**顶部**：Re=100时通过圆柱的不可压缩流和动态旋涡脱落。时空训练数据对应于圆柱尾迹中描绘的矩形区域。**底部**：流向和横向速度分量的训练数据点位置。

![img](https://maziarraissi.github.io/assets/img/NavierStokes_prediction.png)

> *Navier-Stokes方程*：**顶部**：代表性时刻的预测与精确瞬时压力场。根据定义，压力可以恢复到一个常数，因此证明两个图之间的不同大小是合理的。这种显著的定性一致性突出了物理信息神经网络识别整个压力场的能力，尽管在模型训练期间没有使用压力数据。**底部**：修正偏微分方程和已识别的方程。


到目前为止，我们的方法假设在整个时空域中都有分散的数据可用。然而，在许多实际情况下，人们可能只能在不同的时刻观察系统。在下一节中，我们将介绍一种不同的方法，该方法仅使用两个数据快照来解决数据驱动的发现问题。我们将看到，通过利用经典的Runge-Kutta时间步进方案，可以构建离散时间物理信息神经网络，即使数据快照之间的时间间隔非常大，也可以保持较高的预测精度。

#### 离散时间模型

--------Discrete Time Models

我们首先采用一般形式的q阶Runge-Kutta方法，并得到
$$
\begin{align}
& \large u^{n+c_i} = u^n - \Delta t \sum^{q}_{j=1} a_{ij} \cal N[u^{n+c_j}; \lambda], i=1,\dots,q, \\
& \large u^{n+1} = u^n - \Delta t \sum^{q}_{j=1} b_j \cal N[u^{n+c_j}; \lambda]. \\
\end{align}
$$
这里， $\large u^{n+c_j}(x) = u(t^n+c_j \Delta t, x)$，对于 $j=1,\dots,q$。根据参数$\{a_{ij}, b_j, c_j\}$的选择，这种通用形式封装了隐式和显式时间步进方案。上述方程式可等效表示为
$$
\begin{align}
& \large u^n = u^n_i, i=1,\dots,q, \\
& \large u^n = u^{n+1}_i, i=1,\dots,q, \\
& \large u^n_i = u^{n+c_i} + \Delta t \sum^{q}_{j=1} a_{ij} \cal N[u^{n+c_j}; \lambda], i=1,\dots,q, \\
& \large u^{n+1}_i = u^{n+c_i} + \Delta t \sum^{q}_{j=1} (a_{ij} - b_j) \cal N[u^{n+c_j}; \lambda],
                     i=1,\dots,q. \\
\end{align}
$$
我们将一个多输出神经网络置于 $\large [u^{n+c_1}(x), \dots, u^{n+c_q}(x)]$。
这一先验假设与上述方程一起产生了一个以 $x$ 为输入和输出的物理信息神经网络 $\large [u^n_1(x), \dots, u^n_q(x), u^n_{q+1}(x)]$ 。

给定系统在时间 $t^n$ 和 $t^{n+1}$ 分别在两个不同的时间快照 $\{\bf x^n, u^n\}$ 和 $\{\bf x^{n+1}, u^{n+1}\}$ 处的噪声测量，可通过最小化误差平方和来训练神经网络的共享参数以及微分算子的参数$λ$：
$$
\large
\begin{align}
 SSE  &= SSE_n + SSE_{n+1},\\
SSE_n &= \sum^{q}_{j=1} \sum^{N_n}_{i=1} |u^n_j(x^{n,i}) - u^{n,i}|^2,\\
SSE_{n+1} &= \sum^{q}_{j=1} \sum^{N_{n+1}}_{i=1} |u^{n+1}_j(x^{n+1,i}) - u^{n+1,i}|^2,\\
\end{align}
$$
这里， $\large x^n=\{x^{n,i}\}^{N_n}_{i=1}, u^n=\{u^{n,i}\}^{N_n}_{i=1}, x^{n+1}=\{x^{n+1,i}\}^{N_{n+1}}_{i=1}, and \  u^{n+1}=\{u^{n+1,i}\}^{N_{n+1}}_{i=1}$。

##### 示例5(Korteweg–de Vries方程)

我们的最后一个例子旨在强调所提出的框架处理涉及高阶导数的偏微分方程的能力。在这里，我们考虑在浅水表面上的波浪的数学模型：[Korteweg de Vries(KdV)方程](https://en.wikipedia.org/wiki/Korteweg–de_Vries_equation)。KdV方程如下所示：
$$
\large u_t + \lambda_1 uu_x + \lambda_2 u_{xxx} = 0,
$$
其中 $(λ_1,λ_2)$ 为未知参数。对于KdV方程，非线性算子由下式给出：
$$
\large \cal N[u^{n+c_j}] = \lambda_1 u^{n+c_j}u^{n+c_j}_x - \lambda_2 u^{n+c_j}_{xxx},
$$
通过最小化上述误差平方和，可以学习神经网络的共享参数以及KdV方程的参数 $λ = (λ_1,λ_2)$ 。
下图总结了该实验的结果。

![img](https://maziarraissi.github.io/assets/img/KdV.png)

> *KdV方程*：**顶部**：解以及两个训练快照的时间位置。**中间**：与顶部面板中垂直虚线所示的两个时间快照相对应的训练数据和精确解。**底部**：修正偏微分方程和已识别的方程。



### 结论 Conclusion

虽然提出了一系列有希望的结果，但读者可能会同意，这篇由两部分组成的论文所提出的问题多于所回答的问题。在更广泛的背景下，在寻求对这些工具的进一步理解的过程中，我们认为这项工作提倡机器学习和经典计算物理之间富有成效的协同作用，这有可能丰富这两个领域，并带来高影响力的发展。



### 致谢 Acknowledgements

这项工作得到了[国防部高级研究计划局设备批准N66001-15-2-4055和AFOSR授权FA9550-17-1-0013](the DARPA EQUiPS grant N66001-15-2-4055 and the AFOSR grant FA9550-17-1-0013)的支持。所有数据和代码都可以在[GitHub](https://github.com/maziarraissi/PINNs)上公开获取。



### 引用 Citation

```Text
@article{raissi2019physics,
  title={Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations},
  author={Raissi, Maziar and Perdikaris, Paris and Karniadakis, George E},
  journal={Journal of Computational Physics},
  volume={378},
  pages={686--707},
  year={2019},
  publisher={Elsevier}
}

@article{raissi2017physicsI,
  title={Physics Informed Deep Learning (Part I): Data-driven Solutions of Nonlinear Partial Differential Equations},
  author={Raissi, Maziar and Perdikaris, Paris and Karniadakis, George Em},
  journal={arXiv preprint arXiv:1711.10561},
  year={2017}
}

@article{raissi2017physicsII,
  title={Physics Informed Deep Learning (Part II): Data-driven Discovery of Nonlinear Partial Differential Equations},
  author={Raissi, Maziar and Perdikaris, Paris and Karniadakis, George Em},
  journal={arXiv preprint arXiv:1711.10566},
  year={2017}
}
```

[PINNs](https://github.com/maziarraissi/PINNs)由[maziarraissi](https://github.com/maziarraissi)维护。此页面由[GitHub Pages](https://pages.github.com/)生成。










