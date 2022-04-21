# 1 输运方程(Transport equation)

下面记 $u = u(x,t), x=(x_1,\cdots,x_n) \in \mathbb{R}^n, t \geq 0$ ，$x$ 是空间中的一个点，$t$ 是时间。$Du = D_x u = (u_{x_1},\cdots,u_{x_n})$是代表 $u$ 关于空间变量 $x$ 的梯度。$u_t$ 代表 $u$ 关于时间的偏导数。

**定义1.1** 偏微分方程
$$
\large u_t + b \cdot D_u = 0,(x,t) \in \mathbb{R}^n \times (0,\infty)    \tag{1.1}
$$
是**输运方程(transport equation)**，这里 $b=(b_1,\cdots,b_n) \in \mathbb{R}^n$ 是个固定的向量。

为了解这个PDE，我们现在就不妨设 $u$ 有某个光滑的解然后再尝试计算它。首先注意到方程(1.1)表明 $u$ 的某个特定方向的导数为0, 我们固定任意的点 $(x,t) \in \mathbb{R}^n \times (0,\infty)$ 并定义 $\large z(s) \triangleq u(x+sb,t+s) s \in \mathbb{R}$，

以下都记 $\frac {d} {ds} = \dot{} $，利用方程(1.1)计算得到 $\large \dot z(s) = Du(x+sb,t+s) \sdot b + u_t(x+sb,t+s) = 0$。

因此 $z(\sdot{})$ 是关于 $s$ 的常函数，所以对每个点 $(x,t)$，$u$ 在穿过 $(x,t)$ 且方向是 $(b,1) \in \mathbb{R}^{n+1}$ 的直线(记为 $l$ )上是个常数。因此，如果我们知道了直线 $l$ 上任意一个点的 $u$ 值，我们就知道了这条直线 $l$ 的值。

## 1.1 初值问题

为了确定所需要的解, 考虑如下初值问题
$$
\large
\begin{cases}
\begin{aligned}
u_t+b\sdot D_u &= 0, in \ \mathbb{R}^n \times (0,\infty) \\ 
u &= g, on \ \mathbb{R}^n \times \{t=0\}.  \\ 
\end{aligned}
\end{cases} 
\tag{1.2}
$$
这里 $b \in \mathbb{R}^n,g:\mathbb{R}^n \to \mathbb{R}$ 给定, 现在要求出 $u$ 的表达式。

固定了上述 $(x,t)$，穿过 $(x,t)$ 且方向为 $(b,1)$ 的直线可以用参数方程 $(x+sb,t+s) (s \in \mathbb{R})$ 表示，当 $s=-t$ 时，这条直线打在平面$\Gamma := \mathbb{R}^n \times \{t=0\}$ 上，且交点是 $(x-tb,0)$。由于 $u$ 是直线上的常数，且 $u(x-tb,0)=g(x-tb)$，因此
$$
\large u(x,t) = g(x-tb),x \in \mathbb{R}^n,t \geq 0.\tag{1.3}
$$
因此如果方程(1.2)有足够好的解 $u$，则这个解一定形如(1.3)。另一方面，容易判断如果$g \in C^1$，则式(1.3)满足方程(1.2)。

**注:** 如果 $g \not \in C^1$，显然方程(1.2)没有解 $C^1$，则这个解叫做“**弱解**”。不过有时候不光滑甚至是不连续的函数都可以作为PDE的一个弱解。

## 1.2 非齐次初值问题

下面考虑非齐次方程
$$
\large
\begin{cases}
\begin{aligned}
u_t+b \sdot D_u &= f, in \ \mathbb{R}^n \times (0,\infty) \\ 
u &= g, on \ \mathbb{R}^n \times \{t=0\}.  \\ 
\end{aligned}
\end{cases}
\tag{1.4}
$$
受前面过程启发，我们与前面类似，固定 $(x,t) \in \mathbb{R}^{n+1}$，记 $\large z(s) \triangleq u(x+sb,t+s) s \in \mathbb{R}$，则$\large \dot z(s) = Du(x+sb,t+s) \sdot b + u_t(x+sb,t+s) = f(x+sb,t+s)$。

因此
$$
\large
\begin{align*}
u(x,t) - g(x-tb) &= z(0) - z(-t) = \int^0_{-t} \dot{z}(s) ds \\
&= \int^0_{-t} f(x+sb,t+s) ds   \\ 
&= \int^t_{0} f(x+(s-t)b,s) ds,   \\
\end{align*}
$$
因此 $\large u(x,t) = g(x-tb) + \int^t_{0} f(x+(s-t)b,s) ds, x\in \mathbb{R}^n,t \geq 0$是初值问题(1.4)的解。这个方程可以被用来解一维的波动方程。

**注：**注意到我们实际上是通过有效地把PDE转变成ODE最终得到PDE的解，这些步骤是“**特性曲线法**”的一种特例。



# 2 拉普拉斯方程(Laplace's equation)

本小节介绍Laplace方程的定义以及它的基本解。



## 2.1 基本定义

记 $\large x=(x_1,\cdots,x_n) \in \mathbb{R}^n, u=u(x), 拉普拉斯算子 \Delta u = \sum^n_{i=1} u_{x_i x_i} $，Laplace方程又称为“**位势方程**”, 最有用的PDE之一无疑包括Laplace方程
$$
\large \Delta u = 0 \tag{2.1}
$$
以及Poisson方程
$$
\large - \Delta u = f.  \tag{2.2}
$$
在上面两个方程中，$x \in U$ 且未知量为 $\large u:\bar{U} \to \mathbb{R}, u = u(x)$，这里 $u \subset \mathbb{R}^n$ 是开集. 而在第二个方程中 $f:U \to \mathbb{R}$ 也是给定的。

**定义2.1[调和函数]** 假设$u \in C^2$ 且满足 $\Delta u = 0$，则称 $u$ 为调和函数。



## 2.2 基本解

基本解(fundamental solution)的来源: 研究PDE的一个好的方式是去找某个特解. 由于这个PDE是线性的, 所以可以用特解去找更复杂的解. 此外, 为了寻找显然的特解, 通常会把注意力集中在某类具有**对称性**的函数.

**定义2.2** 函数
$$
\large
\Phi(x) = 
\begin{cases}
\begin{aligned}
&-\frac{1}{2\pi} \ln|x|, & n=2\\
&\frac{1}{n(n-2)\alpha(n)}   \frac{1}{|x|^{n-2}}, & n\geq3\\
\end{aligned}
\end{cases}
(x \in \mathbb{R}^n, x \neq 0) \tag{2.3} \\
$$
叫做Laplace方程的**基本解**，这里 $\large \alpha(n) = \mathbb{R}^n中单位球的体积 = \frac{\sqrt{\pi^n}}{ \Gamma(\frac{n}{2}+1)}$。



# 3 热方程(Heat equation)

"热像引力一样穿透宇宙所有物质，辐射整个空间“。                                

​																									——傅立叶《热分析理论》1822

目录：

1. 什么是热传导方程
2. 基本解
3. 齐次+初值不为0的热传导方程的解
4. 非齐次+初值为0的热传导方程的解
5. 非齐次+初值不为0的热传导方程的解

## 3.1 什么是热传导方程

热传导方程(heat equation)
$$
\large u_t - \Delta u = 0,  \tag{3.1}
$$
以及非齐次热传导方程
$$
\large u_t - \Delta u = f.  \tag{3.2}
$$
其中给定合适的初值与边界条件. 这里 $t>0,x\in U \subset \mathbb{R}^n$ 为开集. 不确定的东西是 $\large u:\bar{U} \times[0,\infty) \to \mathbb{R}, u = u(x,t)$，而Laplace算子是关于空间 $x=(x_1,\cdots,x_n)$ 的, 即 $\Delta u = \Delta_x u = \sum^n_{i=1} u_{x_i x_i}$ 在非齐次的方程中 $f:U\times[0,\infty) \to \mathbb{R}$ 是给定的.



## 3.2 基本解

**定义3.1 [热传导方程的基本解]** 定义函数
$$
\large
\Phi(x,t) \triangleq 
\begin{cases}
\begin{aligned}
&\frac{1}{(4\pi t)^{n/2}} e^{-\frac{|x|^2}{4t}}, & x \in \mathbb{R}^n, t > 0\\
&0, & x \in \mathbb{R}^n, t \leq 0\\
\end{aligned}
\end{cases}
\tag{3.3} \\
$$
为热传导方程的基本解。



## 3.3 齐次初值问题

下面用 $\Phi(x,t) $ 来解下面的初值(Cauchy)PDE:
$$
\large
\begin{cases}
\begin{aligned}
u_{t}-\Delta u &= 0, in \ \mathbb{R}^n \times (0,\infty) \\ 
u &= g, on \ \mathbb{R}^n \times \{t=0\}.  \\ 
\end{aligned}
\end{cases} 
\tag{3,4}
$$

仿照Laplace方程的过程, 如果我们作一个卷积
$$
\large
\begin{align*}
u(x,t) &= \int_{\mathbb{R}^n} \Phi(x-y,t)g(y)dy\\
       &= \frac{1}{(4\pi t)^{n/2}} \int_{\mathbb{R}^n} e^{-\frac{|x-y|^2}{4t}}g(y)dy(x\in \mathbb{R}^n,t>0) \\
\tag{3.5}

\end{align*}
$$
**定理1.2 [初值问题的解]** 设 $\large g \in C(\mathbb{R}^n) \cap L^{\infty}(\mathbb{R}^n)$，定义 $u$ 如式(3.5)，则

（1） $\large u\in C^{\infty}(\mathbb{R}^n \times (0,\infty))$，

（2）$\large u_t(x,t)-\Delta u(x,t) = 0, (x\in \mathbb{R}^n,t>0)$，

（3）$\large \lim_{(x,t)\to (x^0,0)} u(x,t) = g(x^0), \forall x^0 \in \mathbb{R}^n, 这里x\in \mathbb{R}^n, t>0$。

## 3.4 非齐次问题

下面考虑非齐次问题
$$
\large
\begin{cases}
\begin{aligned}
u_{t}-\Delta u &= f, in \ \mathbb{R}^n \times (0,\infty) \\ 
u &= 0, on \ \mathbb{R}^n \times \{t=0\}.  \\ 
\end{aligned}
\end{cases} 
\tag{3,6}
$$

$$
\large
\begin{align*}
u(x,t) &= \int^t_0 u(x,t;s)ds \\
       &= \int^t_0 \int_{\mathbb{R}^n} \Phi(x-y,t-s)f(y,s)dyds \\
       &= \int^t_0 \frac{1}{(4\pi (t-s))^{n/2}} \int_{\mathbb{R}^n} e^{-\frac{|x-y|^2}{4(t-s)}}f(y,s)dyds,\\
\tag{3.7}
\end{align*}
$$

**定理1.3 [非齐次问题的解]** 定义 $u$ 如式(3.7), 则

（1） $\large u\in C^2_1(\mathbb{R}^n \times (0,\infty))$，

（2）$\large u_t(x,t)-\Delta u(x,t) = f(x,t), (x\in \mathbb{R}^n,t>0)$，

（3）$\large \lim_{(x,t)\to (x^0,0)} u(x,t) = 0, \forall x^0 \in \mathbb{R}^n, 这里x\in \mathbb{R}^n, t>0$。



# 4 波动方程(Wave equation)

下面我们讨论波动方程 $u_{tt}-\Delta u = 0$ 和非齐次形式 $u_{tt}-\Delta u = f$ 的解, 其中给定一定的初始和边界条件. 这里 $t>0,x\in U,U\subset \mathbb{R}^n$。要解的东西是 $\large u:\bar{U} \times[0,\infty) \to \mathbb{R}, u = u(x,t)$。这里拉普拉斯算子是关于空间变量 $x = (x_1,\cdots,x_n)$ 的。另外 $f:U \times[0,\infty) \to \mathbb{R}$ 给定. 通常我们也记 $\Box u := u_{tt} - \Delta u$。



## 4.1 1维情形: d'Alembert公式

**定理1.1 [d'Alembert公式] [一维波动方程的解]** 设 $\large g \in C^2(\mathbb{R}), h \in C^1(\mathbb{R})$，定义

$\large u(x,t) = \frac{1}{2} [g(x+t)+g(x-t)] + \frac{1}{2} \int^{x+t}_{x-t}h(y)dy$，

则：

（1） $\large u\in C^2(\mathbb{R} \times [0,\infty))$，

（2）$\large u_{tt} - u_{xx} = 0, (x,t) \in \mathbb{R}\times (0,\infty)$，

（3）$\large \lim_{(x,t)\to (x^0,0^+)} u(x,t) = g(x^0), \lim_{(x,t)\to (x^0,0)} u_t(x,t) = h(x^0), \forall x^0 \in \mathbb{R}$。



## 4.2 3维情形: Kirchhoff公式

对 $x\in \mathbb{R}^3, t>0,$ 
$$
\large
\begin{align*}
u(x,t) &= \frac{1}{n\alpha(n)t^{n-1}} \int_{\partial B(x,t)} th(y)+g(y)+Dg(y)\cdot (y-x)dS(y)。\\
\tag{3.8}
\end{align*}
$$
这就是初值问题(1)的三维情形的**Kirchhoff公式**。

## 4.3 2维情形: Poisson公式

对 $x\in \mathbb{R}^2, t>0,$ 
$$
\large
\begin{align*}
u(x,t) &= \frac{1}{2\pi t} \int_{B(x,t)} \frac{g(y)+th(y)+Dg(y)\cdot (y-x)}{(t^2-|y-x|^2)^{1/2}} dy,\\
\tag{3.9}
\end{align*}
$$
这就是二维情形下问题(1)的Poisson公式。

# 参考文献

L.C. Evans《Partial Differential Equations》2nd Ed, Berkeley

偏微分方程笔记(1)——输运方程 https://zhuanlan.zhihu.com/p/74576825 

偏微分方程笔记(2)——Laplace(位势)方程的基本解 https://zhuanlan.zhihu.com/p/75732103 

偏微分方程笔记(5)——热传导方程的基本解 https://zhuanlan.zhihu.com/p/80214720 

偏微分方程笔记(8)——波动方程的低维解 https://zhuanlan.zhihu.com/p/84755975


