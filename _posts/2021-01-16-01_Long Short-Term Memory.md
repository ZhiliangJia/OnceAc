---
layout: post
title: 长短时记忆网络
date: 2021-01-16
author: Zhiliang 
tags: [Machine Learning]
toc: true
mathjax: true
---

长短期记忆网络（LSTM，Long Short-Term Memory）是一种时间循环神经网络，是为了解决一般的RNN（循环神经网络）存在的长期依赖问题而专门设计出来的，所有的RNN都具有一种重复神经网络模块的链式形式。在标准RNN中，这个重复的结构模块只有一个非常简单的结构，例如一个tanh层。

<!-- more -->

决定了上一时刻的单元状态$\mathbf{c}_{t-1}$有多少保留到当前时刻$\mathbf{c}_t$；另一个是**输入门（input gate）**，它决定了当前时刻网络的输入$\mathbf{x}_t$有多少保存到单元状态$\mathbf{c}_t$。LSTM用**输出门（output gate）**来控制单元状态$\mathbf{x}_t$有多少输出到LSTM的当前输出值$\mathbf{h}_t$。

我们先来看一下遗忘门：

$$

\mathbf{f}_t=\sigma(W_f\cdot[\mathbf{h}_{t-1},\mathbf{x}_t]+\mathbf{b}_f)\tag{2.1}

$$



上式中，$W_f$是遗忘门的权重矩阵，$[\mathbf{h}_{t-1},\mathbf{x}_t]$表示把两个向量连接成一个更长的向量，$\mathbf{b}_f$是遗忘门的偏置项，$\sigma$是sigmoid函数。如果输入的维度是$d_x$，隐藏层的维度是$d_h$，单元状态的维度是$d_c$（通常$d_c=d_h$），则遗忘门的权重矩阵$W_f$维度是$d_c\times (d_h+d_x)$。事实上，权重矩阵$W_f$都是两个矩阵拼接而成的：一个是$W_{fh}$，它对应着输入项$h_{t-1}$，其维度为$d_c\times d_h$；一个是$W_{fx}$，它对应着输入项$\mathbf{x}_t$，其维度为$d_c\times d_x$。可以写为：

$$

\begin{align}
\begin{bmatrix}W_f\end{bmatrix}\begin{bmatrix}\mathbf{h}_{t-1}\\
\mathbf{x}_t\end{bmatrix}&=
\begin{bmatrix}W_{fh}&W_{fx}\end{bmatrix}\begin{bmatrix}\mathbf{h}_{t-1}\\
\mathbf{x}_t\end{bmatrix}\\
&=W_{fh}\mathbf{h}_{t-1}+W_{fx}\mathbf{x}_t
\end{align}

$$

下图显示了遗忘门的计算：

![](https://gitee.com/zhiliangj/Typora_Img/raw/master/2256672-c7f7ca0aa64b562f.png)

接下来看看输入门：

$$

\mathbf{i}_t=\sigma(W_i\cdot[\mathbf{h}_{t-1},\mathbf{x}_t]+\mathbf{b}_i)\tag{2.2}

$$

上式中，$W_i$是输入门的权重矩阵，$b_i$是输入门的偏置项。下图表示了输入门的计算：

![](https://gitee.com/zhiliangj/Typora_Img/raw/master/2256672-89529fa23d9c8a7d.png)

接下来，我们计算用于描述当前输入的单元状态$\mathbf{\tilde{c}}_t$，它<u>是根据上一次的输出和本次输入来计算</u>的：

$$

\mathbf{\tilde{c}}_t=\tanh(W_c\cdot[\mathbf{h}_{t-1},\mathbf{x}_t]+\mathbf{b}_c)\tag{2.3}

$$

下图是$\mathbf{\tilde{c}}_t$的计算：

![](https://gitee.com/zhiliangj/Typora_Img/raw/master/2256672-73a0246cafc1d10d.png)

现在，我们计算当前时刻的单元状态$\mathbf{c}_t$。它是由上一次的单元状态$\mathbf{c}_{t-1}$按元素乘以遗忘门$f_t$，再用当前输入的单元状态$\mathbf{\tilde{c}}_t$按元素乘以输入门$i_t$，再将两个积加和产生的：

$$

\mathbf{c}_t=f_t\circ{\mathbf{c}_{t-1}}+i_t\circ{\mathbf{\tilde{c}}_t}\tag{2.4}

$$

符号$\circ$表示**按元素乘**。下图是的计算：

![](https://gitee.com/zhiliangj/Typora_Img/raw/master/2256672-5c766f3d734334b1.png)

这样，我们就把LSTM关于当前的记忆$\mathbf{\tilde{c}}_t$和长期的记忆$\mathbf{c}_{t-1}$组合在一起，形成了新的单元状态$\mathbf{c}_{t}$。由于遗忘门的控制，它可以保存很久很久之前的信息，由于输入门的控制，它又可以避免当前无关紧要的内容进入记忆。下面，我们要看看输出门，它控制了长期记忆对当前输出的影响：

$$

\mathbf{o}_t=\sigma(W_o\cdot[\mathbf{h}_{t-1},\mathbf{x}_t]+\mathbf{b}_o)\tag{2.5}

$$

下图表示输出门的计算：

![](https://gitee.com/zhiliangj/Typora_Img/raw/master/2256672-fd4d91d1b68b3759.png)

LSTM最终的输出，是由输出门和单元状态共同确定的：

$$

\mathbf{h}_t=\mathbf{o}_t\circ \tanh(\mathbf{c}_t)\tag{2.6}

$$

下图表示LSTM最终输出的计算：

![](https://gitee.com/zhiliangj/Typora_Img/raw/master/2256672-7ea82e4f1ac6cd75.png)

**式2.1**到**式2.6**就是LSTM前向计算的全部公式。至此，我们就把LSTM前向计算讲完了。

# 长短时记忆网络的训练

熟悉我们这个系列文章的同学都清楚，训练部分往往比前向计算部分复杂多了。LSTM的前向计算都这么复杂，那么，可想而知，它的训练算法一定是非常非常复杂的。现在只有做几次深呼吸，再一头扎进公式海洋吧。

## LSTM训练算法框架

LSTM的训练算法仍然是反向传播算法，对于这个算法，我们已经非常熟悉了。主要有下面三个步骤：

1. 前向计算每个神经元的输出值，对于LSTM来说，即$\mathbf{f}_t$、$\mathbf{i}_t$、$\mathbf{c}_t$、$\mathbf{o}_t$、$\mathbf{h}_t$五个向量的值。计算方法已经在上一节中描述过了。
2. 反向计算每个神经元的**误差项**值。与**循环神经网络**一样，LSTM误差项的反向传播也是包括两个方向：一个是沿时间的反向传播，即从当前t时刻开始，计算每个时刻的误差项；一个是将误差项向上一层传播。
3. 根据相应的误差项，计算每个权重的梯度。

## 关于公式和符号的说明

首先，我们对推导中用到的一些公式、符号做一下必要的说明。

接下来的推导中，我们设定gate的激活函数为sigmoid函数，输出的激活函数为tanh函数。他们的导数分别为：

$$

\begin{align}
\sigma(z)&=y=\frac{1}{1+e^{-z}}\\
\sigma'(z)&=y(1-y)\\
\tanh(z)&=y=\frac{e^z-e^{-z}}{e^z+e^{-z}}\\
\tanh'(z)&=1-y^2
\end{align}

$$

从上面可以看出，sigmoid和tanh函数的导数都是原函数的函数。这样，我们一旦计算原函数的值，就可以用它来计算出导数的值。

LSTM需要学习的参数共有8组，分别是：遗忘门的权重矩阵$W_f$和偏置项$\mathbf{b}\_f$、输入门的权重矩阵$W_i$和偏置项$\mathbf{b}\_i$、输出门的权重矩阵$W_o$和偏置项$\mathbf{b}\_o$，以及计算单元状态的权重矩阵$W\_c$和偏置项$\mathbf{b}\_c$。因为权重矩阵的两部分在反向传播中使用不同的公式，因此在后续的推导中，权重矩阵$W_{f}$、$W_{i}$、$W_{o}$、$W_{c}$都将被写为分开的两个矩阵：$W_{fh}$、$W_{fx}$、$W_{ih}$、$W_{ix}$、$W_{oh}$、$W_{ox}$、$W_{ch}$、$W_{cx}$。

我们解释一下按元素乘$\circ$符号。当$\circ$作用于两个**向量**时，运算如下：

$$

\mathbf{a}\circ\mathbf{b}=\begin{bmatrix}
a_1\\a_2\\a_3\\...\\a_n
\end{bmatrix}\circ\begin{bmatrix}
b_1\\b_2\\b_3\\...\\b_n
\end{bmatrix}=\begin{bmatrix}
a_1b_1\\a_2b_2\\a_3b_3\\...\\a_nb_n
\end{bmatrix}

$$

当$\circ$作用于一个**向量**和一个**矩阵**时，运算如下：

$$

\begin{align}
\mathbf{a}\circ X&=\begin{bmatrix}
a_1\\a_2\\a_3\\...\\a_n
\end{bmatrix}\circ\begin{bmatrix}
x_{11} & x_{12} & x_{13} & ... & x_{1n}\\
x_{21} & x_{22} & x_{23} & ... & x_{2n}\\
x_{31} & x_{32} & x_{33} & ... & x_{3n}\\
& & ...\\
x_{n1} & x_{n2} & x_{n3} & ... & x_{nn}\\
\end{bmatrix}\\
&=\begin{bmatrix}
a_1x_{11} & a_1x_{12} & a_1x_{13} & ... & a_1x_{1n}\\
a_2x_{21} & a_2x_{22} & a_2x_{23} & ... & a_2x_{2n}\\
a_3x_{31} & a_3x_{32} & a_3x_{33} & ... & a_3x_{3n}\\
& & ...\\
a_nx_{n1} & a_nx_{n2} & a_nx_{n3} & ... & a_nx_{nn}\\
\end{bmatrix}
\end{align}

$$

当$\circ$作用于两个**矩阵**时，<u>两个矩阵对应位置的元素相乘</u>。按元素乘可以在某些情况下简化矩阵和向量运算。例如，当一个对角矩阵右乘一个矩阵时，相当于用对角矩阵的对角线组成的向量按元素乘那个矩阵：

$$

diag[\mathbf{a}]X=\mathbf{a}\circ X

$$

当一个行向量右乘一个对角矩阵时，相当于这个行向量按元素乘那个矩阵对角线组成的向量：

$$

\mathbf{a}^Tdiag[\mathbf{b}]=\mathbf{a}\circ\mathbf{b}

$$

上面这两点，在我们后续推导中会多次用到。

在t时刻，LSTM的输出值为$\mathbf{h}_t$。我们定义t时刻的误差项$\delta_t$为：

$$

\delta_t\overset{def}{=}\frac{\partial{E}}{\partial{\mathbf{h}_t}}

$$

注意，和前面几篇文章不同，我们这里假设误差项是损失函数对输出值的导数，而不是对加权输入$net_t^l$的导数。因为LSTM有四个加权输入，分别对应$\mathbf{f}_t$、$\mathbf{i}_t$、$\mathbf{c}_t$、$\mathbf{o}_t$，我们希望往上一层传递一个误差项而不是四个。但我们仍然需要定义出这四个加权输入，以及他们对应的误差项。

$$

\begin{align}
\mathbf{net}_{f,t}&=W_f[\mathbf{h}_{t-1},\mathbf{x}_t]+\mathbf{b}_f\\
&=W_{fh}\mathbf{h}_{t-1}+W_{fx}\mathbf{x}_t+\mathbf{b}_f\\
\mathbf{net}_{i,t}&=W_i[\mathbf{h}_{t-1},\mathbf{x}_t]+\mathbf{b}_i\\
&=W_{ih}\mathbf{h}_{t-1}+W_{ix}\mathbf{x}_t+\mathbf{b}_i\\
\mathbf{net}_{\tilde{c},t}&=W_c[\mathbf{h}_{t-1},\mathbf{x}_t]+\mathbf{b}_c\\
&=W_{ch}\mathbf{h}_{t-1}+W_{cx}\mathbf{x}_t+\mathbf{b}_c\\
\mathbf{net}_{o,t}&=W_o[\mathbf{h}_{t-1},\mathbf{x}_t]+\mathbf{b}_o\\
&=W_{oh}\mathbf{h}_{t-1}+W_{ox}\mathbf{x}_t+\mathbf{b}_o\\
\delta_{f,t}&\overset{def}{=}\frac{\partial{E}}{\partial{\mathbf{net}_{f,t}}}\\
\delta_{i,t}&\overset{def}{=}\frac{\partial{E}}{\partial{\mathbf{net}_{i,t}}}\\
\delta_{\tilde{c},t}&\overset{def}{=}\frac{\partial{E}}{\partial{\mathbf{net}_{\tilde{c},t}}}\\
\delta_{o,t}&\overset{def}{=}\frac{\partial{E}}{\partial{\mathbf{net}_{o,t}}}\\
\end{align}

$$


## 误差项沿时间的反向传递

沿时间反向传递误差项，就是要计算出$t-1$时刻的误差项$\delta_{t-1}$。

$$

\begin{align}
\delta_{t-1}^T&=\frac{\partial{E}}{\partial{\mathbf{h_{t-1}}}}\\
&=\frac{\partial{E}}{\partial{\mathbf{h_t}}}\frac{\partial{\mathbf{h_t}}}{\partial{\mathbf{h_{t-1}}}}\\
&=\delta_{t}^T\frac{\partial{\mathbf{h_t}}}{\partial{\mathbf{h_{t-1}}}}
\end{align}

$$

我们知道，$\frac{\partial{\mathbf{h_t}}}{\partial{\mathbf{h_{t-1}}}}$是一个Jacobian矩阵。如果隐藏层h的维度是N的话，那么它就是一个矩阵。为了求出它，我们列出的计算公式，即前面的**式2.6**和**式2.4**：

$$

\begin{align}
\mathbf{h}_t&=\mathbf{o}_t\circ \tanh(\mathbf{c}_t)\\
\mathbf{c}_t&=\mathbf{f}_t\circ\mathbf{c}_{t-1}+\mathbf{i}_t\circ\mathbf{\tilde{c}}_t
\end{align}

$$

显然，$\mathbf{o}_t$、$\mathbf{f}_t$、$\mathbf{i}_t$、$\mathbf{\tilde{c}}_t$都是$\mathbf{h}_{t-1}$的函数，那么利用全导数公式可得：

$$

\begin{align}
\delta_t^T\frac{\partial{\mathbf{h_t}}}{\partial{\mathbf{h_{t-1}}}}&=\delta_t^T\frac{\partial{\mathbf{h_t}}}{\partial{\mathbf{o}_t}}\frac{\partial{\mathbf{o}_t}}{\partial{\mathbf{net}_{o,t}}}\frac{\partial{\mathbf{net}_{o,t}}}{\partial{\mathbf{h_{t-1}}}}
+\delta_t^T\frac{\partial{\mathbf{h_t}}}{\partial{\mathbf{c}_t}}\frac{\partial{\mathbf{c}_t}}{\partial{\mathbf{f_{t}}}}\frac{\partial{\mathbf{f}_t}}{\partial{\mathbf{net}_{f,t}}}\frac{\partial{\mathbf{net}_{f,t}}}{\partial{\mathbf{h_{t-1}}}}
+\delta_t^T\frac{\partial{\mathbf{h_t}}}{\partial{\mathbf{c}_t}}\frac{\partial{\mathbf{c}_t}}{\partial{\mathbf{i_{t}}}}\frac{\partial{\mathbf{i}_t}}{\partial{\mathbf{net}_{i,t}}}\frac{\partial{\mathbf{net}_{i,t}}}{\partial{\mathbf{h_{t-1}}}}
+\delta_t^T\frac{\partial{\mathbf{h_t}}}{\partial{\mathbf{c}_t}}\frac{\partial{\mathbf{c}_t}}{\partial{\mathbf{\tilde{c}}_{t}}}\frac{\partial{\mathbf{\tilde{c}}_t}}{\partial{\mathbf{net}_{\tilde{c},t}}}\frac{\partial{\mathbf{net}_{\tilde{c},t}}}{\partial{\mathbf{h_{t-1}}}}\\
&=\delta_{o,t}^T\frac{\partial{\mathbf{net}_{o,t}}}{\partial{\mathbf{h_{t-1}}}}
+\delta_{f,t}^T\frac{\partial{\mathbf{net}_{f,t}}}{\partial{\mathbf{h_{t-1}}}}
+\delta_{i,t}^T\frac{\partial{\mathbf{net}_{i,t}}}{\partial{\mathbf{h_{t-1}}}}
+\delta_{\tilde{c},t}^T\frac{\partial{\mathbf{net}_{\tilde{c},t}}}{\partial{\mathbf{h_{t-1}}}}
\end{align}\tag{3.1}

$$

下面，我们要把**式3.1**中的每个偏导数都求出来。根据**式2.6**，我们可以求出：

$$

\begin{align}
\frac{\partial{\mathbf{h_t}}}{\partial{\mathbf{o}_t}}&=diag[\tanh(\mathbf{c}_t)]\\
\frac{\partial{\mathbf{h_t}}}{\partial{\mathbf{c}_t}}&=diag[\mathbf{o}_t\circ(1-\tanh(\mathbf{c}_t)^2)]
\end{align}

$$

根据**式2.4**，我们可以求出：

$$

\begin{align}
\frac{\partial{\mathbf{c}_t}}{\partial{\mathbf{f_{t}}}}&=diag[\mathbf{c}_{t-1}]\\
\frac{\partial{\mathbf{c}_t}}{\partial{\mathbf{i_{t}}}}&=diag[\mathbf{\tilde{c}}_t]\\
\frac{\partial{\mathbf{c}_t}}{\partial{\mathbf{\tilde{c}_{t}}}}&=diag[\mathbf{i}_t]\\
\end{align}

$$

因为：

$$

\begin{align}
\mathbf{o}_t&=\sigma(\mathbf{net}_{o,t})\\
\mathbf{net}_{o,t}&=W_{oh}\mathbf{h}_{t-1}+W_{ox}\mathbf{x}_t+\mathbf{b}_o\\\\
\mathbf{f}_t&=\sigma(\mathbf{net}_{f,t})\\
\mathbf{net}_{f,t}&=W_{fh}\mathbf{h}_{t-1}+W_{fx}\mathbf{x}_t+\mathbf{b}_f\\\\
\mathbf{i}_t&=\sigma(\mathbf{net}_{i,t})\\
\mathbf{net}_{i,t}&=W_{ih}\mathbf{h}_{t-1}+W_{ix}\mathbf{x}_t+\mathbf{b}_i\\\\
\mathbf{\tilde{c}}_t&=\tanh(\mathbf{net}_{\tilde{c},t})\\
\mathbf{net}_{\tilde{c},t}&=W_{ch}\mathbf{h}_{t-1}+W_{cx}\mathbf{x}_t+\mathbf{b}_c\\
\end{align}

$$

我们很容易得出：

$$

\begin{align}
\frac{\partial{\mathbf{o}_t}}{\partial{\mathbf{net}_{o,t}}}&=diag[\mathbf{o}_t\circ(1-\mathbf{o}_t)]\\
\frac{\partial{\mathbf{net}_{o,t}}}{\partial{\mathbf{h_{t-1}}}}&=W_{oh}\\
\frac{\partial{\mathbf{f}_t}}{\partial{\mathbf{net}_{f,t}}}&=diag[\mathbf{f}_t\circ(1-\mathbf{f}_t)]\\
\frac{\partial{\mathbf{net}_{f,t}}}{\partial{\mathbf{h}_{t-1}}}&=W_{fh}\\
\frac{\partial{\mathbf{i}_t}}{\partial{\mathbf{net}_{i,t}}}&=diag[\mathbf{i}_t\circ(1-\mathbf{i}_t)]\\
\frac{\partial{\mathbf{net}_{i,t}}}{\partial{\mathbf{h}_{t-1}}}&=W_{ih}\\
\frac{\partial{\mathbf{\tilde{c}}_t}}{\partial{\mathbf{net}_{\tilde{c},t}}}&=diag[1-\mathbf{\tilde{c}}_t^2]\\
\frac{\partial{\mathbf{net}_{\tilde{c},t}}}{\partial{\mathbf{h}_{t-1}}}&=W_{ch}
\end{align}

$$

将上述偏导数带入到**式3.1**，我们得到：

$$

\begin{align}
\delta_{t-1}&=\delta_{o,t}^T\frac{\partial{\mathbf{net}_{o,t}}}{\partial{\mathbf{h_{t-1}}}}
+\delta_{f,t}^T\frac{\partial{\mathbf{net}_{f,t}}}{\partial{\mathbf{h_{t-1}}}}
+\delta_{i,t}^T\frac{\partial{\mathbf{net}_{i,t}}}{\partial{\mathbf{h_{t-1}}}}
+\delta_{\tilde{c},t}^T\frac{\partial{\mathbf{net}_{\tilde{c},t}}}{\partial{\mathbf{h_{t-1}}}}\\
&=\delta_{o,t}^T W_{oh}
+\delta_{f,t}^TW_{fh}
+\delta_{i,t}^TW_{ih}
+\delta_{\tilde{c},t}^TW_{ch}\\
\end{align}\tag{3.2}

$$

根据$\delta_{o,t}$、$\delta_{f,t}$、$\delta_{i,t}$、$\delta_{\tilde{c},t}$的定义，可知：

$$

\begin{align}
\delta_{o,t}^T&=\delta_t^T\circ\tanh(\mathbf{c}_t)\circ\mathbf{o}_t\circ(1-\mathbf{o}_t)\tag{3.3}\\
\delta_{f,t}^T&=\delta_t^T\circ\mathbf{o}_t\circ(1-\tanh(\mathbf{c}_t)^2)\circ\mathbf{c}_{t-1}\circ\mathbf{f}_t\circ(1-\mathbf{f}_t)\tag{3.4}\\
\delta_{i,t}^T&=\delta_t^T\circ\mathbf{o}_t\circ(1-\tanh(\mathbf{c}_t)^2)\circ\mathbf{\tilde{c}}_t\circ\mathbf{i}_t\circ(1-\mathbf{i}_t)\tag{3.5}\\
\delta_{\tilde{c},t}^T&=\delta_t^T\circ\mathbf{o}_t\circ(1-\tanh(\mathbf{c}_t)^2)\circ\mathbf{i}_t\circ(1-\mathbf{\tilde{c}}^2)\tag{3.6}\\
\end{align}

$$

**式3.2**到**式3.6**就是将误差沿时间反向传播一个时刻的公式。有了它，我们可以写出将误差项向前传递到任意k时刻的公式：

$$

\delta_k^T=\prod_{j=k}^{t-1}\delta_{o,j}^TW_{oh}
+\delta_{f,j}^TW_{fh}
+\delta_{i,j}^TW_{ih}
+\delta_{\tilde{c},j}^TW_{ch}\tag{3.7}

$$


## 将误差项传递到上一层

我们假设当前为第$l$层，定义$l-1$层的误差项是误差函数对$l-1$层**加权输入**的导数，即：

$$

\delta_t^{l-1}\overset{def}{=}\frac{\partial{E}}{\mathbf{net}_t^{l-1}}

$$

本次LSTM的输入$x_t$由下面的公式计算：

$$

\mathbf{x}_t^l=f^{l-1}(\mathbf{net}_t^{l-1})

$$

上式中，$f^{l-1}$表示第$l-1$层的**激活函数**。

因为$\mathbf{net}_{f,t}^l$、$\mathbf{net}_{i,t}^l$、$\mathbf{net}_{\tilde{c},t}^l$、$\mathbf{net}_{o,t}^l$都是$\mathbf{x}_t$的函数，$\mathbf{x}_t$又是$\mathbf{net}_t^{l-1}$的函数，因此，要求出$E$对$\mathbf{net}_t^{l-1}$的导数，就需要使用全导数公式：

$$

\begin{align}
\frac{\partial{E}}{\partial{\mathbf{net}_t^{l-1}}}&=\frac{\partial{E}}{\partial{\mathbf{\mathbf{net}_{f,t}^l}}}\frac{\partial{\mathbf{\mathbf{net}_{f,t}^l}}}{\partial{\mathbf{x}_t^l}}\frac{\partial{\mathbf{x}_t^l}}{\partial{\mathbf{\mathbf{net}_t^{l-1}}}}
+\frac{\partial{E}}{\partial{\mathbf{\mathbf{net}_{i,t}^l}}}\frac{\partial{\mathbf{\mathbf{net}_{i,t}^l}}}{\partial{\mathbf{x}_t^l}}\frac{\partial{\mathbf{x}_t^l}}{\partial{\mathbf{\mathbf{net}_t^{l-1}}}}
+\frac{\partial{E}}{\partial{\mathbf{\mathbf{net}_{\tilde{c},t}^l}}}\frac{\partial{\mathbf{\mathbf{net}_{\tilde{c},t}^l}}}{\partial{\mathbf{x}_t^l}}\frac{\partial{\mathbf{x}_t^l}}{\partial{\mathbf{\mathbf{net}_t^{l-1}}}}
+\frac{\partial{E}}{\partial{\mathbf{\mathbf{net}_{o,t}^l}}}\frac{\partial{\mathbf{\mathbf{net}_{o,t}^l}}}{\partial{\mathbf{x}_t^l}}\frac{\partial{\mathbf{x}_t^l}}{\partial{\mathbf{\mathbf{net}_t^{l-1}}}}\\
&=\delta_{f,t}^TW_{fx}\circ f'(\mathbf{net}_t^{l-1})+\delta_{i,t}^TW_{ix}\circ f'(\mathbf{net}_t^{l-1})+\delta_{\tilde{c},t}^TW_{cx}\circ f'(\mathbf{net}_t^{l-1})+\delta_{o,t}^TW_{ox}\circ f'(\mathbf{net}_t^{l-1})\\
&=(\delta_{f,t}^TW_{fx}+\delta_{i,t}^TW_{ix}+\delta_{\tilde{c},t}^TW_{cx}+\delta_{o,t}^TW_{ox})\circ f'(\mathbf{net}_t^{l-1})
\end{align}\tag{3.8}

$$

**式3.8**就是将误差传递到上一层的公式。

## 权重梯度的计算

对于$W_{fh}$、$W_{ih}$、$W_{ch}$、$W_{oh}$的权重梯度，我们知道它的梯度是各个时刻梯度之和（证明过程请参考文章[零基础入门深度学习(5) - 循环神经网络](https://zybuluo.com/hanbingtao/note/541458)），我们首先求出它们在t时刻的梯度，然后再求出他们最终的梯度。

我们已经求得了误差项$\delta_{o,t}$、$\delta_{f,t}$、$\delta_{i,t}$、$\delta_{\tilde{c},t}$，很容易求出t时刻的$W_{oh}$、的$W_{ih}$、的$W_{fh}$、的$W_{ch}$：

$$

\begin{align}
\frac{\partial{E}}{\partial{W_{oh,t}}}&=\frac{\partial{E}}{\partial{\mathbf{net}_{o,t}}}\frac{\partial{\mathbf{net}_{o,t}}}{\partial{W_{oh,t}}}\\
&=\delta_{o,t}\mathbf{h}_{t-1}^T\\\\
\frac{\partial{E}}{\partial{W_{fh,t}}}&=\frac{\partial{E}}{\partial{\mathbf{net}_{f,t}}}\frac{\partial{\mathbf{net}_{f,t}}}{\partial{W_{fh,t}}}\\
&=\delta_{f,t}\mathbf{h}_{t-1}^T\\\\
\frac{\partial{E}}{\partial{W_{ih,t}}}&=\frac{\partial{E}}{\partial{\mathbf{net}_{i,t}}}\frac{\partial{\mathbf{net}_{i,t}}}{\partial{W_{ih,t}}}\\
&=\delta_{i,t}\mathbf{h}_{t-1}^T\\\\
\frac{\partial{E}}{\partial{W_{ch,t}}}&=\frac{\partial{E}}{\partial{\mathbf{net}_{\tilde{c},t}}}\frac{\partial{\mathbf{net}_{\tilde{c},t}}}{\partial{W_{ch,t}}}\\
&=\delta_{\tilde{c},t}\mathbf{h}_{t-1}^T\\
\end{align}

$$

将各个时刻的梯度加在一起，就能得到最终的梯度：

$$

\begin{align}
\frac{\partial{E}}{\partial{W_{oh}}}&=\sum_{j=1}^t\delta_{o,j}\mathbf{h}_{j-1}^T\\
\frac{\partial{E}}{\partial{W_{fh}}}&=\sum_{j=1}^t\delta_{f,j}\mathbf{h}_{j-1}^T\\
\frac{\partial{E}}{\partial{W_{ih}}}&=\sum_{j=1}^t\delta_{i,j}\mathbf{h}_{j-1}^T\\
\frac{\partial{E}}{\partial{W_{ch}}}&=\sum_{j=1}^t\delta_{\tilde{c},j}\mathbf{h}_{j-1}^T\\
\end{align}

$$

对于偏置项$\mathbf{b}_f$、$\mathbf{b}_i$、$\mathbf{b}_c$、$\mathbf{b}_o$的梯度，也是将各个时刻的梯度加在一起。下面是各个时刻的偏置项梯度：

$$

\begin{align}
\frac{\partial{E}}{\partial{\mathbf{b}_{o,t}}}&=\frac{\partial{E}}{\partial{\mathbf{net}_{o,t}}}\frac{\partial{\mathbf{net}_{o,t}}}{\partial{\mathbf{b}_{o,t}}}\\
&=\delta_{o,t}\\\\
\frac{\partial{E}}{\partial{\mathbf{b}_{f,t}}}&=\frac{\partial{E}}{\partial{\mathbf{net}_{f,t}}}\frac{\partial{\mathbf{net}_{f,t}}}{\partial{\mathbf{b}_{f,t}}}\\
&=\delta_{f,t}\\\\
\frac{\partial{E}}{\partial{\mathbf{b}_{i,t}}}&=\frac{\partial{E}}{\partial{\mathbf{net}_{i,t}}}\frac{\partial{\mathbf{net}_{i,t}}}{\partial{\mathbf{b}_{i,t}}}\\
&=\delta_{i,t}\\\\
\frac{\partial{E}}{\partial{\mathbf{b}_{c,t}}}&=\frac{\partial{E}}{\partial{\mathbf{net}_{\tilde{c},t}}}\frac{\partial{\mathbf{net}_{\tilde{c},t}}}{\partial{\mathbf{b}_{c,t}}}\\
&=\delta_{\tilde{c},t}\\
\end{align}

$$

下面是最终的偏置项梯度，即将各个时刻的偏置项梯度加在一起：

$$

\begin{align}
\frac{\partial{E}}{\partial{\mathbf{b}_o}}&=\sum_{j=1}^t\delta_{o,j}\\
\frac{\partial{E}}{\partial{\mathbf{b}_i}}&=\sum_{j=1}^t\delta_{i,j}\\
\frac{\partial{E}}{\partial{\mathbf{b}_f}}&=\sum_{j=1}^t\delta_{f,j}\\
\frac{\partial{E}}{\partial{\mathbf{b}_c}}&=\sum_{j=1}^t\delta_{\tilde{c},j}\\
\end{align}

$$

对于$W_{fx}$、$W_{ix}$、$W_{cx}$、$W_{ox}$的权重梯度，只需要根据相应的误差项直接计算即可：

$$

\begin{align}
\frac{\partial{E}}{\partial{W_{ox}}}&=\frac{\partial{E}}{\partial{\mathbf{net}_{o,t}}}\frac{\partial{\mathbf{net}_{o,t}}}{\partial{W_{ox}}}\\
&=\delta_{o,t}\mathbf{x}_{t}^T\\\\
\frac{\partial{E}}{\partial{W_{fx}}}&=\frac{\partial{E}}{\partial{\mathbf{net}_{f,t}}}\frac{\partial{\mathbf{net}_{f,t}}}{\partial{W_{fx}}}\\
&=\delta_{f,t}\mathbf{x}_{t}^T\\\\
\frac{\partial{E}}{\partial{W_{ix}}}&=\frac{\partial{E}}{\partial{\mathbf{net}_{i,t}}}\frac{\partial{\mathbf{net}_{i,t}}}{\partial{W_{ix}}}\\
&=\delta_{i,t}\mathbf{x}_{t}^T\\\\
\frac{\partial{E}}{\partial{W_{cx}}}&=\frac{\partial{E}}{\partial{\mathbf{net}_{\tilde{c},t}}}\frac{\partial{\mathbf{net}_{\tilde{c},t}}}{\partial{W_{cx}}}\\
&=\delta_{\tilde{c},t}\mathbf{x}_{t}^T\\
\end{align}

$$

以上就是LSTM的训练算法的全部公式。因为这里面存在很多重复的模式，仔细看看，会发觉并不是太复杂。

当然，LSTM存在着相当多的变体，读者可以在互联网上找到很多资料。因为大家已经熟悉了基本LSTM的算法，因此理解这些变体比较容易，因此本文就不再赘述了。

转载自：[零基础入门深度学习(6) - 长短时记忆网络(LSTM)](https://zybuluo.com/hanbingtao/note/581764)