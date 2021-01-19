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

# 长短时记忆网络是啥

我们首先了解一下长短时记忆网络产生的背景。回顾一下[循环神经网络](https://zybuluo.com/hanbingtao/note/541458)中推导的，误差项沿时间反向传播的公式：

$$

\begin{align}
\delta_k^T=&\delta_t^T\prod_{i=k}^{t-1}diag[f'(\mathbf{net}_{i})]W\\
\end{align}

$$

我们可以根据下面的不等式，来获取$\delta_k^T$的模的上界（模可以看做对$\delta_k^T$中每一项值的大小的度量）：

$$

\begin{align}
\|\delta_k^T\|\leqslant&\|\delta_t^T\|\prod_{i=k}^{t-1}\|diag[f'(\mathbf{net}_{i})]\|\|W\|\\
\leqslant&\|\delta_t^T\|(\beta_f\beta_W)^{t-k}
\end{align}

$$

我们可以看到，误差项$\delta$从t时刻传递到k时刻，其值的上界是$\beta_f\beta_w$的指数函数。$\beta_f\beta_w$分别是对角矩阵$diag[f'(\mathbf{net}_{i})]$和矩阵W模的上界。显然，除非$\beta_f\beta_w$乘积的值位于1附近，否则，当t-k很大时（也就是误差传递很多个时刻时），整个式子的值就会变得极小（当$\beta_f\beta_w$乘积小于1）或者极大（当$\beta_f\beta_w$乘积大于1），前者就是**梯度消失**，后者就是**梯度爆炸**。虽然科学家们搞出了很多技巧（比如怎样初始化权重），让的值尽可能贴近于1，终究还是难以抵挡指数函数的威力。

**梯度消失**到底意味着什么？在[循环神经网络](https://zybuluo.com/hanbingtao/note/541458)中我们已证明，权重数组W最终的梯度是各个时刻的梯度之和，即：

$$

\begin{align}
\nabla_WE&=\sum_{k=1}^t\nabla_{Wk}E\\
&=\nabla_{Wt}E+\nabla_{Wt-1}E+\nabla_{Wt-2}E+...+\nabla_{W1}E
\end{align}

$$

假设某轮训练中，各时刻的梯度以及最终的梯度之和如下图：

![](https://gitee.com/zhiliangj/Typora_Img/raw/master/2256672-48784f6366412472.png)

我们就可以看到，从上图的t-3时刻开始，梯度已经几乎减少到0了。那么，从这个时刻开始再往之前走，得到的梯度（几乎为零）就不会对最终的梯度值有任何贡献，这就相当于无论t-3时刻之前的网络状态h是什么，在训练中都不会对权重数组W的更新产生影响，也就是网络事实上已经忽略了t-3时刻之前的状态。这就是原始RNN无法处理长距离依赖的原因。

既然找到了问题的原因，那么我们就能解决它。从问题的定位到解决，科学家们大概花了7、8年时间。终于有一天，Hochreiter和Schmidhuber两位科学家发明出**长短时记忆网络**，一举解决这个问题。

其实，**长短时记忆网络**的思路比较简单。原始RNN的隐藏层只有一个状态，即h，它对于短期的输入非常敏感。那么，假如我们再增加一个状态，即c，让它来保存长期的状态，那么问题不就解决了么？如下图所示：

![](https://gitee.com/zhiliangj/Typora_Img/raw/master/2256672-71de4194da5a5ec4.png)

新增加的状态c，称为**单元状态(cell state)**。我们把上图按照时间维度展开：

![](https://gitee.com/zhiliangj/Typora_Img/raw/master/2256672-715658c134b9d6f1.png)

上图仅仅是一个示意图，我们可以看出，在t时刻，LSTM的输入有三个：当前时刻网络的输入值$\mathbf{x}_t$、上一时刻LSTM的输出值$\mathbf{h}_{t-1}$、以及上一时刻的单元状态$\mathbf{c}_{t-1}$；LSTM的输出有两个：当前时刻LSTM输出值$\mathbf{h}_t$、和当前时刻的单元状态$\mathbf{c}_t$。注意$\mathbf{x}$、$\mathbf{h}$、$\mathbf{c}$都是**向量**。

LSTM的关键，就是怎样控制长期状态c。在这里，LSTM的思路是使用三个控制开关。第一个开关，负责控制继续保存长期状态c；第二个开关，负责控制把即时状态输入到长期状态c；第三个开关，负责控制是否把长期状态c作为当前的LSTM的输出。三个开关的作用如下图所示：

![](https://gitee.com/zhiliangj/Typora_Img/raw/master/2256672-bff9353b92b9c488.png)

接下来，我们要描述一下，输出h和单元状态c的具体计算方法。

# 长短时记忆网络的前向计算

前面描述的开关是怎样在算法中实现的呢？这就用到了**门（gate）**的概念。门实际上就是一层**全连接层**，它的输入是一个向量，输出是一个0到1之间的实数向量。假设W是门的权重向量，$b$是偏置项，那么门可以表示为：

$$

g(\mathbf{x})=\sigma(W\mathbf{x}+\mathbf{b})

$$

门的使用，就是用门的输出向量按元素乘以我们需要控制的那个向量。因为门的输出是0到1之间的实数向量，那么，当门输出为0时，任何向量与之相乘都会得到0向量，这就相当于啥都不能通过；输出为1时，任何向量与之相乘都不会有任何改变，这就相当于啥都可以通过。因为$\sigma$（也就是sigmoid函数）的值域是(0,1)，所以门的状态都是半开半闭的。

LSTM用两个门来控制单元状态c的内容，一个是**遗忘门（forget gate）**，它决定了上一时刻的单元状态$\mathbf{c}_{t-1}$有多少保留到当前时刻$\mathbf{c}_t$；另一个是**输入门（input gate）**，它决定了当前时刻网络的输入$\mathbf{x}_t$有多少保存到单元状态$\mathbf{c}_t$。LSTM用**输出门（output gate）**来控制单元状态$\mathbf{x}_t$有多少输出到LSTM的当前输出值$\mathbf{h}_t$。

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

LSTM需要学习的参数共有8组，分别是：遗忘门的权重矩阵$W_f$和偏置项$\mathbf{b}_f$、输入门的权重矩阵$W_i$和偏置项$\mathbf{b}_i$、输出门的权重矩阵$W_o$和偏置项$\mathbf{b}_o$，以及计算单元状态的权重矩阵$W_c$和偏置项$\mathbf{b}_c$。因为权重矩阵的两部分在反向传播中使用不同的公式，因此在后续的推导中，权重矩阵$W_{f}$、$W_{i}$、$W_{o}$、$W_{c}$都将被写为分开的两个矩阵：$W_{fh}$、$W_{fx}$、$W_{ih}$、$W_{ix}$、$W_{oh}$、$W_{ox}$、$W_{ch}$、$W_{cx}$。

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

我们知道，$\frac{\partial{\mathbf{h_t}}}{\partial{\mathbf{h_{t-1}}}}$是一个Jacobian矩阵。如果隐藏层h的维度是N的话，那么它就是一个矩阵。为了求出它，我们列出的计算公式，即前面的**式6**和**式4**：

转载自：[零基础入门深度学习(6) - 长短时记忆网络(LSTM)](https://zybuluo.com/hanbingtao/note/581764)