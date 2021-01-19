---
layout: post
title: 循环神经网络
date: 2021-01-12
author: Zhiliang 
tags: [Machine Learning]
toc: true
mathjax: true
---

循环神经网络（Recurrent Neural Network, RNN）是一类以序列（sequence）数据为输入，在序列的演进方向进行递归（recursion）且所有节点（循环单元）按链式连接的递归神经网络（recursive neural network）。

<!-- more -->

# 语言模型

RNN是在**自然语言处理**领域中最先被用起来的，比如，RNN可以为**语言模型**来建模。那么，什么是语言模型呢？

我们可以和电脑玩一个游戏，我们写出一个句子前面的一些词，然后，让电脑帮我们写下接下来的一个词。比如下面这句：

> 我昨天上学迟到了，老师批评了____。

我们给电脑展示了这句话前面这些词，然后，让电脑写下接下来的一个词。在这个例子中，接下来的这个词最有可能是『我』，而不太可能是『小明』，甚至是『吃饭』。

**语言模型**就是这样的东西：给定一个一句话前面的部分，预测接下来最有可能的一个词是什么。

**语言模型**是对一种语言的特征进行建模，它有很多很多用处。比如在语音转文本(STT)的应用中，声学模型输出的结果，往往是若干个可能的候选词，这时候就需要**语言模型**来从这些候选词中选择一个最可能的。当然，它同样也可以用在图像到文本的识别中(OCR)。

使用RNN之前，语言模型主要是采用N-Gram。N可以是一个自然数，比如2或者3。它的含义是，假设一个词出现的概率只与前面N个词相关。我们以2-Gram为例。首先，对前面的一句话进行切词：

> 我 昨天 上学 迟到 了 ，老师 批评 了 ____。

如果用2-Gram进行建模，那么电脑在预测的时候，只会看到前面的『了』，然后，电脑会在语料库中，搜索『了』后面最可能的一个词。不管最后电脑选的是不是『我』，我们都知道这个模型是不靠谱的，因为『了』前面说了那么一大堆实际上是没有用到的。如果是3-Gram模型呢，会搜索『批评了』后面最可能的词，感觉上比2-Gram靠谱了不少，但还是远远不够的。因为这句话最关键的信息『我』，远在9个词之前！

现在读者可能会想，可以提升继续提升N的值呀，比如4-Gram、5-Gram.......。实际上，这个想法是没有实用性的。因为我们想处理任意长度的句子，N设为多少都不合适；另外，模型的大小和N的关系是指数级的，4-Gram模型就会占用海量的存储空间。

所以，该轮到RNN出场了，RNN理论上可以往前看(往后看)任意多个词。

# 循环神经网络是啥

循环神经网络种类繁多，我们先从最简单的基本循环神经网络开始吧。

## 基本循环神经网络

下图是一个简单的循环神经网络如，它由输入层、一个隐藏层和一个输出层组成：

![](https://gitee.com/zhiliangj/Typora_Img/raw/master/20210108172833.png)

纳尼？！相信第一次看到这个玩意的读者内心和我一样是崩溃的。因为**循环神经网络**实在是太难画出来了，网上所有大神们都不得不用了这种抽象艺术手法。不过，静下心来仔细看看的话，其实也是很好理解的。如果把上面有W的那个带箭头的圈去掉，它就变成了最普通的**全连接神经网络**。

$x$是一个向量，它表示**输入层**的值（这里面没有画出来表示神经元节点的圆圈）；$s$是一个向量，它表示**隐藏层**的值（这里隐藏层面画了一个节点，你也可以想象这一层其实是多个节点，节点数与向量$s$的维度相同）；$U$是输入层到隐藏层的**权重矩阵**（读者可以回到第三篇文章[神经网络和反向传播算法](https://www.zybuluo.com/hanbingtao/note/476663)，看看我们是怎样用矩阵来表示**全连接神经网络**的计算的）；$o$也是一个向量，它表示**输出层**的值；$V$是隐藏层到输出层的**权重矩阵**。那么，现在我们来看看$W$是什么。**循环神经网络**的**隐藏层**的值$s$不仅仅取决于当前这次的输入$x$，还取决于上一次**隐藏层**的值$s$。**权重矩阵**$W$就是**隐藏层**上一次的值作为这一次的输入的权重。

如果我们把上面的图展开，**循环神经网络**也可以画成下面这个样子：

![](https://gitee.com/zhiliangj/Typora_Img/raw/master/2256672-cf18bb1f06e750a4.jpg)

现在看上去就比较清楚了，这个网络在$t$时刻接收到输入$x_t$之后，隐藏层的值是$s_t$，输出值是$o_t$。关键一点是，$s_t$的值不仅仅取决于$x_{t-1}$，还取决于$s_{t-1}$。我们可以用下面的公式来表示**循环神经网络**的计算方法：

$$

\begin{align}
\mathrm{o}_t&=g(V\mathrm{s}_t)\tag{2.1}\\
\mathrm{s}_t&=f(U\mathrm{x}_t+W\mathrm{s}_{t-1})\tag{2.2}\\
\end{align}

$$

**式2.1**是**输出层**的计算公式，输出层是一个**全连接层**，也就是它的每个节点都和隐藏层的每个节点相连。$V$是输出层的**权重矩阵**，$g$是**激活函数**。**式2.2**是隐藏层的计算公式，它是**循环层**。$U$是输入$x$的权重矩阵，$W$是上一次的值$s_{t-1}$作为这一次的输入的**权重矩阵**，$f$是**激活函数**。

从上面的公式我们可以看出，**循环层**和**全连接层**的区别就是**循环层**多了一个**权重矩阵**$W$。

如果反复把**式2.2**带入到**式2.1**，我们将得到：

$$

\begin{align}
\mathrm{o}_t&=g(V\mathrm{s}_t)\\
&=Vf(U\mathrm{x}_t+W\mathrm{s}_{t-1})\\
&=Vf(U\mathrm{x}_t+Wf(U\mathrm{x}_{t-1}+W\mathrm{s}_{t-2}))\\
&=Vf(U\mathrm{x}_t+Wf(U\mathrm{x}_{t-1}+Wf(U\mathrm{x}_{t-2}+W\mathrm{s}_{t-3})))\\
&=Vf(U\mathrm{x}_t+Wf(U\mathrm{x}_{t-1}+Wf(U\mathrm{x}_{t-2}+Wf(U\mathrm{x}_{t-3}+...))))
\end{align}

$$

从上面可以看出，**循环神经网络**的输出值$o_t$，是受前面历次输入值$x_{t}$、$x_{t-1}$、$x_{t-2}$、$x_{t-3}$、...影响的，这就是为什么**循环神经网络**可以往前看任意多个**输入值**的原因。

## 双向循环神经网络

对于**语言模型**来说，很多时候光看前面的词是不够的，比如下面这句话：

> 我的手机坏了，我打算____一部新手机。

可以想象，如果我们只看横线前面的词，手机坏了，那么我是打算修一修？换一部新的？还是大哭一场？这些都是无法确定的。但如果我们也看到了横线后面的词是『一部新手机』，那么，横线上的词填『买』的概率就大得多了。

在上一小节中的**基本循环神经网络**是无法对此进行建模的，因此，我们需要**双向循环神经网络**，如下图所示：



![](https://gitee.com/zhiliangj/Typora_Img/raw/master/2256672-039a45251aa5d220.png)

当遇到这种从未来穿越回来的场景时，难免处于懵逼的状态。不过我们还是可以用屡试不爽的老办法：先分析一个特殊场景，然后再总结一般规律。我们先考虑上图中，$y_2$的计算。

从上图可以看出，**双向卷积神经网络**的隐藏层要保存两个值，一个A参与正向计算，另一个值A'参与反向计算。最终的输出值取决于$A_2$和$A_2^{'}$。其计算方法为：

$$

\mathrm{y}_2=g(VA_2+V'A_2')

$$

$A_2$和$A_2^{'}$则分别计算：

$$

\begin{align}
A_2&=f(WA_1+U\mathrm{x}_2)\\
A_2'&=f(W'A_3'+U'\mathrm{x}_2)\\
\end{align}

$$

现在，我们已经可以看出一般的规律：正向计算时，隐藏层的值$s_t$与$s_{t-1}$有关；<u>反向计算时，隐藏层的值$s'_{t}$与$s'_{t+1}$有关；（有待考虑，下面的公式也是）</u>最终的输出取决于正向和反向计算的**加和**。现在，我们仿照**式2.1**和**式2.2**，写出双向循环神经网络的计算方法：

$$

\begin{align}
\mathrm{o}_t&=g(V\mathrm{s}_t+V'\mathrm{s}_t')\\
\mathrm{s}_t&=f(U\mathrm{x}_t+W\mathrm{s}_{t-1})\\
\mathrm{s}_t'&=f(U'\mathrm{x}_t+W'\mathrm{s}_{t+1}')\\
\end{align}

$$

从上面三个公式我们可以看到，正向计算和反向计算**不共享权重**，也就是说U和U'、W和W'、V和V'都是不同的**权重矩阵**。

## 深度循环神经网络(<u>不太懂</u>)

前面我们介绍的**循环神经网络**只有一个隐藏层，我们当然也可以堆叠两个以上的隐藏层，这样就得到了**深度循环神经网络**。如下图所示：

![](https://gitee.com/zhiliangj/Typora_Img/raw/master/2256672-df137de8007c3d26.png)

我们把第$i$个隐藏层的值表示为$\mathrm{s}_t^{(i)}$、$\mathrm{s'}_t^{(i)}$，则**深度循环神经网络**的计算方式可以表示为：

$$

\begin{align}
\mathrm{o}_t&=g(V^{(i)}\mathrm{s}_t^{(i)}+V'^{(i)}\mathrm{s}_t'^{(i)})\\
\mathrm{s}_t^{(i)}&=f(U^{(i)}\mathrm{s}_t^{(i-1)}+W^{(i)}\mathrm{s}_{t-1})\\
\mathrm{s}_t'^{(i)}&=f(U'^{(i)}\mathrm{s}_t'^{(i-1)}+W'^{(i)}\mathrm{s}_{t+1}')\\
...\\
\mathrm{s}_t^{(1)}&=f(U^{(1)}\mathrm{x}_t+W^{(1)}\mathrm{s}_{t-1})\\
\mathrm{s}_t'^{(1)}&=f(U'^{(1)}\mathrm{x}_t+W'^{(1)}\mathrm{s}_{t+1}')\\
\end{align}

$$


# 循环神经网络的训练

## 循环神经网络的训练算法：BPTT

BPTT算法是针对**循环层**的训练算法，它的基本原理和BP算法是一样的，也包含同样的三个步骤：

1. 前向计算每个神经元的输出值；
2. 反向计算每个神经元的**误差项**$\delta_j$值，它是误差函数E对神经元j的**加权输入**$net_{j}$的偏导数；
3. 计算每个权重的梯度。

最后再用**随机梯度下降**算法更新权重。

循环层如下图所示：

![](https://gitee.com/zhiliangj/Typora_Img/raw/master/2256672-3b20294694c3904b.png)

### 前向计算

使用前面的**式2.2**对循环层进行前向计算：

$$

\mathrm{s}_t=f(U\mathrm{x}_t+W\mathrm{s}_{t-1})

$$

注意，上面的$s_t$、$x_t$、$s_{t-1}$都是向量，用**黑体字母**表示；而$U$、$V$是**矩阵**，用大写字母表示。**向量的下标**表示**时刻**，例如，$s_t$表示在t时刻向量s的值。

我们假设输入向量$x$的维度是$m$，输出向量$s$的维度是$n$，则矩阵U的维度是$n×m$，矩阵$W$的维度是$n×n$。下面是上式展开成矩阵的样子，看起来更直观一些：

$$

\begin{align}
\begin{bmatrix}
s_1^t\\
s_2^t\\
.\\.\\
s_n^t\\
\end{bmatrix}=f(
\begin{bmatrix}
u_{11} u_{12} ... u_{1m}\\
u_{21} u_{22} ... u_{2m}\\
.\\.\\
u_{n1} u_{n2} ... u_{nm}\\
\end{bmatrix}
\begin{bmatrix}
x_1\\
x_2\\
.\\.\\
x_m\\
\end{bmatrix}+
\begin{bmatrix}
w_{11} w_{12} ... w_{1n}\\
w_{21} w_{22} ... w_{2n}\\
.\\.\\
w_{n1} w_{n2} ... w_{nn}\\
\end{bmatrix}
\begin{bmatrix}
s_1^{t-1}\\
s_2^{t-1}\\
.\\.\\
s_n^{t-1}\\
\end{bmatrix})
\end{align}

$$

在这里我们用**手写体字母**表示向量的一个**元素**，它的下标表示它是这个向量的第几个元素，它的上标表示第几个**时刻**。例如，$s_j^{t}$表示向量s的第j个元素在t时刻的值。$u_{ji}$表示**输入层**第i个神经元到**循环层**第j个神经元的权重。$w_{ji}$表示**循环层**第t-1时刻的第i个神经元到**循环层**第t个时刻的第j个神经元的权重。

### 误差项的计算

BTPP算法将第l层t时刻的**误差项**值沿两个方向传播，一个方向是其传递到上一层网络，得到$\delta_{t}^{l-1}$，这部分只和权重矩阵U有关；另一个是方向是将其沿时间线传递到初始$t_1$时刻，得到$\delta_1^l$，这部分只和权重矩阵W有关。

我们用$net_t$向量表示神经元在t时刻的**加权输入**，因为：

$$

\begin{align}
\mathrm{net}_t&=U\mathrm{x}_t+W\mathrm{s}_{t-1}\\
\mathrm{s}_{t-1}&=f(\mathrm{net}_{t-1})\\
\end{align}

$$

因此：

$$

\begin{align}
\frac{\partial{\mathrm{net}_t}}{\partial{\mathrm{net}_{t-1}}}&=\frac{\partial{\mathrm{net}_t}}{\partial{\mathrm{s}_{t-1}}}\frac{\partial{\mathrm{s}_{t-1}}}{\partial{\mathrm{net}_{t-1}}}\\
\end{align}

$$

我们用a表示列向量，$a^T$用表示行向量。上式的第一项是向量函数对向量求导，其结果为Jacobian矩阵：

$$

\begin{align}
\frac{\partial{\mathrm{net}_t}}{\partial{\mathrm{s}_{t-1}}}&=
\begin{bmatrix}
\frac{\partial{net_1^t}}{\partial{s_1^{t-1}}}& \frac{\partial{net_1^t}}{\partial{s_2^{t-1}}}& ...&  \frac{\partial{net_1^t}}{\partial{s_n^{t-1}}}\\
\frac{\partial{net_2^t}}{\partial{s_1^{t-1}}}& \frac{\partial{net_2^t}}{\partial{s_2^{t-1}}}& ...&  \frac{\partial{net_2^t}}{\partial{s_n^{t-1}}}\\
&.\\&.\\
\frac{\partial{net_n^t}}{\partial{s_1^{t-1}}}& \frac{\partial{net_n^t}}{\partial{s_2^{t-1}}}& ...&  \frac{\partial{net_n^t}}{\partial{s_n^{t-1}}}\\
\end{bmatrix}\\
&=\begin{bmatrix}
w_{11} & w_{12} & ... & w_{1n}\\
w_{21} & w_{22} & ... & w_{2n}\\
&.\\&.\\
w_{n1} & w_{n2} & ... & w_{nn}\\
\end{bmatrix}\\
&=W
\end{align}

$$

同理，上式第二项也是一个Jacobian矩阵：

$$

\begin{align}
\frac{\partial{\mathrm{s}_{t-1}}}{\partial{\mathrm{net}_{t-1}}}&=
\begin{bmatrix}
\frac{\partial{s_1^{t-1}}}{\partial{net_1^{t-1}}}& \frac{\partial{s_1^{t-1}}}{\partial{net_2^{t-1}}}& ...&  \frac{\partial{s_1^{t-1}}}{\partial{net_n^{t-1}}}\\
\frac{\partial{s_2^{t-1}}}{\partial{net_1^{t-1}}}& \frac{\partial{s_2^{t-1}}}{\partial{net_2^{t-1}}}& ...&  \frac{\partial{s_2^{t-1}}}{\partial{net_n^{t-1}}}\\
&.\\&.\\
\frac{\partial{s_n^{t-1}}}{\partial{net_1^{t-1}}}& \frac{\partial{s_n^{t-1}}}{\partial{net_2^{t-1}}}& ...&  \frac{\partial{s_n^{t-1}}}{\partial{net_n^{t-1}}}\\
\end{bmatrix}\\
&=\begin{bmatrix}
f'(net_1^{t-1}) & 0 & ... & 0\\
0 & f'(net_2^{t-1}) & ... & 0\\
&.\\&.\\
0 & 0 & ... & f'(net_n^{t-1})\\
\end{bmatrix}\\
&=diag[f'(\mathrm{net}_{t-1})]
\end{align}

$$

其中，$diag[a]$表示根据向量a创建一个对角矩阵，即

$$

diag(\mathrm{a})=\begin{bmatrix}
a_1 & 0 & ... & 0\\
0 & a_2 & ... & 0\\
&.\\&.\\
0 & 0 & ... & a_n\\
\end{bmatrix}\\

$$

最后，将两项合在一起，可得：

$$

\begin{align}
\frac{\partial{\mathrm{net}_t}}{\partial{\mathrm{net}_{t-1}}}&=\frac{\partial{\mathrm{net}_t}}{\partial{\mathrm{s}_{t-1}}}\frac{\partial{\mathrm{s}_{t-1}}}{\partial{\mathrm{net}_{t-1}}}\\
&=Wdiag[f'(\mathrm{net}_{t-1})]\\
&=\begin{bmatrix}
w_{11}f'(net_1^{t-1}) & w_{12}f'(net_2^{t-1}) & ... & w_{1n}f(net_n^{t-1})\\
w_{21}f'(net_1^{t-1}) & w_{22} f'(net_2^{t-1}) & ... & w_{2n}f(net_n^{t-1})\\
&.\\&.\\
w_{n1}f'(net_1^{t-1}) & w_{n2} f'(net_2^{t-1}) & ... & w_{nn} f'(net_n^{t-1})\\
\end{bmatrix}\\
\end{align}

$$

上式描述了将$\delta$沿时间往前传递一个时刻的规律，有了这个规律，我们就可以求得任意时刻k的**误差项**$\delta_k$：

$$

\begin{align}
\delta_k^T=&\frac{\partial{E}}{\partial{\mathrm{net}_k}}\\
=&\frac{\partial{E}}{\partial{\mathrm{net}_t}}\frac{\partial{\mathrm{net}_t}}{\partial{\mathrm{net}_k}}\\
=&\frac{\partial{E}}{\partial{\mathrm{net}_t}}\frac{\partial{\mathrm{net}_t}}{\partial{\mathrm{net}_{t-1}}}\frac{\partial{\mathrm{net}_{t-1}}}{\partial{\mathrm{net}_{t-2}}}...\frac{\partial{\mathrm{net}_{k+1}}}{\partial{\mathrm{net}_{k}}}\\
=&Wdiag[f'(\mathrm{net}_{t-1})]
Wdiag[f'(\mathrm{net}_{t-2})]
...
Wdiag[f'(\mathrm{net}_{k})]
\delta_t^l\\
=&\delta_t^T\prod_{i=k}^{t-1}Wdiag[f'(\mathrm{net}_{i})]
\end{align}\tag{3.1}

$$

**式3.1**就是将误差项沿时间反向传播的算法。

**循环层**将**误差项**反向传递到上一层网络，与普通的**全连接层**是完全一样的，这在前面的文章[神经网络和反向传播算法](https://www.zybuluo.com/hanbingtao/note/476663)中已经详细讲过了，在此仅简要描述一下。

**循环层**的**加权输入**$net^{l}$与上一层的**加权输入**$net^{l-1}$关系如下：

$$

\begin{align}
\mathrm{net}_t^l=&U\mathrm{a}_t^{l-1}+W\mathrm{s}_{t-1}\\
\mathrm{a}_t^{l-1}=&f^{l-1}(\mathrm{net}_t^{l-1})
\end{align}

$$

上式中$net_t^{l}$是第$l$层神经元的**加权输入**(假设第$l$层是**循环层**)；$net_t^{l-1}$是第$l-1$层神经元的**加权输入**；$a_t^{l-1}$是第$l-1$层神经元的输出；$f^{l-1}$是第$l-1$层的**激活函数**。

$$

\begin{align}
\frac{\partial{\mathrm{net}_t^l}}{\partial{\mathrm{net}_t^{l-1}}}=&\frac{\partial{\mathrm{net}^l}}{\partial{\mathrm{a}_t^{l-1}}}\frac{\partial{\mathrm{a}_t^{l-1}}}{\partial{\mathrm{net}_t^{l-1}}}\\
=&Udiag[f'^{l-1}(\mathrm{net}_t^{l-1})]
\end{align}

$$

所以，

$$

\begin{align}
(\delta_t^{l-1})^T=&\frac{\partial{E}}{\partial{\mathrm{net}_t^{l-1}}}\\
=&\frac{\partial{E}}{\partial{\mathrm{net}_t^l}}\frac{\partial{\mathrm{net}_t^l}}{\partial{\mathrm{net}_t^{l-1}}}\\
=&(\delta_t^l)^TUdiag[f'^{l-1}(\mathrm{net}_t^{l-1})]
\end{align}\tag{3.1}

$$

**式3.2**就是将误差项传递到上一层算法。

### 权重梯度的计算

现在，我们终于来到了BPTT算法的最后一步：计算每个权重的梯度。

首先，我们计算误差函数E对权重矩阵W的梯度$\frac{\partial{E}}{\partial{W}}$。

![](https://gitee.com/zhiliangj/Typora_Img/raw/master/2256672-f7d034c8f05812f7.png)

上图展示了我们到目前为止，在前两步中已经计算得到的量，包括每个时刻t**循环层**的输出值$s_t$，以及误差项$\delta_t$。

回忆一下我们在文章[神经网络和反向传播算法](https://www.zybuluo.com/hanbingtao/note/476663)介绍的全连接网络的权重梯度计算算法：只要知道了任意一个时刻的**误差项**$\delta_t$，以及上一个时刻循环层的输出值$s_{t-1}$，就可以按照下面的公式求出权重矩阵在t时刻的梯度$\nabla_{Wt}E$：

$$

\nabla_{W_t}E=\begin{bmatrix}
\delta_1^ts_1^{t-1} & \delta_1^ts_2^{t-1} & ... &  \delta_1^ts_n^{t-1}\\
\delta_2^ts_1^{t-1} & \delta_2^ts_2^{t-1} & ... &  \delta_2^ts_n^{t-1}\\
.\\.\\
\delta_n^ts_1^{t-1} & \delta_n^ts_2^{t-1} & ... &  \delta_n^ts_n^{t-1}\\
\end{bmatrix}\tag{3.3}

$$

在**式3.3**中，$\delta^{t}_{i}$表示t时刻**误差项**向量的第$i$个分量；$s_i^{t-1}$表示$t-1$时刻**循环层**第$i$个神经元的输出值。

我们下面可以简单推导一下**式3.3**。

我们知道：

$$

\begin{align}
\mathrm{net}_t=&U\mathrm{x}_t+W\mathrm{s}_{t-1}\\
\begin{bmatrix}
net_1^t\\
net_2^t\\
.\\.\\
net_n^t\\
\end{bmatrix}=&U\mathrm{x}_t+
\begin{bmatrix}
w_{11} & w_{12} & ... & w_{1n}\\
w_{21} & w_{22} & ... & w_{2n}\\
.\\.\\
w_{n1} & w_{n2} & ... & w_{nn}\\
\end{bmatrix}
\begin{bmatrix}
s_1^{t-1}\\
s_2^{t-1}\\
.\\.\\
s_n^{t-1}\\
\end{bmatrix}\\
=&U\mathrm{x}_t+
\begin{bmatrix}
w_{11}s_1^{t-1}+w_{12}s_2^{t-1}...w_{1n}s_n^{t-1}\\
w_{21}s_1^{t-1}+w_{22}s_2^{t-1}...w_{2n}s_n^{t-1}\\
.\\.\\
w_{n1}s_1^{t-1}+w_{n2}s_2^{t-1}...w_{nn}s_n^{t-1}\\
\end{bmatrix}\\
\end{align}

$$

因为对W求导与$Ux_t$无关，我们不再考虑。现在，我们考虑对权重项$w_{ji}$求导。通过观察上式我们可以看到$w_{ji}$只与$net_j^t$有关，所以：

$$

\begin{align}
\frac{\partial{E}}{\partial{w_{ji}}}=&\frac{\partial{E}}{\partial{net_j^t}}\frac{\partial{net_j^t}}{\partial{w_{ji}}}\\
=&\delta_j^ts_i^{t-1}
\end{align}

$$

按照上面的规律就可以生成**式3.3**里面的矩阵。

我们已经求得了权重矩阵W在t时刻的梯度$\nabla_{Wt}E$，最终的梯度$\nabla_{W}E$是各个时刻的梯度**之和**：

$$

\begin{align}
\nabla_WE=&\sum_{i=1}^t\nabla_{W_i}E\\
=&\begin{bmatrix}
\delta_1^ts_1^{t-1} & \delta_1^ts_2^{t-1} & ... &  \delta_1^ts_n^{t-1}\\
\delta_2^ts_1^{t-1} & \delta_2^ts_2^{t-1} & ... &  \delta_2^ts_n^{t-1}\\
.\\.\\
\delta_n^ts_1^{t-1} & \delta_n^ts_2^{t-1} & ... &  \delta_n^ts_n^{t-1}\\
\end{bmatrix}
+...+
\begin{bmatrix}
\delta_1^1s_1^0 & \delta_1^1s_2^0 & ... &  \delta_1^1s_n^0\\
\delta_2^1s_1^0 & \delta_2^1s_2^0 & ... &  \delta_2^1s_n^0\\
.\\.\\
\delta_n^1s_1^0 & \delta_n^1s_2^0 & ... &  \delta_n^1s_n^0\\
\end{bmatrix}
\end{align}\tag{3.4}

$$

**式3.4**就是计算**循环层**权重矩阵W的梯度的公式。

`----------数学公式超高能预警----------`

前面已经介绍了$\nabla_WE$的计算方法，看上去还是比较直观的。然而，读者也许会困惑，为什么最终的梯度是各个时刻的梯度**之和**呢？我们前面只是直接用了这个结论，实际上这里面是有道理的，只是这个数学推导比较绕脑子。感兴趣的同学可以仔细阅读接下来这一段，它用到了矩阵对矩阵求导、张量与向量相乘运算的一些法则。

我们还是从这个式子开始：<u>(懵)</u>

$$

\mathrm{net}_t=U\mathrm{x}_t+Wf(\mathrm{net}_{t-1})

$$

因为$U\mathrm{x}\_t$与W完全无关，我们把它看做常量。现在，考虑第一个式子加号右边的部分，因为W和$f(\mathrm{net}\_{t-1})$都是W的函数，因此我们要用到大学里面都学过的导数乘法运算：

$$

(uv)'=u'v+uv'

$$

因此，上面第一个式子写成：

$$

\frac{\partial{\mathrm{net}_t}}{\partial{W}}=\frac{\partial{W}}{\partial{W}}f(\mathrm{net}_{t-1})+W\frac{\partial{f(\mathrm{net}_{t-1})}}{\partial{W}}\\

$$

我们最终需要计算的是$\nabla_WE$：

$$

\begin{align}
\nabla_WE=&\frac{\partial{E}}{\partial{W}}\\
=&\frac{\partial{E}}{\partial{\mathrm{net}_t}}\frac{\partial{\mathrm{net}_t}}{\partial{W}}\\
=&\delta_t^T\frac{\partial{W}}{\partial{W}}f(\mathrm{net}_{t-1})+ \delta_t^TW\frac{\partial{f(\mathrm{net}_{t-1})}}{\partial{W}}\\
\end{align}\tag{3.5}

$$

我们先计算**式3.5**加号左边的部分。$\frac{\partial{W}}{\partial{W}}$是**矩阵对矩阵求导**，其结果是一个四维**张量(tensor)**，如下所示：

$$

\begin{align}
\frac{\partial{W}}{\partial{W}}=&
\begin{bmatrix}
\frac{\partial{w_{11}}}{\partial{W}} & \frac{\partial{w_{12}}}{\partial{W}} & ... & \frac{\partial{w_{1n}}}{\partial{W}}\\
\frac{\partial{w_{21}}}{\partial{W}} & \frac{\partial{w_{22}}}{\partial{W}} & ... & \frac{\partial{w_{2n}}}{\partial{W}}\\
.\\.\\
\frac{\partial{w_{n1}}}{\partial{W}} & \frac{\partial{w_{n2}}}{\partial{W}} & ... & \frac{\partial{w_{nn}}}{\partial{W}}\\
\end{bmatrix}\\
=&
\begin{bmatrix}
\begin{bmatrix}
\frac{\partial{w_{11}}}{\partial{w_{11}}} & \frac{\partial{w_{11}}}{\partial{w_{12}}} & ... & \frac{\partial{w_{11}}}{\partial{_{1n}}}\\
\frac{\partial{w_{11}}}{\partial{w_{21}}} & \frac{\partial{w_{11}}}{\partial{w_{22}}} & ... & \frac{\partial{w_{11}}}{\partial{_{2n}}}\\
.\\.\\
\frac{\partial{w_{11}}}{\partial{w_{n1}}} & \frac{\partial{w_{11}}}{\partial{w_{n2}}} & ... & \frac{\partial{w_{11}}}{\partial{_{nn}}}\\
\end{bmatrix} &
\begin{bmatrix}
\frac{\partial{w_{12}}}{\partial{w_{11}}} & \frac{\partial{w_{12}}}{\partial{w_{12}}} & ... & \frac{\partial{w_{12}}}{\partial{_{1n}}}\\
\frac{\partial{w_{12}}}{\partial{w_{21}}} & \frac{\partial{w_{12}}}{\partial{w_{22}}} & ... & \frac{\partial{w_{12}}}{\partial{_{2n}}}\\
.\\.\\
\frac{\partial{w_{12}}}{\partial{w_{n1}}} & \frac{\partial{w_{12}}}{\partial{w_{n2}}} & ... & \frac{\partial{w_{12}}}{\partial{_{nn}}}\\
\end{bmatrix}&...\\
.\\.\\
\end{bmatrix}\\
=&
\begin{bmatrix}
\begin{bmatrix}
1 & 0 & ... & 0\\
0 & 0 & ... & 0\\
.\\.\\
0 & 0 & ... & 0\\
\end{bmatrix} &
\begin{bmatrix}
0 & 1 & ... & 0\\
0 & 0 & ... & 0\\
.\\.\\
0 & 0 & ... & 0\\
\end{bmatrix}&...\\
.\\.\\
\end{bmatrix}\\
\end{align}

$$

接下来，我们知道$s_{t-1}=f({\mathrm{net}_{t-1}})$，它是一个**列向量**。我们让上面的四维张量与这个向量相乘，得到了一个三维张量，再左乘行向量$\delta_t^T$，最终得到一个矩阵：

$$

\begin{align}
\delta_t^T\frac{\partial{W}}{\partial{W}}f({\mathrm{net}_{t-1}})=&
\delta_t^T\frac{\partial{W}}{\partial{W}}{\mathrm{s}_{t-1}}\\
=&\delta_t^T
\begin{bmatrix}
\begin{bmatrix}
1 & 0 & ... & 0\\
0 & 0 & ... & 0\\
.\\.\\
0 & 0 & ... & 0\\
\end{bmatrix} &
\begin{bmatrix}
0 & 1 & ... & 0\\
0 & 0 & ... & 0\\
.\\.\\
0 & 0 & ... & 0\\
\end{bmatrix}&...\\
.\\.\\
\end{bmatrix}
\begin{bmatrix}
s_1^{t-1}\\
s_2^{t-1}\\
.\\.\\
s_n^{t-1}\\
\end{bmatrix}\\
=&\delta_t^T
\begin{bmatrix}
\begin{bmatrix}
s_1^{t-1}\\
0\\
.\\.\\
0\\
\end{bmatrix} &
\begin{bmatrix}
s_2^{t-1}\\
0\\
.\\.\\
0\\
\end{bmatrix}&...\\
.\\.\\
\end{bmatrix}\\
=&
\begin{bmatrix}
\delta_1^t & \delta_2^t & ... &\delta_n^t
\end{bmatrix}
\begin{bmatrix}
\begin{bmatrix}
s_1^{t-1}\\
0\\
.\\.\\
0\\
\end{bmatrix} &
\begin{bmatrix}
s_2^{t-1}\\
0\\
.\\.\\
0\\
\end{bmatrix}&...\\
.\\.\\
\end{bmatrix}\\
=&
\begin{bmatrix}
\delta_1^ts_1^{t-1} & \delta_1^ts_2^{t-1} & ... &  \delta_1^ts_n^{t-1}\\
\delta_2^ts_1^{t-1} & \delta_2^ts_2^{t-1} & ... &  \delta_2^ts_n^{t-1}\\
.\\.\\
\delta_n^ts_1^{t-1} & \delta_n^ts_2^{t-1} & ... &  \delta_n^ts_n^{t-1}\\
\end{bmatrix}\\
=&\nabla_{Wt}E
\end{align}

$$

接下来，我们计算**式3.5**加号右边的部分：

$$

\begin{align}
\delta_t^TW\frac{\partial{f(\mathrm{net}_{t-1})}}{\partial{W}}=&
\delta_t^TW\frac{\partial{f(\mathrm{net}_{t-1})}}{\partial{\mathrm{net}_{t-1}}}\frac{\partial{\mathrm{net}_{t-1}}}{\partial{W}}\\
=&\delta_t^TWf'(\mathrm{net}_{t-1})\frac{\partial{\mathrm{net}_{t-1}}}{\partial{W}}\\
=&\delta_t^T\frac{\partial{\mathrm{net}_t}}{\partial{\mathrm{net}_{t-1}}}\frac{\partial{\mathrm{net}_{t-1}}}{\partial{W}}\\
=&\delta_{t-1}^T\frac{\partial{\mathrm{net}_{t-1}}}{\partial{W}}\\
\end{align}

$$

于是，我们得到了如下递推公式：

$$

\begin{align}
\nabla_WE=&\frac{\partial{E}}{\partial{W}}\\
=&\frac{\partial{E}}{\partial{\mathrm{net}_t}}\frac{\partial{\mathrm{net}_t}}{\partial{W}}\\
=&\nabla_{Wt}E+\delta_{t-1}^T\frac{\partial{\mathrm{net}_{t-1}}}{\partial{W}}\\
=&\nabla_{Wt}E+\nabla_{Wt-1}E+\delta_{t-2}^T\frac{\partial{\mathrm{net}_{t-2}}}{\partial{W}}\\
=&\nabla_{Wt}E+\nabla_{Wt-1}E+...+\nabla_{W1}E\\
=&\sum_{k=1}^t\nabla_{Wk}E
\end{align}

$$

这样，我们就证明了：最终的梯度是各个时刻的梯度之和。

`----------数学公式超高能预警解除----------`

同权重矩阵W类似，我们可以得到权重矩阵U的计算方法。

$$

\nabla_{U_t}E=\begin{bmatrix}
\delta_1^tx_1^t & \delta_1^tx_2^t & ... &  \delta_1^tx_m^t\\
\delta_2^tx_1^t & \delta_2^tx_2^t & ... &  \delta_2^tx_m^t\\
.\\.\\
\delta_n^tx_1^t & \delta_n^tx_2^t & ... &  \delta_n^tx_m^t\\
\end{bmatrix}\tag{3.6}

$$

**式3.6**是误差函数在t时刻对权重矩阵U的梯度。和权重矩阵W一样，最终的梯度也是各个时刻的梯度之和：

$$

\nabla_UE=\sum_{i=1}^t\nabla_{U_i}E

$$

具体的证明这里就不再赘述了，之后自己证明。

### RNN的梯度爆炸和消失问题

不幸的是，实践中前面介绍的几种RNNs并不能很好的处理较长的序列。一个主要的原因是，RNN在训练中很容易发生**梯度爆炸**和**梯度消失**，这导致训练时梯度不能在较长序列中一直传递下去，从而使RNN无法捕捉到长距离的影响。

为什么RNN会产生梯度爆炸和消失问题呢？我们接下来将详细分析一下原因。我们根据**式3.1**可得：

$$

\begin{align}
\delta_k^T=&\delta_t^T\prod_{i=k}^{t-1}Wdiag[f'(\mathrm{net}_{i})]\\
\|\delta_k^T\|\leqslant&\|\delta_t^T\|\prod_{i=k}^{t-1}\|W\|\|diag[f'(\mathrm{net}_{i})]\|\\
\leqslant&\|\delta_t^T\|(\beta_W\beta_f)^{t-k}
\end{align}

$$

上式的$\beta$定义为矩阵的模的上界。因为上式是一个指数函数，如果t-k很大的话（也就是向前看很远的时候），会导致对应的**误差项**的值增长或缩小的非常快，这样就会导致相应的**梯度爆炸**和**梯度消失**问题（取决于$\beta$大于1还是小于1）。

通常来说，**梯度爆炸**更容易处理一些。因为梯度爆炸的时候，我们的程序会收到$NaN$错误。我们也可以设置一个梯度阈值，当梯度超过这个阈值的时候可以直接截取。

**梯度消失**更难检测，而且也更难处理一些。总的来说，我们有三种方法应对梯度消失问题：

1. 合理的初始化权重值。初始化权重，使每个神经元尽可能不要取极大或极小值，以躲开梯度消失的区域。
2. 使用relu代替sigmoid和tanh作为激活函数。原理请参考上一篇文章[卷积神经网络](https://www.zybuluo.com/hanbingtao/note/485480)的**激活函数**一节。
3. 使用其他结构的RNNs，比如长短时记忆网络（LTSM）和Gated Recurrent Unit（GRU），这是最流行的做法。我们将在以后的文章中介绍这两种网络。

转载自：[零基础入门深度学习(5) - 循环神经网络](https://zybuluo.com/hanbingtao/note/541458)