---
layout: post
title: 卷积神经网络
date: 2021-01-08
author: Zhiliang 
tags: [Machine Learning]
toc: true
mathjax: true
---

在[深度学习中](https://en.wikipedia.org/wiki/Deep_learning)，**卷积神经网络**（**CNN**或**ConvNet**）是一类[深度神经网络](https://en.wikipedia.org/wiki/Deep_neural_network)，最常用于分析视觉图像。基于它们的共享权重架构和[平移不变性](https://en.wikipedia.org/wiki/Translation_invariance)特征，它们也被称为**位移不变**或**空间不变的人工神经网络**（**SIANN**）。



<!-- more -->

# 一个新的激活函数——Relu

最近几年卷积神经网络中，激活函数往往不选择sigmoid或tanh函数，而是选择relu函数。Relu函数的定义是：

$$

f(x)= max(0,x)

$$

Relu函数图像如下图所示：

![](https://gitee.com/zhiliangj/Typora_Img/raw/master/2256672-0ac9923bebd3c9dd.png)

Relu函数作为激活函数，有下面几大优势：

- **速度快** 和sigmoid函数需要计算指数和倒数相比，relu函数其实就是一个max(0,x)，计算代价小很多。
- **减轻梯度消失问题** 回忆一下计算梯度的公式 $ \nabla=\sigma'\delta x $ 。其中，是sigmoid函数的导数。在使用反向传播算法进行梯度计算时，每经过一层sigmoid神经元，梯度就要乘上一个 $ \sigma' $ 。从下图可以看出， $ \sigma' $ 函数最大值是1/4。因此，乘一个 $ \sigma' $ 会导致梯度越来越小，这对于深层网络的训练是个很大的问题。而relu函数的导数是1，不会导致梯度变小。当然，激活函数仅仅是导致梯度减小的一个因素，但无论如何在这方面relu的表现强于sigmoid。使用relu激活函数可以让你训练更深的网络。

![](http://upload-images.jianshu.io/upload_images/2256672-ad98d6b22f1a66ab.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/360)

- **稀疏性** 通过对大脑的研究发现，大脑在工作的时候只有大约5%的神经元是激活的，而采用sigmoid激活函数的人工神经网络，其激活率大约是50%。有论文声称人工神经网络在15%-30%的激活率时是比较理想的。因为relu函数在输入小于0时是完全不激活的，因此可以获得一个更低的激活率。

# 全连接网络 VS 卷积网络

全连接神经网络之所以不太适合图像识别任务，主要有以下几个方面的问题：

- **参数数量太多** 考虑一个输入1000\*1000像素的图片(一百万像素，现在已经不能算大图了)，输入层有1000\*1000=100万节点。假设第一个隐藏层有100个节点(这个数量并不多)，那么仅这一层就有(1000\*1000+1)\*100=1亿参数，这实在是太多了！我们看到图像只扩大一点，参数数量就会多很多，因此它的扩展性很差。
- **没有利用像素之间的位置信息** 对于图像识别任务来说，每个像素和其周围像素的联系是比较紧密的，和离得很远的像素的联系可能就很小了。如果一个神经元和上一层所有神经元相连，那么就相当于对于一个像素来说，把图像的所有像素都等同看待，这不符合前面的假设。当我们完成每个连接权重的学习之后，最终可能会发现，有大量的权重，它们的值都是很小的(也就是这些连接其实无关紧要)。努力学习大量并不重要的权重，这样的学习必将是非常低效的。
- **网络层数限制** 我们知道网络层数越多其表达能力越强，但是通过梯度下降方法训练深度全连接神经网络很困难，因为全连接神经网络的梯度很难传递超过3层。因此，我们不可能得到一个很深的全连接神经网络，也就限制了它的能力。

那么，卷积神经网络又是怎样解决这个问题的呢？主要有三个思路：

- **局部连接** 这个是最容易想到的，每个神经元不再和上一层的所有神经元相连，而只和一小部分神经元相连。这样就减少了很多参数。
- **权值共享** 一组连接可以共享同一个权重，而不是每个连接有一个不同的权重，这样又减少了很多参数。
- **下采样** 可以使用Pooling来减少每层的样本数，进一步减少参数数量，同时还可以提升模型的鲁棒性。

对于图像识别任务来说，卷积神经网络通过尽可能保留重要的参数，去掉大量不重要的参数，来达到更好的学习效果。

接下来，我们将详述卷积神经网络到底是何方神圣。

# 卷积神经网络是啥

首先，我们先获取一个感性认识，下图是一个卷积神经网络的示意图：

![图1](https://gitee.com/zhiliangj/Typora_Img/raw/master/2256672-a36210f89c7164a7.png)

## 网络架构

如**图1**所示，一个卷积神经网络由若干**卷积层**、**Pooling层**、**全连接层**组成。你可以构建各种不同的卷积神经网络，它的常用架构模式为：

`INPUT -> [[CONV]*N -> POOL?]*M -> [FC]*K`

也就是N个卷积层叠加，然后(可选)叠加一个Pooling层，重复这个结构M次，最后叠加K个全连接层。

对于**图1**展示的卷积神经网络：

`INPUT -> CONV -> POOL -> CONV -> POOL -> FC -> FC`

按照上述模式可以表示为：

`INPUT -> [[CONV]*1 -> POOL]*2 -> [FC]*2`

也就是`N=1, M=2, K=2`。

## 三维的层结构

从**图1**我们可以发现**卷积神经网络**的层结构和**全连接神经网络**的层结构有很大不同。**全连接神经网络**每层的神经元是按照**一维**排列的，也就是排成一条线的样子；而**卷积神经网络**每层的神经元是按照**三维**排列的，也就是排成一个长方体的样子，有**宽度**、**高度**和**深度**。

对于**图1**展示的神经网络，我们看到输入层的宽度和高度对应于输入图像的宽度和高度，而它的深度为1。接着，第一个卷积层对这幅图像进行了卷积操作(后面我们会讲如何计算卷积)，得到了三个Feature Map。这里的"3"可能是让很多初学者迷惑的地方，实际上，就是这个卷积层包含三个Filter，也就是三套参数，每个Filter都可以把原始输入图像卷积得到一个Feature Map，三个Filter就可以得到三个Feature Map。至于一个卷积层可以有多少个Filter，那是可以自由设定的。也就是说，卷积层的Filter个数也是一个**超参数**。我们可以把Feature Map可以看做是通过卷积变换提取到的图像特征，三个Filter就对原始图像提取出三组不同的特征，也就是得到了三个Feature Map，也称做三个**通道(channel)**。

继续观察**图1**，在第一个卷积层之后，Pooling层对三个Feature Map做了**下采样**(后面我们会讲如何计算下采样)，得到了三个更小的Feature Map。接着，是第二个**卷积层**，它有5个Filter。每个Fitler都把前面**下采样**之后的**3个Feature Map卷积**在一起，得到一个新的Feature Map。这样，5个Filter就得到了5个Feature Map。接着，是第二个Pooling，继续对5个Feature Map进行**下采样**，得到了5个更小的Feature Map。

**图1**所示网络的最后两层是全连接层。第一个全连接层的每个神经元，和上一层5个Feature Map中的每个神经元相连，第二个全连接层(也就是输出层)的每个神经元，则和第一个全连接层的每个神经元相连，这样得到了整个网络的输出。

至此，我们对**卷积神经网络**有了最基本的感性认识。接下来，我们将介绍**卷积神经网络**中各种层的计算和训练。

# 卷积神经网络输出值的计算

## 卷积层输出值的计算

我们用一个简单的例子来讲述如何计算**卷积**，然后，我们抽象出**卷积层**的一些重要概念和计算方法。

假设有一个5\*5的图像，使用一个3\*3的filter进行卷积，想得到一个3\*3的Feature Map，如下所示：

![](https://gitee.com/zhiliangj/Typora_Img/raw/master/2256672-548b82ccd7977294.png)

为了清楚的描述**卷积**计算过程，我们首先对图像的每个像素进行编号，用 $ x_{i,j} $ 表示图像的第 $ i $ 行第 $ j $ 列元素；对filter的每个权重进行编号，用 $ w_{m,n} $ 表示第 $ m $ 行第 $ n $ 列权重，用 $ w_b $ 表示filter的**偏置项**；对Feature Map的每个元素进行编号，用 $ a_{i,j} $ 表示Feature Map的第 $ i $ 行第 $ j $ 列元素；用 $ f $ 表示**激活函数**(这个例子选择**relu函数**作为激活函数)。然后，使用下列公式计算卷积：

$$

a_{i,j}=f(\sum_{m=0}^{2}\sum_{n=0}^{2}w_{m,n}x_{i+m,j+n}+w_b) \tag{4.1}

$$

例如，对于Feature Map左上角元素 $ a_{0,0} $ 来说，其卷积计算方法为：

$$

\begin{align}
a_{0,0}&=f(\sum_{m=0}^{2}\sum_{n=0}^{2}w_{m,n}x_{m+0,n+0}+w_b)\\
&=relu(w_{0,0}x_{0,0}+w_{0,1}x_{0,1}+w_{0,2}x_{0,2}+w_{1,0}x_{1,0}+w_{1,1}x_{1,1}+w_{1,2}x_{1,2}+w_{2,0}x_{2,0}+w_{2,1}x_{2,1}+w_{2,2}x_{2,2}+w_b)\\
&=relu(1+0+1+0+1+0+0+0+1+0)\\
&=relu(4)\\
&=4
\end{align}

$$

计算结果如下图所示：

![](https://gitee.com/zhiliangj/Typora_Img/raw/master/2256672-318017ad134effc5.png)

接下来，Feature Map的元素 $ a_{0,1} $ 的卷积计算方法为：

$$

\begin{align}
a_{0,1}&=f(\sum_{m=0}^{2}\sum_{n=0}^{2}w_{m,n}x_{m+0,n+1}+w_b)\\
&=relu(w_{0,0}x_{0,1}+w_{0,1}x_{0,2}+w_{0,2}x_{0,3}+w_{1,0}x_{1,1}+w_{1,1}x_{1,2}+w_{1,2}x_{1,3}+w_{2,0}x_{2,1}+w_{2,1}x_{2,3}+w_{2,2}x_{2,3}+w_b)\\
&=relu(1+0+0+0+1+0+0+0+1+0)\\
&=relu(3)\\
&=3
\end{align}

$$

计算结果如下图所示：

![](https://gitee.com/zhiliangj/Typora_Img/raw/master/2256672-b05427072f4c548d.png)

可以依次计算出Feature Map中所有元素的值。下面的动画显示了整个Feature Map的计算过程：

![图2](https://gitee.com/zhiliangj/Typora_Img/raw/master/2256672-19110dee0c54c0b2.gif)

上面的计算过程中，步幅(stride)为1。步幅可以设为大于1的数。例如，当步幅为2时，Feature Map计算如下：

![](https://gitee.com/zhiliangj/Typora_Img/raw/master/2256672-273e3d9cf9dececb.png)

![](https://gitee.com/zhiliangj/Typora_Img/raw/master/2256672-7f362ea9350761d9.png)

![](https://gitee.com/zhiliangj/Typora_Img/raw/master/2256672-f5fa1e904cb0287e.png)

![](https://gitee.com/zhiliangj/Typora_Img/raw/master/2256672-7919cabd375b4cfd.png)

我们注意到，当**步幅**设置为2的时候，Feature Map就变成2\*2了。这说明图像大小、步幅和卷积后的Feature Map大小是有关系的。事实上，它们满足下面的关系：

$$

\begin{align}
W_2 &= (W_1 - F + 2P)/S + 1\qquad(4.2)\\
H_2 &= (H_1 - F + 2P)/S + 1\qquad(4.3)
\end{align}

$$

在上面两个公式中， $ W_2 $ 是卷积后Feature Map的宽度； $ W_1 $ 是卷积前图像的宽度； $ F $ 是filter的宽度； $ P $ 是**Zero Padding**数量，**Zero Padding**是指在原始图像周围补几圈0，如果的值是1，那么就补1圈0； $ S $ 是**步幅**； $ H_2 $ 是卷积后Feature Map的高度； $ H_1 $ 是卷积前图像的高度。**式2**和**式3**本质上是一样的。

以前面的例子来说，图像宽度 $ W_1=5 $ ，filter宽度 $ F=3 $ ，**Zero Padding** $ P=0 $ ，**步幅** $ S=2 $ ，则

$$

\begin{align}
W_2 &= (W_1 - F + 2P)/S + 1\\
&= (5 - 3 + 0)/2 + 1\\
&=2
\end{align}

$$

说明Feature Map宽度是2。同样，我们也可以计算出Feature Map高度也是2。

前面我们已经讲了深度为1的卷积层的计算方法，如果深度大于1怎么计算呢？其实也是类似的。<u>如果卷积前的图像深度为D，那么相应的filter的深度也必须为D。</u>我们扩展一下**式1**，得到了深度大于1的卷积计算公式：

$$

a_{i,j}=f(\sum_{d=0}^{D-1}\sum_{m=0}^{F-1}\sum_{n=0}^{F-1}w_{d,m,n}x_{d,i+m,j+n}+w_b) \tag{4.4}

$$

在**式4.4**中，D是深度；F是filter的大小(宽度或高度，两者相同)； $ w_{d,m,n} $ 表示filter的第 $ d $ 层第 $ m $ 行第 $ n $ 列权重； $ a_{d,i,j} $ 表示图像的第 $ d $ 层第 $ i $ 行第 $ j $ 列像素；其它的符号含义和**式4.1**是相同的，不再赘述。

我们前面还曾提到，每个卷积层可以有多个filter。每个filter和原始图像进行卷积后，都可以得到一个Feature Map。因此，卷积后Feature Map的深度(个数)和卷积层的filter个数是相同的。

下面的动画显示了包含两个filter的卷积层的计算。我们可以看到7\*7\*3输入，经过两个3\*3\*3filter的卷积(步幅为2)，得到了3\*3\*2的输出。另外我们也会看到下图的**Zero padding**是1，也就是在输入元素的周围补了一圈0。**Zero padding**对于图像边缘部分的特征提取是很有帮助的。

![](http://upload-images.jianshu.io/upload_images/2256672-958f31b01695b085.gif)

以上就是卷积层的计算方法。这里面体现了**局部连接**和**权值共享**：每层神经元只和上一层部分神经元相连(卷积计算规则)，且filter的权值对于上一层所有神经元都是一样的。对于包含两个3\*3\*3的fitler的卷积层来说，其参数数量仅有(3\*3\*3+1)*2=56个，且参数数量与上一层神经元个数无关。与**全连接神经网络**相比，其参数数量大大减少了。

### 用卷积公式来表达卷积层计算

不想了解太多数学细节的读者可以跳过这一节，不影响对全文的理解。

**式4.4**的表达很是繁冗，最好能简化一下。就像利用矩阵可以简化表达**全连接神经网络**的计算一样，我们利用**卷积公式**可以简化**卷积神经网络**的表达。

下面我们介绍**二维卷积公式**。

设矩阵 $ A $ ， $ B $ ，其行、列数分别为 $ m_a $ 、 $ n_a $ 、 $ m_b $ 、 $ n_b $ ，则**二维卷积公式**如下：

$$

\begin{align}
C_{s,t}&=\sum_0^{m_a-1}\sum_0^{n_a-1} A_{m,n}B_{s-m,t-n}
\end{align} \tag{4.5}

$$

且 $ s $ , $ t $ 满足条件 $ 0\le{s}\lt{m_a+m_b-1}, 0\le{t}\lt{n_a+n_b-1} $ 。

我们可以把上式写成

$$

C = A * B \tag{4.6}

$$

如果我们按照**式4.6**来计算卷积，我们可以发现矩阵A实际上是filter，而矩阵B是待卷积的输入，位置关系也有所不同：

![](https://gitee.com/zhiliangj/Typora_Img/raw/master/2256672-8d30f15073885d7b.png)

从上图可以看到，A左上角的值 $ a_{0,0} $ 与B对应区块中右下角的值 $ b_{1,1} $ 相乘，而不是与左上角 $ b_{0,0} $ 的相乘。因此，**数学**中的卷积和**卷积神经网络**中的『卷积』还是有区别的，为了避免混淆，我们把**卷积神经网络**中的『卷积』操作叫做**互相关(cross-correlation)**操作。

**卷积**和**互相关**操作是可以转化的。首先，我们把矩阵A翻转180度，然后再交换A和B的位置（即把B放在左边而把A放在右边。卷积满足交换率，这个操作不会导致结果变化），那么**卷积**就变成了**互相关**。

如果我们不去考虑两者这么一点点的区别，我们可以把**式4.6**代入到**式4.4**：

$$

A=f(\sum_{d=0}^{D-1}X_d*W_d+w_b) \tag{4.7}

$$

其中， $ A $ 是卷积层输出的feature map。同**式4.4**相比，**式4.7**就简单多了。然而，这种简洁写法只适合步长为1的情况。

## Pooling层输出值的计算

Pooling层主要的作用是**下采样**，通过去掉Feature Map中不重要的样本，进一步减少参数数量。Pooling的方法很多，最常用的是**Max Pooling**。**Max Pooling**实际上就是在n\*n的样本中取最大值，作为采样后的样本值。下图是2\*2 max pooling：

![](https://gitee.com/zhiliangj/Typora_Img/raw/master/2256672-03bfc7683ad2e3ad.png)

除了**Max Pooing**之外，常用的还有**Mean Pooling**——取各样本的平均值。

对于深度为D的Feature Map，各层独立做Pooling，因此Pooling后的深度仍然为D。

## 全连接层

全连接层输出值的计算和上一篇文章[神经网络和反向传播算法](https://www.zybuluo.com/hanbingtao/note/476663)讲过的**全连接神经网络**是一样的，这里就不再赘述了。

# 卷积神经网络的训练

和**全连接神经网络**相比，**卷积神经网络**的训练要复杂一些。但训练的原理是一样的：利用链式求导计算损失函数对每个权重的偏导数（梯度），然后根据梯度下降公式更新权重。训练算法依然是反向传播算法。

我们先回忆一下上一篇文章[零基础入门深度学习(3) - 神经网络和反向传播算法](https://www.zybuluo.com/hanbingtao/note/476663)介绍的反向传播算法，整个算法分为三个步骤：

1. 前向计算每个神经元的**输出值** $ a_j $ （ $ j $ 表示网络的第 $ j $ 个神经元，以下同）；
2. 反向计算每个神经元的**误差项** $ \delta_j $ ， $ \delta_j $ 在有的文献中也叫做**敏感度**(sensitivity)。它实际上是网络的损失函数 $ E_d $ 对神经元**加权输入** $ net_j $ 的偏导数，即 $ \delta_j=\frac{\partial{E_d}}{\partial{net_j}} $ ；
3. 计算每个神经元连接权重 $ w_{ji} $ 的**梯度**（ $ w_{ji} $ 表示从神经元 $ i $ 连接到神经元 $ j $ 的权重），公式为 $ \frac{\partial{E_d}}{\partial{w_{ji}}}=a_i\delta_j $ ，其中， $ a_i $ 表示神经元 $ i $ 的输出。

最后，根据梯度下降法则更新每个权重即可。

对于卷积神经网络，由于涉及到**局部连接**、**下采样**的等操作，影响到了第二步**误差项** $ \delta $ 的具体计算方法，而**权值共享**影响了第三步**权重** $ w $ 的**梯度**的计算方法。接下来，我们分别介绍卷积层和Pooling层的训练算法。

## 卷积层的训练

对于卷积层，我们先来看看上面的第二步，即如何将**误差项**传递到上一层；然后再来看看第三步，即如何计算filter每个权值的**梯度**。

### 卷积层误差项的传递

#### 最简单情况下误差项的传递

我们先来考虑步长为1、输入的深度为1、filter个数为1的最简单的情况。

假设输入的大小为3\*3，filter大小为2\*2，按步长为1卷积，我们将得到2\*2的**feature map**。如下图所示：

![](https://gitee.com/zhiliangj/Typora_Img/raw/master/2256672-52295dad2641037f.png)

在上图中，为了描述方便，我们为每个元素都进行了编号。用 $ \delta^{l-1}\_{i,j} $ 表示第 $ l-1 $ 层第 $ i $ 行第 $ j $ 列的**误差项**；用 $ w_{m,n} $ 表示filter第 $ m $ 行第 $ n $ 列权重，用 $ w_{b} $ 表示filter的**偏置项**；用 $ a^{l-1}\_{i,j} $ 表示第 $ l-1 $ 层第 $ i $ 行第 $ j $ 列神经元的**输出**；用 $ net_{i,j}^{l-1} $ 表示第 $ l-1 $ 行神经元的**加权输入**；用 $ \delta^{l}\_{i,j} $ 表示第 $ l $ 层第 $ i $ 行第 $ j $ 列的**误差项**；用 $ f^{l-1} $ 表示第 $ l-1 $ 层的**激活函数**。它们之间的关系如下：

$$

\begin{align}
net^l&=conv(W^l, a^{l-1})+w_b\\
a^{l-1}_{i,j}&=f^{l-1}(net^{l-1}_{i,j})
\end{align}

$$

上式中， $ net^{l} $ 、 $ W^{l} $ 、 $ a^{l-1} $ 都是数组， $ W^l $ 是由 $ w_{m,n} $ 组成的数组， $ conv $ 表示卷积操作。

在这里，我们假设第 $ l $ 中的每个 $ \delta^{l} $ 值都已经算好，我们要做的是计算第 $ l-1 $ 层每个神经元的**误差项** $ \delta^{l-1} $ 。

根据链式求导法则：

$$

\begin{align}
\delta^{l-1}_{i,j}&=\frac{\partial{E_d}}{\partial{net^{l-1}_{i,j}}}\\
&=\frac{\partial{E_d}}{\partial{a^{l-1}_{i,j}}}\frac{\partial{a^{l-1}_{i,j}}}{\partial{net^{l-1}_{i,j}}}
\end{align}

$$

我们先求第一项 $ \frac{\partial{E\_d}}{\partial{a^{l-1}\_{i,j}}} $ 。我们先来看几个特例，然后从中总结出一般性的规律。

例1，计算 $ \frac{\partial{E\_d}}{\partial{a^{l-1}\_{1,1}}} $ ， $ a^{l-1}\_{1,1} $ 仅与 $ net^l_{1,1} $ 的计算有关：

$$

net^j_{1,1}=w_{1,1}a^{l-1}_{1,1}+w_{1,2}a^{l-1}_{1,2}+w_{2,1}a^{l-1}_{2,1}+w_{2,2}a^{l-1}_{2,2}+w_b

$$

因此：

$$

\begin{align}
\frac{\partial{E_d}}{\partial{a^{l-1}_{1,1}}}&=\frac{\partial{E_d}}{\partial{net^{l}_{1,1}}}\frac{\partial{net^{l}_{1,1}}}{\partial{a^{l-1}_{1,1}}}\\
&=\delta^l_{1,1}w_{1,1}
\end{align}

$$

例2，计算 $ \frac{\partial{E\_d}}{\partial{a^{l-1}\_{1,2}}} $ ， $ a^{l-1}\_{1,2} $ 与 $ net^{l}\_{1,1} $ 和 $ net^{l}\_{1,2} $ 的计算都有关：

$$

net^j_{1,1}=w_{1,1}a^{l-1}_{1,1}+w_{1,2}a^{l-1}_{1,2}+w_{2,1}a^{l-1}_{2,1}+w_{2,2}a^{l-1}_{2,2}+w_b\\
net^j_{1,2}=w_{1,1}a^{l-1}_{1,2}+w_{1,2}a^{l-1}_{1,3}+w_{2,1}a^{l-1}_{2,2}+w_{2,2}a^{l-1}_{2,3}+w_b\\

$$

因此：

$$

\begin{align}
\frac{\partial{E_d}}{\partial{a^{l-1}_{1,2}}}&=\frac{\partial{E_d}}{\partial{net^{l}_{1,1}}}\frac{\partial{net^{l}_{1,1}}}{\partial{a^{l-1}_{1,2}}}+\frac{\partial{E_d}}{\partial{net^{l}_{1,2}}}\frac{\partial{net^{l}_{1,2}}}{\partial{a^{l-1}_{1,2}}}\\
&=\delta^l_{1,1}w_{1,2}+\delta^l_{1,2}w_{1,1}
\end{align}

$$

例3，计算 $ \frac{\partial{E\_d}}{\partial{a^{l-1}\_{2,2}}} $ ， $ a^{l-1}\_{2,2} $ 与 $ net^l\_{1,1} $ 、 $ net^l_{1,2} $ 、 $ net^l_{2,1} $ 和 $ net^l_{2,2} $ 的计算都有关：

$$

net^j_{1,1}=w_{1,1}a^{l-1}_{1,1}+w_{1,2}a^{l-1}_{1,2}+w_{2,1}a^{l-1}_{2,1}+w_{2,2}a^{l-1}_{2,2}+w_b\\
net^j_{1,2}=w_{1,1}a^{l-1}_{1,2}+w_{1,2}a^{l-1}_{1,3}+w_{2,1}a^{l-1}_{2,2}+w_{2,2}a^{l-1}_{2,3}+w_b\\
net^j_{2,1}=w_{1,1}a^{l-1}_{2,1}+w_{1,2}a^{l-1}_{2,2}+w_{2,1}a^{l-1}_{3,1}+w_{2,2}a^{l-1}_{3,2}+w_b\\
net^j_{2,2}=w_{1,1}a^{l-1}_{2,2}+w_{1,2}a^{l-1}_{2,3}+w_{2,1}a^{l-1}_{3,2}+w_{2,2}a^{l-1}_{3,3}+w_b

$$

因此：

$$

\begin{align}
\frac{\partial{E_d}}{\partial{a^{l-1}_{2,2}}}&=\frac{\partial{E_d}}{\partial{net^{l}_{1,1}}}\frac{\partial{net^{l}_{1,1}}}{\partial{a^{l-1}_{2,2}}}+\frac{\partial{E_d}}{\partial{net^{l}_{1,2}}}\frac{\partial{net^{l}_{1,2}}}{\partial{a^{l-1}_{2,2}}}+\frac{\partial{E_d}}{\partial{net^{l}_{2,1}}}\frac{\partial{net^{l}_{2,1}}}{\partial{a^{l-1}_{2,2}}}+\frac{\partial{E_d}}{\partial{net^{l}_{2,2}}}\frac{\partial{net^{l}_{2,2}}}{\partial{a^{l-1}_{2,2}}}\\
&=\delta^l_{1,1}w_{2,2}+\delta^l_{1,2}w_{2,1}+\delta^l_{2,1}w_{1,2}+\delta^l_{2,2}w_{1,1}
\end{align}

$$

从上面三个例子，我们发挥一下想象力，不难发现，计算 $ \frac{\partial{E_d}}{\partial{a^{l-1}}} $ ，相当于把第 $ l $ 层的sensitive map周围补一圈0，在与180度翻转后的filter进行**cross-correlation**，就能得到想要结果，如下图所示：

![](https://gitee.com/zhiliangj/Typora_Img/raw/master/2256672-2fb37b0a3ff0e1f9.png)

因为**卷积**相当于将filter旋转180度的**cross-correlation**，因此上图的计算可以用卷积公式完美的表达：

$$

\frac{\partial{E_d}}{\partial{a_l}}=\delta^l*W^l

$$

上式中的 $ W^{l} $ 表示第 $ l $ 层的filter的权重数组。也可以把上式的卷积展开，写成求和的形式：

$$

\frac{\partial{E_d}}{\partial{a^l_{i,j}}}=\sum_m\sum_n{w^l_{m,n}\delta^l_{i+m,j+n}}

$$

现在，我们再求第二项 $ \frac{\partial{a^{l-1}\_{i,j}}}{\partial{net^{l-1}\_{i,j}}} $ 。因为

$$

a^{l-1}_{i,j}=f(net^{l-1}_{i,j})

$$

所以这一项极其简单，仅求激活函数 $ f $ 的导数就行了。

$$

\frac{\partial{a^{l-1}_{i,j}}}{\partial{net^{l-1}_{i,j}}}=f'(net^{l-1}_{i,j})

$$

将第一项和第二项组合起来，我们得到最终的公式：

$$

\begin{align}
\delta^{l-1}_{i,j}&=\frac{\partial{E_d}}{\partial{net^{l-1}_{i,j}}}\\
&=\frac{\partial{E_d}}{\partial{a^{l-1}_{i,j}}}\frac{\partial{a^{l-1}_{i,j}}}{\partial{net^{l-1}_{i,j}}}\\
&=\sum_m\sum_n{w^l_{m,n}\delta^l_{i+m,j+n}}f'(net^{l-1}_{i,j})
\end{align}\tag{5.1}

$$

也可以将**式5.1**写成卷积的形式：

$$

\delta^{l-1}=\delta^l*W^l\circ f'(net^{l-1})\tag{5.2}

$$

其中，符号 $ \circ $ 表示**element-wise product**，即将矩阵中每个对应元素相乘。注意**式5.2**中的 $ \delta^{l-1} $ 、 $ \delta^{l} $ 、 $ net^{l-1} $ 都是**矩阵**。

以上就是步长为1、输入的深度为1、filter个数为1的最简单的情况，卷积层误差项传递的算法。下面我们来推导一下步长为S的情况。

#### 卷积步长为S时的误差传递

我们先来看看步长为S与步长为1的差别。

![](https://gitee.com/zhiliangj/Typora_Img/raw/master/2256672-754f37eb7603e99f.png)

如上图，上面是步长为1时的卷积结果，下面是步长为2时的卷积结果。我们可以看出，因为步长为2，得到的feature map跳过了步长为1时相应的部分。因此，当我们反向计算**误差项**时，我们可以对步长为S的sensitivity map相应的位置进行补0，将其『还原』成步长为1时的sensitivity map，再用**式5.2**进行求解。

#### 输入层深度为D时的误差传递

当输入深度为D时，filter的深度也必须为D， $ l-1 $ 层 $ d_i $ 的通道只与filter的 $ d_i $ 通道的权重进行计算。因此，反向计算**误差项**时，我们可以使用**式5.2**，用filter的第 $ d_i $ 通道权重对第 $ l $ 层sensitivity map进行卷积，得到第 $ l-1 $ 层通道的sensitivity map。如下图所示：

![](https://gitee.com/zhiliangj/Typora_Img/raw/master/2256672-af2da9701a03dc3c.png)

#### filter数量为N时的误差传递

filter数量为N时，输出层的深度也为N，第 $ i $ 个filter卷积产生输出层的第 $ i $ 个feature map。由于第 $ l-1 $ 层**每个加权输入** $ net^{l-1}_{d, i,j} $ 都同时影响了第 $ l $ 层所有feature map的输出值，因此，反向计算**误差项**时，需要使用全导数公式。也就是，我们先使用第 $ d $ 个filter对第 $ l $ 层相应的第 $ d $ 个sensitivity map进行卷积，得到一组N个 $ l-1 $ 层的偏sensitivity map。依次用每个filter做这种卷积，就得到D组偏sensitivity map。最后在各组之间将N个偏sensitivity map **按元素相加**，得到最终的N个层 $ l-1 $ 的sensitivity map：

$$

\delta^{l-1}=\sum_{d=0}^D\delta_d^l*W_d^l\circ f'(net^{l-1})\tag{5.3}

$$

以上就是卷积层误差项传递的算法，如果读者还有所困惑，可以参考后面的代码实现来理解。

### 卷积层filter权重梯度的计算

我们要在得到第 $ l $ 层sensitivity map的情况下，计算filter的权重的梯度，由于卷积层是**权重共享**的，因此梯度的计算稍有不同。

![](https://gitee.com/zhiliangj/Typora_Img/raw/master/2256672-afe6d3a863b7cbcc.png)

如上图所示， $ a_{i,j}^{l} $ 是第 $ l-1 $ 层的输出， $ w_{i,j} $ 是第 $ l $ 层filter的权重， $ \delta^{l}\_{i,j} $ 是第 $ l $ 层的sensitivity map。我们的任务是计算 $ w_{i,j} $ 的梯度，即 $ \frac{\partial{E_d}}{\partial{w_{i,j}}} $ 。

为了计算偏导数，我们需要考察权重 $ w_{i,j} $ 对 $ E_d $ 的影响。权重项 $ w_{i,j} $ 通过影响 $ net^{l}\_{i,j} $ 的值，进而影响 $ E_d $ 。我们仍然通过几个具体的例子来看权重项 $ w_{i,j} $ 对 $ net^{l}\_{i,j} $ 的影响，然后再从中总结出规律。

例1，计算 $ \frac{\partial{E_d}}{\partial{w_{1,1}}} $ ：

$$

net^j_{1,1}=w_{1,1}a^{l-1}_{1,1}+w_{1,2}a^{l-1}_{1,2}+w_{2,1}a^{l-1}_{2,1}+w_{2,2}a^{l-1}_{2,2}+w_b\\
net^j_{1,2}=w_{1,1}a^{l-1}_{1,2}+w_{1,2}a^{l-1}_{1,3}+w_{2,1}a^{l-1}_{2,2}+w_{2,2}a^{l-1}_{2,3}+w_b\\
net^j_{2,1}=w_{1,1}a^{l-1}_{2,1}+w_{1,2}a^{l-1}_{2,2}+w_{2,1}a^{l-1}_{3,1}+w_{2,2}a^{l-1}_{3,2}+w_b\\
net^j_{2,2}=w_{1,1}a^{l-1}_{2,2}+w_{1,2}a^{l-1}_{2,3}+w_{2,1}a^{l-1}_{3,2}+w_{2,2}a^{l-1}_{3,3}+w_b

$$

从上面的公式看出，由于**权值共享**，权值 $ w_{1,1} $ 对所有的 $ net_{i,j}^{l} $ 都有影响。 $ E_d $ 是 $ net_{1,1}^{l} $ 、 $ net_{1,2}^{l} $ 、 $ net_{2,1}^{l} $ ...的函数，而 $ net_{1,1}^{l} $ 、 $ net_{1,2}^{l} $ 、 $ net_{2,1}^{l} $ ...又是 $ w_{1,1} $ 的函数，根据**全导数**公式，计算 $ \frac{\partial{E_d}}{\partial{w_{1,1}}} $ 就是要把每个偏导数都加起来：

$$

\begin{align}
\frac{\partial{E_d}}{\partial{w_{1,1}}}&=\frac{\partial{E_d}}{\partial{net^{l}_{1,1}}}\frac{\partial{net^{l}_{1,1}}}{\partial{w_{1,1}}}+\frac{\partial{E_d}}{\partial{net^{l}_{1,2}}}\frac{\partial{net^{l}_{1,2}}}{\partial{w_{1,1}}}+\frac{\partial{E_d}}{\partial{net^{l}_{2,1}}}\frac{\partial{net^{l}_{2,1}}}{\partial{w_{1,1}}}+\frac{\partial{E_d}}{\partial{net^{l}_{2,2}}}\frac{\partial{net^{l}_{2,2}}}{\partial{w_{1,1}}}\\
&=\delta^l_{1,1}a^{l-1}_{1,1}+\delta^l_{1,2}a^{l-1}_{1,2}+\delta^l_{2,1}a^{l-1}_{2,1}+\delta^l_{2,2}a^{l-1}_{2,2}
\end{align}

$$

例2，计算 $ \frac{\partial{E_d}}{\partial{w_{1,2}}} $ ：

通过查看 $ w_{1,2} $ 与 $ net_{i,j}^{l} $ 的关系，我们很容易得到：

$$

\frac{\partial{E_d}}{\partial{w_{1,2}}}=\delta^l_{1,1}a^{l-1}_{1,2}+\delta^l_{1,2}a^{l-1}_{1,3}+\delta^l_{2,1}a^{l-1}_{2,2}+\delta^l_{2,2}a^{l-1}_{2,3}

$$

实际上，每个**权重项**都是类似的，我们不一一举例了。现在，是我们再次发挥想象力的时候，我们发现计算 $ \frac{\partial{E_d}}{\partial{w_{i,j}}} $ 规律是：

$$

\frac{\partial{E_d}}{\partial{w_{i,j}}}=\sum_m\sum_n\delta_{m,n}a^{l-1}_{i+m,j+n}

$$

也就是用sensitivity map作为卷积核，在input上进行**cross-correlation**，如下图所示：

![](https://gitee.com/zhiliangj/Typora_Img/raw/master/2256672-aeba8c8666a22e72.png)

最后，我们来看一看偏置项的梯度 $ \frac{\partial{E_d}}{\partial{w_b}} $ 。通过查看前面的公式，我们很容易发现：

$$

\begin{align}
\frac{\partial{E_d}}{\partial{w_b}}&=\frac{\partial{E_d}}{\partial{net^{l}_{1,1}}}\frac{\partial{net^{l}_{1,1}}}{\partial{w_b}}+\frac{\partial{E_d}}{\partial{net^{l}_{1,2}}}\frac{\partial{net^{l}_{1,2}}}{\partial{w_b}}+\frac{\partial{E_d}}{\partial{net^{l}_{2,1}}}\frac{\partial{net^{l}_{2,1}}}{\partial{w_b}}+\frac{\partial{E_d}}{\partial{net^{l}_{2,2}}}\frac{\partial{net^{l}_{2,2}}}{\partial{w_b}}\\
&=\delta^l_{1,1}+\delta^l_{1,2}+\delta^l_{2,1}+\delta^l_{2,2}\\
&=\sum_i\sum_j\delta^l_{i,j}
\end{align}

$$

也就是**偏置项**的**梯度**就是sensitivity map所有**误差项**之和。

对于步长为S的卷积层，处理方法与传递**误差项**是一样的，首先将sensitivity map『还原』成步长为1时的sensitivity map，再用上面的方法进行计算。

获得了所有的**梯度**之后，就是根据**梯度下降算法**来更新每个权重。这在前面的文章中已经反复写过，这里就不再重复了。

至此，我们已经解决了卷积层的训练问题，接下来我们看一看Pooling层的训练。

## Pooling层的训练

无论max pooling还是mean pooling，都没有需要学习的参数。因此，在**卷积神经网络**的训练中，Pooling层需要做的仅仅是将**误差项**传递到上一层，而没有**梯度**的计算。

### Max Pooling误差项的传递

如下图，假设第 $ l-1 $ 层大小为4\*4，pooling filter大小为2\*2，步长为2，这样，max pooling之后，第 $ l $ 层大小为2\*2。假设第 $ l $ 层的 $ \delta $ 值都已经计算完毕，我们现在的任务是计算第 $ l-1 $ 层的 $ \delta $ 值。

![](https://gitee.com/zhiliangj/Typora_Img/raw/master/2256672-a30c883f19db53b4.png)

我们用 $ net_{i,j}^{l-1} $ 表示第 $ l-1 $ 层的**加权输入**； $ net_{i,j}^{l} $ 用表示第 $ l $ 层的**加权输入**。我们先来考察一个具体的例子，然后再总结一般性的规律。对于max pooling：

$$

net^l_{1,1}=max(net^{l-1}_{1,1},net^{l-1}_{1,2},net^{l-1}_{2,1},net^{l-1}_{2,2})

$$

也就是说，只有区块中最大的 $ net_{i,j}^{l-1} $ 才会对
$$
net_{i,j}^{l}
$$
的值产生影响。我们假设最大的值是
$$
net_{1,1}^{l-1}
$$
，则上式相当于：

$$

net^l_{1,1}=net^{l-1}_{1,1}

$$

那么，我们不难求得下面几个偏导数：

$$

\begin{align}
\frac{\partial{net^l_{1,1}}}{\partial{net^{l-1}_{1,1}}}=1\\
\frac{\partial{net^l_{1,1}}}{\partial{net^{l-1}_{1,2}}}=0\\
\frac{\partial{net^l_{1,1}}}{\partial{net^{l-1}_{2,1}}}=0\\
\frac{\partial{net^l_{1,1}}}{\partial{net^{l-1}_{2,2}}}=0
\end{align}

$$

因此：

$$

\begin{align}
\delta^{l-1}_{1,1}&=\frac{\partial{E_d}}{\partial{net^{l-1}_{1,1}}}\\
&=\frac{\partial{E_d}}{\partial{net^{l}_{1,1}}}\frac{\partial{net^{l}_{1,1}}}{\partial{net^{l-1}_{1,1}}}\\
&=\delta^{l}_{1,1}\\
\end{align}

$$

而：

$$

\begin{align}
\delta^{l-1}_{1,2}&=\frac{\partial{E_d}}{\partial{net^{l-1}_{1,2}}}\\
&=\frac{\partial{E_d}}{\partial{net^{l}_{1,1}}}\frac{\partial{net^{l}_{1,1}}}{\partial{net^{l-1}_{1,2}}}\\
&=0\\
\delta^{l-1}_{2,1}&=\frac{\partial{E_d}}{\partial{net^{l-1}_{2,1}}}\\
&=\frac{\partial{E_d}}{\partial{net^{l}_{1,1}}}\frac{\partial{net^{l}_{1,1}}}{\partial{net^{l-1}_{2,1}}}\\
&=0\\
\delta^{l-1}_{1,1}&=\frac{\partial{E_d}}{\partial{net^{l-1}_{2,2}}}\\
&=\frac{\partial{E_d}}{\partial{net^{l}_{1,1}}}\frac{\partial{net^{l}_{1,1}}}{\partial{net^{l-1}_{2,2}}}\\
&=0\\
\end{align}

$$

现在，我们发现了规律：对于max pooling，下一层的**误差项**的值会原封不动的传递到上一层对应区块中的最大值所对应的神经元，而其他神经元的**误差项**的值都是0。如下图所示(假设 $ a_{1,1}^{l-1} $ 、 $ a_{1,4}^{l-1} $ 、 $ a_{4,1}^{l-1} $ 、 $ a_{4,4}^{l-1} $ 为所在区块中的最大输出值)：

![](https://gitee.com/zhiliangj/Typora_Img/raw/master/2256672-af77e98c09fad84c.png)

### Mean Pooling误差项的传递

我们还是用前面屡试不爽的套路，先研究一个特殊的情形，再扩展为一般规律。

![](https://gitee.com/zhiliangj/Typora_Img/raw/master/2256672-a30c883f19db53b4.png)

如上图，我们先来考虑计算 $ \delta_{1,1}^{l-1} $ 。我们先来看看 $ net_{1,1}^{l-1} $ 如何影响 $ net_{1,1}^{l} $ 。

$$

net^j_{1,1}=\frac{1}{4}(net^{l-1}_{1,1}+net^{l-1}_{1,2}+net^{l-1}_{2,1}+net^{l-1}_{2,2})

$$

根据上式，我们一眼就能看出来：

$$

\frac{\partial{net^l_{1,1}}}{\partial{net^{l-1}_{1,1}}}=\frac{1}{4}\\
\frac{\partial{net^l_{1,1}}}{\partial{net^{l-1}_{1,2}}}=\frac{1}{4}\\
\frac{\partial{net^l_{1,1}}}{\partial{net^{l-1}_{2,1}}}=\frac{1}{4}\\
\frac{\partial{net^l_{1,1}}}{\partial{net^{l-1}_{2,2}}}=\frac{1}{4}\\

$$

所以，根据链式求导法则，我们不难算出：

$$

\begin{align}
\delta^{l-1}_{1,1}&=\frac{\partial{E_d}}{\partial{net^{l-1}_{1,1}}}\\
&=\frac{\partial{E_d}}{\partial{net^{l}_{1,1}}}\frac{\partial{net^{l}_{1,1}}}{\partial{net^{l-1}_{1,1}}}\\
&=\frac{1}{4}\delta^{l}_{1,1}\\
\end{align}

$$

同样，我们可以算出 $ \delta^{l-1}\_{1,2} $ 、 $ \delta^{l-1}\_{2,1} $ 、 $ \delta^{l-1}\_{2,2} $ ：

$$

\begin{align}
\delta^{l-1}_{1,2}&=\frac{\partial{E_d}}{\partial{net^{l-1}_{1,2}}}\\
&=\frac{\partial{E_d}}{\partial{net^{l}_{1,1}}}\frac{\partial{net^{l}_{1,1}}}{\partial{net^{l-1}_{1,2}}}\\
&=\frac{1}{4}\delta^{l}_{1,1}\\
\delta^{l-1}_{2,1}&=\frac{\partial{E_d}}{\partial{net^{l-1}_{2,1}}}\\
&=\frac{\partial{E_d}}{\partial{net^{l}_{1,1}}}\frac{\partial{net^{l}_{1,1}}}{\partial{net^{l-1}_{2,1}}}\\
&=\frac{1}{4}\delta^{l}_{1,1}\\
\delta^{l-1}_{2,2}&=\frac{\partial{E_d}}{\partial{net^{l-1}_{2,2}}}\\
&=\frac{\partial{E_d}}{\partial{net^{l}_{1,1}}}\frac{\partial{net^{l}_{1,1}}}{\partial{net^{l-1}_{2,2}}}\\
&=\frac{1}{4}\delta^{l}_{1,1}\\
\end{align}

$$

现在，我们发现了规律：对于mean pooling，下一层的**误差项**的值会**平均分配**到上一层对应区块中的所有神经元。如下图所示：

![](https://gitee.com/zhiliangj/Typora_Img/raw/master/2256672-c3a6772cb07b416a.png)

上面这个算法可以表达为高大上的**克罗内克积(Kronecker product)**的形式，有兴趣的读者可以研究一下。

$$

\delta^{l-1} = \delta^l\otimes(\frac{1}{n^2})_{n\times n}

$$

其中， $ n $ 是pooling层filter的大小， $ \delta^{l-1} $ 、 $ \delta^{l} $ 都是矩阵。

至此，我们已经把**卷积层**、**Pooling层**的训练算法介绍完毕，加上上一篇文章讲的**全连接层**训练算法，您应该已经具备了编写**卷积神经网络**代码所需要的知识。为了加深对知识的理解，接下来，我们将展示如何实现一个简单的**卷积神经网络**。



转载自：[零基础入门深度学习(4) - 卷积神经网络](https://www.zybuluo.com/hanbingtao/note/485480)

