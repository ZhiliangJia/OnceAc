---
layout: post
title: 神经网络和反向传播算法
date: 2020-12-05
author: Zhiliang 
tags: [Machine Learning]
toc: true
mathjax: true
---

神经网络其实就是按照**一定规则**连接起来的多个**神经元**。反向传播算法其实就是链式求导法则的应用。

<!-- more -->

# 神经元

神经元和感知器本质上是一样的，只不过我们说感知器的时候，它的激活函数是**阶跃函数**；而当我们说神经元时，激活函数往往选择为sigmoid函数或tanh函数。如下图所示：

![](https://gitee.com/zhiliangj/Typora_Img/raw/master/2256672-49f06e2e9d3eb29f.gif)

> Sigmoid曲线、tanh曲线....：
>
> ![img](https://gitee.com/zhiliangj/Typora_Img/raw/master/749674-cdc2da4f770158ca.png)

计算一个神经元的输出的方法和计算一个感知器的输出是一样的。假设神经元的输入是向量$\vec{x}$，权重向量是$\vec{w}$(偏置项是$w_0$)，激活函数是sigmoid函数，则其输出$y$：

$$
y=sigmoid(\vec{w}^T\centerdot\vec{x})\qquad \tag{1.1}
$$

sigmoid函数的定义如下：

$$
sigmoid(x)=\frac{1}{1+e^{-x}} \tag{1.2}
$$

将其带入前面的式子，得到

$$
y=\frac{1}{1+e^{-\vec{w}^T\centerdot\vec{x}}} \tag{1.3}
$$

sigmoid函数是一个非线性函数，值域是(0,1)。函数图像如上图。

sigmoid函数的导数是：

$$
\begin{align}
&令y=sigmoid(x)\\
&则y'=y(1-y)
\end{align}\tag{1.4}
$$

可以看到，sigmoid函数的导数非常有趣，它可以用sigmoid函数自身来表示。这样，一旦计算出sigmoid函数的值，计算它的导数的值就非常方便。

# 神经网络是啥

![](https://gitee.com/zhiliangj/Typora_Img/raw/master/2256672-92111b104ce0d571.jpeg)

神经网络其实就是按照**一定规则**连接起来的多个**神经元**。上图展示了一个**全连接(full connected, FC)**神经网络，通过观察上面的图，我们可以发现它的规则包括：

- 神经元按照**层**来布局。最左边的层叫做**输入层**，负责接收输入数据；最右边的层叫**输出层**，我们可以从这层获取神经网络输出数据。输入层和输出层之间的层叫做**隐藏层**，因为它们对于外部来说是不可见的。
- 同一层的神经元之间没有连接。
- 第N层的每个神经元和第N-1层的**所有**神经元相连(这就是full connected的含义)，第N-1层神经元的输出就是第N层神经元的输入。
- 每个连接都有一个**权值**。

上面这些规则定义了全连接神经网络的结构。事实上还存在很多其它结构的神经网络，比如卷积神经网络(CNN)、循环神经网络(RNN)，他们都具有不同的连接规则。

# 计算神经网络的输出

神经网络实际上就是一个输入向量$\vec{x}$到输出向量$\vec{y}$的函数，即：

$$
\vec{y} = f_{network}(\vec{x})\tag{3.1}
$$

根据输入计算神经网络的输出，需要首先将输入向量$\vec{x}$的每个元素$x_i$的值赋给神经网络的输入层的对应神经元，然后根据**式1.1**依次向前计算每一层的每个神经元的值，直到最后一层输出层的所有神经元的值计算完毕。最后，将输出层每个神经元的值串在一起就得到了输出向量$\vec{y}$。

接下来举一个例子来说明这个过程，我们先给神经网络的每个单元写上编号。

![](https://gitee.com/zhiliangj/Typora_Img/raw/master/2256672-bfbb364740f898d1.png)

如上图，输入层有三个节点，我们将其依次编号为1、2、3；隐藏层的4个节点，编号依次为4、5、6、7；最后输出层的两个节点编号为8、9。因为我们这个神经网络是**全连接**网络，所以可以看到每个节点都和**上一层的所有节点**有连接。比如，我们可以看到隐藏层的节点4，它和输入层的三个节点1、2、3之间都有连接，其连接上的权重分别为$w_{41},w_{42},w_{43}$。那么，我们怎样计算节点4的输出值$a_4$呢？

为了计算节点4的输出值，我们必须先得到其所有上游节点（也就是节点1、2、3）的输出值。节点1、2、3是**输入层**的节点，所以，他们的输出值就是输入向量$\vec{x}$本身。按照上图画出的对应关系，可以看到节点1、2、3的输出值分别是$x_1,x_2,x_3$。我们要求**输入向量的维度和输入层神经元个数相同**，而输入向量的某个元素对应到哪个输入节点是可以自由决定的，你偏非要把$x_1$赋值给节点2也是完全没有问题的，但这样除了把自己弄晕之外，并没有什么价值。

一旦我们有了节点1、2、3的输出值，我们就可以根据**式1.1**计算节点4的输出值$a_4$：

$$
\begin{align}
a_4&=sigmoid(\vec{w}^T\centerdot\vec{x})\\
&=sigmoid(w_{41}x_1+w_{42}x_2+w_{43}x_3+w_{4b})
\end{align}\tag{3.2}
$$

上式的$w_{4b}$是节点4的**偏置项**，<u>图中没有画出来</u>。而$w_{41},w_{42},w_{43}$分别为节点1、2、3到节点4连接的权重，在给权重$w_{ji}$编号时，我们把目标节点的编号$j$放在前面，把源节点的编号$i$放在后面。

同样，我们可以继续计算出节点5、6、7的输出值$a_5,a_6,a_7$。这样，隐藏层的4个节点的输出值就计算完成了，我们就可以接着计算输出层的节点8的输出值$y_1$：

$$
\begin{align}
y_1&=sigmoid(\vec{w}^T\centerdot\vec{a})\\
&=sigmoid(w_{84}a_4+w_{85}a_5+w_{86}a_6+w_{87}a_7+w_{8b})
\end{align}\tag{3.3}
$$

同理，我们还可以计算出的值$y_2$。这样输出层所有节点的输出值计算完毕，我们就得到了在输入向量 $\vec{x}=\begin{bmatrix}x_1\\\\x_2\\\\x_3\end{bmatrix}$ 时，神经网络的输出向量 $\vec{y}=\begin{bmatrix}y_1\\\y_2\end{bmatrix}$ 。这里我们也看到，**输出向量的维度和输出层神经元个数相同**。

## 神经网络的矩阵表示

神经网络的计算如果用矩阵来表示会很方便（当然逼格也更高），我们先来看看隐藏层的矩阵表示。

首先我们把隐藏层4个节点的计算依次排列出来：

$$
a_4=sigmoid(w_{41}x_1+w_{42}x_2+w_{43}x_3+w_{4b})\\
a_5=sigmoid(w_{51}x_1+w_{52}x_2+w_{53}x_3+w_{5b})\\
a_6=sigmoid(w_{61}x_1+w_{62}x_2+w_{63}x_3+w_{6b})\\
a_7=sigmoid(w_{71}x_1+w_{72}x_2+w_{73}x_3+w_{7b})\\ \tag{3.4}
$$

接着，定义网络的输入向量$\vec{x}$和隐藏层每个节点的权重向量$\vec{w_j}$。令

$$
\begin{align}
\vec{x}&=\begin{bmatrix}x_1\\x_2\\x_3\\1\end{bmatrix}\\
\vec{w}_4&=[w_{41},w_{42},w_{43},w_{4b}]\\
\vec{w}_5&=[w_{51},w_{52},w_{53},w_{5b}]\\
\vec{w}_6&=[w_{61},w_{62},w_{63},w_{6b}]\\
\vec{w}_7&=[w_{71},w_{72},w_{73},w_{7b}]\\
f&=sigmoid
\end{align}\tag{3.5}
$$

代入到前面的一组式子，得到：

$$
\begin{align}
a_4&=f(\vec{w_4}\centerdot\vec{x})\\
a_5&=f(\vec{w_5}\centerdot\vec{x})\\
a_6&=f(\vec{w_6}\centerdot\vec{x})\\
a_7&=f(\vec{w_7}\centerdot\vec{x})
\end{align}\tag{3.6}
$$

现在，我们把上述计算的四个式子$a_4,a_5,a_6,a_7$写到一个矩阵里面，每个式子作为矩阵的一行，就可以利用矩阵来表示它们的计算了。令

$$
\vec{a}=
\begin{bmatrix}
a_4 \\
a_5 \\
a_6 \\
a_7 \\
\end{bmatrix},\qquad W=
\begin{bmatrix}
\vec{w}_4 \\
\vec{w}_5 \\
\vec{w}_6 \\
\vec{w}_7 \\
\end{bmatrix}=
\begin{bmatrix}
w_{41},w_{42},w_{43},w_{4b} \\
w_{51},w_{52},w_{53},w_{5b} \\
w_{61},w_{62},w_{63},w_{6b} \\
w_{71},w_{72},w_{73},w_{7b} \\
\end{bmatrix}
,\qquad f(
\begin{bmatrix}
x_1\\
x_2\\
x_3\\
.\\
.\\
.\\
\end{bmatrix})=
\begin{bmatrix}
f(x_1)\\
f(x_2)\\
f(x_3)\\
.\\
.\\
.\\
\end{bmatrix}\tag{3.7}
$$

带入前面的一组式子，得到

$$
\vec{a}=f(W\centerdot\vec{x})\qquad \tag{3.8}
$$

在式3.8中，$f$是激活函数，本例中是$sigmoid$函数；$W$是某一层的权重矩阵；$\vec{x}$是某层的输入向量；$\vec{a}$是某层的输出向量。**式3.8**说明神经网络的每一层的作用实际上就是先将输入向量**左乘**一个数组进行线性变换，得到一个新的向量，然后再对这个向量**逐元素**应用一个激活函数。

每一层的算法都是一样的。比如，对于包含一个输入层，一个输出层和三个隐藏层的神经网络，我们假设其权重矩阵分别为$W_1,W_2,W_3,W_4$，每个隐藏层的输出分别是$\vec{a}_1,\vec{a}_2,\vec{a}_3$，神经网络的输入为$\vec{x}$，神经网络的输入为$\vec{y}$，如下图所示：

![](https://gitee.com/zhiliangj/Typora_Img/raw/master/2256672-c1388dc8fdcce427.png)

则每一层的输出向量的计算可以表示为：

$$
\begin{align}
&\vec{a}_1=f(W_1\centerdot\vec{x})\\
&\vec{a}_2=f(W_2\centerdot\vec{a}_1)\\
&\vec{a}_3=f(W_3\centerdot\vec{a}_2)\\
&\vec{y}=f(W_4\centerdot\vec{a}_3)\\
\end{align}
$$

这就是神经网络输出值的计算方法。

# 神经网络的训练

现在，我们需要知道一个神经网络的每个连接上的权值是如何得到的。我们可以说神经网络是一个**模型**，那么这些权值就是模型的**参数**，也就是模型要学习的东西。然而，<u>一个神经网络的连接方式、网络的层数、每层的节点数这些参数，则不是学习出来的，而是人为事先设置的。</u>对于这些人为设置的参数，我们称之为**超参数(Hyper-Parameters)**。

接下来，我们将要介绍神经网络的训练算法：<u>反向传播算法</u>。

## 反向传播算法(Back Propagation)

我们首先直观的介绍反向传播算法，最后再来介绍这个算法的推导。当然读者也可以完全跳过推导部分，因为即使不知道如何推导，也不影响你写出来一个神经网络的训练代码。事实上，现在神经网络成熟的开源实现多如牛毛，除了练手之外，你可能都没有机会需要去写一个神经网络。

我们以**监督学习**为例来解释反向传播算法。在[零基础入门深度学习(2) - 线性单元和梯度下降](https://www.zybuluo.com/hanbingtao/note/448086)一文中我们介绍了什么是**监督学习**，如果忘记了可以再看一下。另外，我们设神经元的激活函数$f$为$sigmoid$函数(不同激活函数的计算公式不同，详情见[反向传播算法的推导](https://www.zybuluo.com/hanbingtao/note/476663#an1)一节)。

我们假设每个训练样本为$(\vec{x},\vec{t})$，其中向量$\vec{x}$是训练样本的特征，而$\vec{t}$是样本的目标值。

![](https://gitee.com/zhiliangj/Typora_Img/raw/master/2256672-6f27ced45cf5c0d8.png)

首先，我们根据上一节介绍的算法，用样本的特征$\vec{x}$，计算出神经网络中每个隐藏层节点的输出$a_i$，以及输出层每个节点的输出$y_i$。

然后，我们按照下面的方法计算出每个节点的误差项$\delta_i$：

- 对于输出层节点$i$，

$$
\delta_i=y_i(1-y_i)(t_i-y_i)\qquad \tag{4.1}
$$

其中，$\delta_i$是节点$i$的误差项，$y_i$是节点$i$的**输出值**，$t_i$是样本对应于节点$i$的**目标值**。举个例子，根据上图，对于输出层节点8来说，它的输出值是$y_1$，而样本的目标值是$t_1$，带入上面的公式得到节点8的误差项应该是：

$$
\delta_8=y_1(1-y_1)(t_1-y_1) \tag{4.2}
$$

- 对于隐藏层节点，

$$
\delta_i=a_i(1-a_i)\sum_{k\in{outputs}}w_{ki}\delta_k\qquad \tag{4.3}
$$

其中，$a_i$是节点$i$的输出值，$w_{ki}$是节点$i$到它的下一层节点$k$的连接的权重，$\delta_k$是节点$i$的下一层节点$k$的误差项。例如，对于隐藏层节点4来说，计算方法如下：

$$
\delta_4=a_4(1-a_4)(w_{84}\delta_8+w_{94}\delta_9) \tag{4.4}
$$

最后，更新每个连接上的权值：

$$
w_{ji}\gets w_{ji}+\eta\delta_jx_{ji}\qquad \tag{4.5}
$$

其中，$w_{ji}$是节点$i$到节点$j$的权重，$\eta$是一个成为**学习速率**的常数，$\delta_j$是节点$j$的误差项，$x_{ji}$是节点$i$传递给节点$j$的输入。例如，权重$w_{84}$的更新方法如下：

$$
w_{84}\gets w_{84}+\eta\delta_8 a_4\tag{4.6}
$$

类似的，权重$w_{41}$的更新方法如下：

$$
w_{41}\gets w_{41}+\eta\delta_4 x_1\tag{4.7}
$$

偏置项的输入值永远为1。例如，节点4的偏置项$w_{4b}$应该按照下面的方法计算：

$$
w_{4b}\gets w_{4b}+\eta\delta_4\tag{4.8}
$$

我们已经介绍了神经网络每个节点误差项的计算和权重更新方法。显然，计算一个节点的误差项，需要先计算每个与其相连的下一层节点的误差项。这就要求误差项的计算顺序必须是从输出层开始，然后反向依次计算每个隐藏层的误差项，直到与输入层相连的那个隐藏层。这就是反向传播算法的名字的含义。当所有节点的误差项计算完毕后，我们就可以根据**式4.5**来更新所有的权重。

以上就是基本的反向传播算法，并不是很复杂，您弄清楚了么？

## 反向传播算法的推导

反向传播算法其实就是链式求导法则的应用。然而，这个如此简单且显而易见的方法，却是在Roseblatt提出感知器算法将近30年之后才被发明和普及的。对此，Bengio这样回应道：

> 很多看似显而易见的想法只有在事后才变得显而易见。

接下来，我们用链式求导法则来推导反向传播算法，也就是上一小节的**式4.1**、**式4.3**、**式4.5**。

> - 4.1
>
> $$
> \delta_i=y_i(1-y_i)(t_i-y_i)\qquad \tag{4.1}
> $$
>
> - 4.3
>
> $$
> \delta_i=a_i(1-a_i)\sum_{k\in{outputs}}w_{ki}\delta_k\qquad \tag{4.3}
> $$
>
> - 4.5
>
> $$
> w_{ji}\gets w_{ji}+\eta\delta_jx_{ji}\qquad \tag{4.5}
> $$

***前方高能预警——接下来是数学公式重灾区，读者可以酌情阅读，不必强求。\***

按照机器学习的通用套路，我们先确定神经网络的目标函数，然后用**随机梯度下降**优化算法去求目标函数最小值时的参数值。

我们取网络所有输出层节点的误差平方和作为目标函数：

$$
E_d\equiv\frac{1}{2}\sum_{i\in outputs}(t_i-y_i)^2\tag{4.9}
$$

其中，$E_d$表示是样本$d$的误差。

然后，我们用文章[零基础入门深度学习(2) - 线性单元和梯度下降](https://www.zybuluo.com/hanbingtao/note/448086)中介绍的**随机梯度下降**算法对目标函数进行优化：

$$
w_{ji}\gets w_{ji}-\eta\frac{\partial{E_d}}{\partial{w_{ji}}}\tag{4.10}
$$

随机梯度下降算法也就是需要求出误差$E_d$对于每个权重的$w_{ji}$偏导数（也就是梯度），怎么求呢？

![](https://gitee.com/zhiliangj/Typora_Img/raw/master/2256672-6f27ced45cf5c0d8.png)

观察上图，我们发现权重$w_{ji}$仅能通过影响节点$j$的输入值影响网络的其它部分，设${net}_j$是节点$j$的**加权输入**，即

$$
\begin{align}
net_j&=\vec{w_j}\centerdot\vec{x_j}\\
&=\sum_{i}{w_{ji}}x_{ji}
\end{align}\tag{4.11}
$$

$E_d$是${net}_j$的函数，而${net}_j$是${w}_{ji}$的函数。根据链式求导法则，可以得到：

$$
\begin{align}
\frac{\partial{E_d}}{\partial{w_{ji}}}&=\frac{\partial{E_d}}{\partial{net_j}}\frac{\partial{net_j}}{\partial{w_{ji}}}\\
&=\frac{\partial{E_d}}{\partial{net_j}}\frac{\partial{\sum_{i}{w_{ji}}x_{ji}}}{\partial{w_{ji}}}\\
&=\frac{\partial{E_d}}{\partial{net_j}}x_{ji}
\end{align}\tag{4.12}
$$

上式中，$x_{ji}$是节点$i$传递给节点$j$的输入值，也就是$i$节点的输出值。

对于$\frac{\partial{E_d}}{\partial{net_j}}$的推导，需要区分**输出层**和**隐藏层**两种情况。

### 输出层权值训练

对于**输出层**来说，${net}_j$仅能通过节点$j$的输出值$y_j$来影响网络其它部分，也就是说$E_d$是$y_j$的函数，而$y_j$是$net_j$的函数，其中$y_j=sigmoid(net_j)$。所以我们可以再次使用链式求导法则：

$$
\begin{align}
\frac{\partial{E_d}}{\partial{net_j}}&=\frac{\partial{E_d}}{\partial{y_j}}\frac{\partial{y_j}}{\partial{net_j}}\\
\end{align}\tag{4.13}
$$

考虑上式第一项:

$$
\begin{align}
\frac{\partial{E_d}}{\partial{y_j}}&=\frac{\partial}{\partial{y_j}}\frac{1}{2}\sum_{i\in outputs}(t_i-y_i)^2\\
&=\frac{\partial}{\partial{y_j}}\frac{1}{2}(t_j-y_j)^2\\
&=-(t_j-y_j)
\end{align}\tag{4.14}
$$

考虑上式第二项：

$$
\begin{align}
\frac{\partial{y_j}}{\partial{net_j}}&=\frac{\partial sigmoid(net_j)}{\partial{net_j}}\\
&=y_j(1-y_j)\\
\end{align}\tag{4.15}
$$

将第一项和第二项带入，得到：

$$
\frac{\partial{E_d}}{\partial{net_j}}=-(t_j-y_j)y_j(1-y_j)\tag{4.16}
$$

如果令$\delta_j=-\frac{\partial{E_d}}{\partial{net_j}}$，也就是一个节点的误差项$\delta$是网络误差对这个节点输入的偏导数的相反数。带入上式，得到：

$$
\delta_j=(t_j-y_j)y_j(1-y_j)\tag{4.17}
$$

上式就是**式4.1**。

将上述推导带入随机梯度下降公式，得到：

$$
\begin{align}
w_{ji}&\gets w_{ji}-\eta\frac{\partial{E_d}}{\partial{w_{ji}}}\\
&=w_{ji}+\eta(t_j-y_j)y_j(1-y_j)x_{ji}\\
&=w_{ji}+\eta\delta_jx_{ji}
\end{align}
$$

上式就是**式4.5**

### 隐藏层权值训练

现在我们要推导出隐藏层$\frac{\partial{E_d}}{\partial{net_j}}$的。

首先，我们需要定义节点$j$的所有直接下游节点的集合$Downstream(j)$。例如，对于节点4来说，它的直接下游节点是节点8、节点9。可以看到$net_j$只能通过影响$Downstream(j)$再影响$E_d$。设$net_k$是节点的下游节点的输入，则是的函数，而${net}_k$是${net}_j$的函数。因为${net}_k$有多个，我们应用全导数公式，可以做出如下推导：

$$
\begin{align}
\frac{\partial{E_d}}{\partial{net_j}}&=\sum_{k\in Downstream(j)}\frac{\partial{E_d}}{\partial{net_k}}\frac{\partial{net_k}}{\partial{net_j}}\\
&=\sum_{k\in Downstream(j)}-\delta_k\frac{\partial{net_k}}{\partial{net_j}}\\
&=\sum_{k\in Downstream(j)}-\delta_k\frac{\partial{net_k}}{\partial{a_j}}\frac{\partial{a_j}}{\partial{net_j}}\\
&=\sum_{k\in Downstream(j)}-\delta_kw_{kj}\frac{\partial{a_j}}{\partial{net_j}}\\
&=\sum_{k\in Downstream(j)}-\delta_kw_{kj}a_j(1-a_j)\\
&=-a_j(1-a_j)\sum_{k\in Downstream(j)}\delta_kw_{kj}
\end{align}
$$

因为$\delta_j=-\frac{\partial{E_d}}{\partial{net_j}}$，带入上式得到：

$$
\delta_j=a_j(1-a_j)\sum_{k\in Downstream(j)}\delta_kw_{kj}
$$

上式就是**式4.3**。

**——数学公式警报解除——**

至此，我们已经推导出了反向传播算法。需要注意的是，我们刚刚推导出的训练规则是根据激活函数是sigmoid函数、平方和误差、全连接网络、随机梯度下降优化算法。如果激活函数不同、误差计算方式不同、网络连接结构不同、优化算法不同，则具体的训练规则也会不一样。但是无论怎样，训练规则的推导方式都是一样的，应用链式求导法则进行推导即可。

转载自：[零基础入门深度学习(3) - 神经网络和反向传播算法](https://www.zybuluo.com/hanbingtao/note/476663)