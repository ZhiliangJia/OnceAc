---
layout: post
title: 吴恩达 机器学习 单变量线性回归
date: 2021-03-31
author: Zhiliang 
tags: [Machine Learning,Andrew Ng,Basic]
toc: true
mathjax: true
---

**机器学习**是[人工智能](https://zh.wikipedia.org/wiki/人工智能)的一个分支。人工智能的研究历史有着一条从以“[推理](https://zh.wikipedia.org/wiki/推理)”为重点，到以“[知识](https://zh.wikipedia.org/wiki/知识)”为重点，再到以“[学习](https://zh.wikipedia.org/wiki/学习)”为重点的自然、清晰的脉络。显然，机器学习是实现人工智能的一个途径，即以机器学习为手段解决人工智能中的问题。机器学习在近30多年已发展为一门多领域[交叉学科](https://zh.wikipedia.org/wiki/交叉学科)，涉及[概率论](https://zh.wikipedia.org/wiki/概率论)、[统计学](https://zh.wikipedia.org/wiki/统计学)、[逼近论](https://zh.wikipedia.org/wiki/逼近论)、[凸分析](https://zh.wikipedia.org/w/index.php?title=凸分析&action=edit&redlink=1)、[计算复杂性理论](https://zh.wikipedia.org/wiki/计算复杂性理论)等多门学科。



<!-- more -->



# 矩阵和向量

如图：这个是4×2矩阵，即4行2列，如果$m$为行，$n$为列，那么$m×n$即4×2

![img](https://gitee.com/zhiliangj/Typora_Img/raw/master/9fa04927c2bd15780f92a7fafb539179.png)

矩阵的维数即行数×列数

矩阵元素（矩阵项）：$A=\left[ \begin{matrix}
	1402&		191\\
	1371&		821\\
	949&		1437\\
	147&		1448\\
\end{matrix} \right]$

$A_{ij}$指第$i$行，第$j$列的元素。

向量是一种特殊的矩阵，讲义中的向量一般都是列向量，如： $y=\left[ \begin{array}{c}
	460\\
	232\\
	315\\
	178\\
\end{array} \right] $为四维列向量（4×1）。

# 加法和标量乘法

矩阵的加法：行列数相等的可以加。例：

![img](https://gitee.com/zhiliangj/Typora_Img/raw/master/ffddfddfdfd.png)

矩阵的标量乘法：每个元素都要乘。例：

![img](https://gitee.com/zhiliangj/Typora_Img/raw/master/fdddddd.png)

组合算法也类似。

# 矩阵向量乘法

矩阵乘法：$m×n$矩阵乘以$n×o$矩阵，变成$m×o$矩阵。

如果这样说不好理解的话就举一个例子来说明一下，比如说现在有两个矩阵$A$和$B$，那么它们的乘积就可以表示为图中所示的形式。

![img](https://gitee.com/zhiliangj/Typora_Img/raw/master/1a9f98df1560724713f6580de27a0bde.jpg)

![img](https://gitee.com/zhiliangj/Typora_Img/raw/master/5ec35206e8ae22668d4b4a3c3ea7b292.jpg)

# 矩阵乘法的性质

矩阵乘法的性质：

- 矩阵的乘法不满足交换律：$A×B≠B×A$
- 矩阵的乘法满足结合律：$A×(B×C)=(A×B)×C$
- 单位矩阵：在矩阵的乘法中，有一种矩阵起着特殊的作用，如同数的乘法中的1,我们称这种矩阵为单位矩阵。它是个方阵，一般用$I$或者$E$表示，本讲义都用$I$代表单位矩阵，从左上角到右下角的对角线（称为主对角线）上的元素均为1以外全都为0。如：$AA^{-1}=A^{-1}A=I$。对于单位矩阵，有$AI=IA=A$

# 逆、转置

矩阵的逆：如矩阵$A$是一个$m×m$矩阵（方阵），如果有逆矩阵，则：$AA^{-1}=A^{-1}A=I$

我们一般在**OCTAVE**或者**MATLAB**中进行计算矩阵的逆矩阵。

矩阵的转置：设$A$为$m×n$阶矩阵（即$m$行$n$列），第$i$行$j$列的元素是$a(i,j)$，即：$A=a(i,j)$

定义$A$的转置为这样一个$n×m$阶矩阵$B$，满足$B=a(j,i)$，即（B的第$i$行第$j$列元素是$A$的第$j$行第$i$列元素），记$A^T=B$。(有些书记为$A'=B$）

直观来看，将$A$的所有元素绕着一条从第1行第1列元素出发的右下方45度的射线作镜面反转，即得到$A$的转置。

例：$\left| \begin{matrix}
	a&		b\\
	c&		d\\
	e&		f\\
\end{matrix} \right|^T=\left| \begin{matrix}
	a&		c&		e\\
	b&		d&		f\\
\end{matrix} \right|$

矩阵的转置基本性质：
$$
\left( A\pm B \right) ^T=A^T\pm B^T
\\
\left( A\times B \right) ^T=B^T\times A^T
\\
\left( A^T \right) ^T=A
\\
\left( KA \right) ^T=KA^T
$$
