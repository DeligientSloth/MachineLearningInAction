# Support Vector Machine概述

## Basic Concept: 

引用Wiki百科的描述

> 在[机器学习](https://zh.wikipedia.org/wiki/%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0)中，**支持向量机**（英语：**support vector machine**，常简称为**SVM**，又名**支持向量网络**[[1\]](https://zh.wikipedia.org/wiki/%E6%94%AF%E6%8C%81%E5%90%91%E9%87%8F%E6%9C%BA#cite_note-CorinnaCortes-1)）是在[分类](https://zh.wikipedia.org/wiki/%E5%88%86%E7%B1%BB%E9%97%AE%E9%A2%98)与[回归分析](https://zh.wikipedia.org/wiki/%E8%BF%B4%E6%AD%B8%E5%88%86%E6%9E%90)中分析数据的[监督式学习](https://zh.wikipedia.org/wiki/%E7%9B%A3%E7%9D%A3%E5%BC%8F%E5%AD%B8%E7%BF%92)模型与相关的学习[算法](https://zh.wikipedia.org/wiki/%E7%AE%97%E6%B3%95)。给定一组训练实例，每个训练实例被标记为属于两个类别中的一个或另一个，SVM训练算法创建一个将新的实例分配给两个类别之一的模型，使其成为非概率[二元](https://zh.wikipedia.org/w/index.php?title=%E4%BA%8C%E5%85%83%E5%88%86%E7%B1%BB%E5%99%A8&action=edit&redlink=1)[线性分类器](https://zh.wikipedia.org/wiki/%E7%BA%BF%E6%80%A7%E5%88%86%E7%B1%BB%E5%99%A8)。SVM模型是将实例表示为空间中的点，这样映射就使得单独类别的实例被尽可能宽的明显的间隔分开。然后，将新的实例映射到同一空间，并基于它们落在间隔的哪一侧来预测所属类别。

​       支持向量机（support vector machine，SVM）是一种二分类的模型，要实现多分类还需要结合其他的方法，一般的书籍都将他描述为：**特征空间上间隔最大的线性分类器** 。支持向量机还有另外一种解释，通过引入合页Loss Function，对错分的样本使用hinge loss计算penalty，从而将svm问题转化为最小化cost function的问题，并引入regularization正则项，这样就与logistic regression统一起来了，但与logistics regression相比，用的cost function形式不同，在后面的笔记里面，我会推导出SVM这两种模型，并与logistic regression做比较。

​       对于给定的训练样本 $D=\lbrace{(x_1,y_1),(x_2,y_2)\cdots,(x_m,y_m)}\rbrace$ ,$y_i\in\lbrace-1,1\rbrace$ ,基本思想是在样本特征空间中找到一个划分超平面，将不同类别的样本分开。划分平面有很多个，我们应该找哪一个呢？下面给出一张机器学习实战的图，可以观察一下：

![img](https://github.com/DeligientSloth/MachineLearningInAction/blob/master/Images/%E6%94%AF%E6%8C%81%E5%90%91%E9%87%8F%E5%88%92%E5%88%86%E5%9B%BE.png)

​       这是一个二维图，我们先从二维的情况入手，不难看出，四张图的分类超平面都成功将数据集划分开，也就是说，这四个分类模型在training example上的表现都是perfect的，但是哪个分类模型更好呢？

​       在这四张图中，图D的分割平面与两类样本点都隔得比较开，可以说出于两类训练样本正中间的划分超平面，图B次之，图C与两类样本点靠的比较近。在实际分类过程中，训练集和测试集的样本由于局限性或者噪声，可能会接近划分平面，而出现错误。如果划分平面与两类样本都间隔的比较开，这样的划分平面对于训练集外的不那么容易出现错分现象，对训练样本外的样本的局部扰动容忍性最好。因此，图D的划分平面对未知样本的generation能力最强。

​      不难看出，SVM在保证数据点正确划分的情况下，实现了**结构风险最小化（structural risk minimization）** ，保证了模型的泛华能力，使分隔平面与两类数据点间隔开可以理解为模型的**regularization term正则项** 

​      在上面给出的例子中，由于数据点都在二维平面上，所以此时分隔超平面就只是一条直线。但是，如果所给的数据集是三维的，那么此时用来分隔数据的就是一个平面。显而易见，更高维的情况可以依此类推。如果数据集是1024维的，那么就需要一个1023维的某某对象来对数据进行分隔。这个1023维的某某对象到底应该叫什么？ $N-1$ 维呢？该对象被称为超平面，也就是分类的决策边界。分布在超平面一侧的所有数据都属于某个类别，而分布在另一侧的所有数据则属于另一个类别。

## 间隔最大化Margin Maximum Model##

​    支持向量机学习的基本思想是：**求解能够找到能正确划分数据集并且使得几何间隔最大的分离超平面，分离超平面可以有无穷多个**， 但是间隔最大的分离超平面只有一个，这里讨论的是硬间隔最大化，当数据近似线形可分，还会引入软间隔最大化，这个会在后面讨论。

​      间隔最大化的一个intuition是：**对充分大的确信度confidence将训练数据进行分类，不仅将正负实例进行分开，而且对分类要求最苛刻的数据点（离超平面最近的数据点）以足够大的确信度将他们分开，这里的确信度就是用训练集的几何间隔来表示，几何间隔越大，确信度越高。

下面给出间隔最大化的完整模型，引入了**函数间隔** 以及**几何间隔** 的定义：

![SVM1](https://github.com/DeligientSloth/MachineLearningInAction/blob/master/Images/SVM%E5%8E%9F%E7%90%861.jpg)

![SVM2](https://github.com/DeligientSloth/MachineLearningInAction/blob/master/Images/SVM%E5%8E%9F%E7%90%865.jpg)

