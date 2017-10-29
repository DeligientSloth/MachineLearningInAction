'''
海洋生物是否为鱼类的demo，
有两个feature，
一个是no surfacing(surface有浮出水面的意思):不冒出水面是否可以生存；
一个是flippers:是否有脚蹼
决策：是否是鱼类
'''
from math import log
import operator
from StatisticsDecisionTree import *
import plotDecisionTree

def createDataSet():
    dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    labels = ['no surfacing','flippers']
    #change to discrete values
    return dataSet, labels
'''
定义计算shannon entropy的函数
Entropy的值越小，纯度越高
所以后面改变类别个数，entropy变大
'''
def calcShannonEnt(dataSet):
    numEntries=len(dataSet)
    labelCounts={}#类别字典，对类别进行计数,key为label，value为出现次数
    for featVec in dataSet:
        currentLabel=featVec[-1]#feature向量读取dataset的每一行，最后一个是所属类别
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel]=0#若不存在，初始化为0
        labelCounts[currentLabel]+=1
    shannonEnt=0.0
    '''for key in labelCounts.keys():
        print(key) '''#返回yes,no
    '''for key in labelCounts.items():
        print(key) '''#返回('yes', 2)，('no', 3)
    for key in labelCounts:
        prob=float(labelCounts[key])/numEntries#计算每一个类别出现的概率
        shannonEnt-=prob*log(prob,2)#以2为底，是为了要中点概率最大吗？
    return shannonEnt#0.9709505944546686没有float类型转换
        #print(key)#返回yes,no,这样遍历字典只会返回key值而不会返回value
'''
根据给定的特征划分数据集，返回的是划分数据子集，需要输入两个参数
@axis:通过哪个feature进行划分
@value:通过feature根据自身离散的取值对特征进行划分，得到多个子集，不同子集对应的feature取值不同
返回的数据子集中无需包含划分的feature的值
'''
#这里函数声明一个返回列表对象，因为python在函数中传递的是列表引用，为了避免dataSet受到影响，所以在声明的列表里面修改
def splitDataSet(dataSet,axis,value):
    retDataSet=[]#data set本来就是一个列表，列表每个元素也是列表，是不同feature取值的集合，以此来描述一个数据点
    for featVec in dataSet:#需要遍历dataset，取出每个数据点，每个数据点根据划分feature的不同取值进入不同的子集合中
        if featVec[axis]==value:#根据不同取值进入不同集合，本函数是返回特定取值的子集合
            reducedFeatVec=featVec[0:axis]#左半部分表示，0-axis-1
            reducedFeatVec.extend(featVec[axis+1:])#axis+1-end,axis数据点被去掉了
            retDataSet.append(reducedFeatVec)#append和extend不同，append是直接将元素插入，extend将列表元素展开插入
    return retDataSet

'''
选择划分最优属性，接受数据集输入
对于每一个特征，尝试对数据集进行划分，求出信息增益最大的属性
'''
def chooseBestFeatureToSplit(dataSet):
    numFeatures=len(dataSet[0])-1#最后一行是label类别的表示
    baseEntropy=calcShannonEnt(dataSet)#未划分前的信息增益
    bestInfoGain=0.0
    bestFeature=-1
    for i in range(numFeatures):#对第i个特征feature，用第i个特征对数据集进行划分
        featList=[example[i] for example in dataSet]#通过列表推导式遍历数据集的样本，获取第i个特征组成列表
        uniqueVals=set(featList)#上面list的目的就是获取每个feature的不同取值，根据不同取值对数据集进行划分
        newEntropy=0.0#初始化划分之后的信息熵
        #下面计算用特征i划分数据之后的信息熵
        for value in uniqueVals:#遍历划分的特征取值，即求出每个划分子集的信息增益总和表示划分后的信息熵
            subDataSet=splitDataSet(dataSet,i,value)
            prob=len(subDataSet)/float(len(dataSet))#每个子集被赋予的权重，即子集样本占总体的比例，比例越大影响越大
            newEntropy+=prob*calcShannonEnt(subDataSet)
        #选择最大信息增益的属性
        infoGain=baseEntropy-newEntropy
        if(infoGain>bestInfoGain):
            bestInfoGain=infoGain
            bestFeature=i
    return bestFeature

'''
编写返回最大类别的函数，当数据集中只有一个类别的时候，以数据集中该类别取值最大的label
'''
def majorityCnt(classList):
    classCount={}#投票法
    for vote in classList:#与计算香农熵的时相同，不过不计算概率
        classCount[vote]=classCount.get(vote,0)+1#实现自增
    #字典排序，用sorted实现，先用dict.items将字典转化为可以迭代的对象,通过operator运算符模块提取第一个分量
    #这里的key是一个函数，提取第一个分量的意思
    '''
    >>>
    >>> dic={'abc':3,'def':2}
    >>> dic
        {'abc': 3, 'def': 2}
    >>> sorted(dic.items(),key=lambda x:x[1],reverse=True)
        [('abc', 3), ('def', 2)]
    >>> sorted(dic,key=lambda x:x[1],reverse=True)
        ['def', 'abc']
        似乎不通过items()将dict变为可迭代对象也可以？
        dict是可以迭代的对象吗？
    >>> import collections
    >>> isinstance(dict,collections.Iterator)
        False
        这里，dict应该可以通过代码解释为迭代对象，返回key值
    '''
    sortedClassCount=sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)#返回的是列表
    return sortedClassCount[0][0]#返回的是类别


def createTree(dataSet,labels):
    labels_copy=labels[:]#这里有问题要仔细斟酌，根据源代码，labels是要不断减小的！因此这里我对labels做了一次拷贝
    classList = [example[-1] for example in dataSet]#最后一列是所属类别，是鱼还是不是鱼
    #编写递归返回条件
    #1.样本所属的类别相同
    if(classList.count(classList[0])==len(classList)):#此时类别集合只有一种类别，此种类别数量等于总数
        return classList[0]#返回此种类别
    #2.特征以及全部划分完，没有剩余的特征，此时dataset只剩下类别
    if(len(dataSet[0])==1):
        return majorityCnt(classList)#以数据中类别最多的类别作为返回类别

    #下面是可以继续选择最优特征划分的情况
    bestFeat=chooseBestFeatureToSplit(dataSet)#选择最佳划分特征
    bestFeatLabel=labels_copy[bestFeat]#最佳划分特征对于的类别值，此类别值不同将数据划分到不同的子集合去
    #decisiontree的数据结构
    decisionTree={bestFeatLabel:{}}#嵌套字典，key是类别，value是字典，子字典的key对于父字典key（类别）的不同取值，value是划分后子集的类别
    del labels_copy[bestFeat]#可以删除了，子集合也没有对应的特征值，那么对应的特征肯定要删除，子集合只有还没划分的特征
    featValues=[example[bestFeat] for example in dataSet]#获知此属性有几个取值
    uniqueVals=set(featValues)
    for value in uniqueVals:
        subLabels=labels_copy[:]#对labels做一次拷贝，浅拷贝和深拷贝的区别?后面labels到函数内部，避免发生改变
        subDataSet=splitDataSet(dataSet,bestFeat,value)#获取划分的数据子集合
        decisionTree[bestFeatLabel][value]=createTree(subDataSet,subLabels)#递归生成子树
    return decisionTree

'''
实现对决策树的存储和读取
'''
import pickle
def storeTree(decisionTree,filename):
    fw=open(filename,'wb')#必须以wb打开，write bytes
    pickle.dump(decisionTree,fw)
    fw.close()
def getTree(filename):
    fr=open(filename,'rb')
    decisionTree=pickle.load(fr)
    fr.close()
    return decisionTree
#对输入特征向量以及对应的特征标签列表进行分类
def classify(inputTree,featLabels,testVec):
    firstStr=list(inputTree.keys())[0]#取出最佳划分特征
    featIndex=featLabels.index(firstStr)
    featValue=testVec[featIndex]#得到输入数据在当前最佳划分属性的值，决定进入哪个数据集
    secondDict=inputTree[firstStr]#获取最优划分属性的子树
    subTree=secondDict[featValue]
    if isinstance(subTree,dict):
        classLabel=classify(subTree,featLabels,testVec)
    else:
        classLabel=subTree
    return classLabel
def test():
    dataSet, labels = createDataSet()
    # shannonEnt=calcShannonEnt(dataSet)
    # print(shannonEnt)
    decisionTree = decisionTree = createTree(dataSet, labels)
    storeTree(decisionTree, 'decisionTree.txt')
    storedecisionTree = getTree('decisionTree.txt')
    print(storedecisionTree)
def ContactLensesClassifier():
    fr=open('lenses.txt')
    lines=fr.readlines()
    dataSet=[ l.strip().split('\t') for l in lines ]
    labels=['age','prescript','astigmatic','tearRate']
    decisionTree=createTree(dataSet,labels)
    print(labels)
    plotDecisionTree.createPlot(decisionTree)
    storeTree(decisionTree,'lensesDecisionTree.txt')
    tree=getTree('lensesDecisionTree.txt')
    plotDecisionTree.createPlot(tree)
if __name__ == '__main__':
    ContactLensesClassifier()