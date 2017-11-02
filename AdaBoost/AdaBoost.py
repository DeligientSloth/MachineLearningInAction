from numpy import *
def loadSimpData():
    datMat=matrix([[1.,2.1],
        [2.,1.1],
        [1.3,1.],
        [1.,1.],
        [2.,1.]])
    classLabels=[1.0,1.0,-1.0,-1.0,1.0]
    return datMat,classLabels
'''
将最小错误率minError设为infinity
    对数据集中的每一个特征（第一层循环）：
      对每个步长（第二层循环）：
        对每个不等号（第三层循环）：
           建立一棵单层决策树并利用加权数据集对它进行测试
           如果错误率低于m in Err0r，则将当前单层决策树设为最佳单层决策树
    返回最佳单雇决策树
'''
def stumpClassify(dataMatrix,dimen,thresVal,threshIneq):#树桩分类
    retArray=ones((shape(dataMatrix)[0],1))#所有都被定义为正例
    if threshIneq=='lt':
        retArray[dataMatrix[:,dimen]<=thresVal]=-1.0
    else:
        retArray[dataMatrix[:,dimen]>thresVal]=-1.0
    return retArray
def buildStump(dataArr,classLabels,D):
    dataMatrix=mat(dataArr)
    m,n=shape(dataMatrix)
    labelMat=mat(classLabels).transpose()
    numSteps=10.0
    bestStump={}#字典
    bestClasEst=mat(zeros((m,1)))
    minError=inf#numpy有一个inf，math里面也有
    for i in range(n):
        rangeMin=dataMatrix[:,i].min()
        rangeMax=dataMatrix[:,i].max()
        stepSize=(rangeMax-rangeMin)/numSteps
        for j in range(-1,int(numSteps)+1):#第二层循环，为什么要加-1？
            for inequal in ['lt','gt']:#第三层循环，对每个不等号
                thresVal=rangeMin+j*stepSize
                #print('j={},thresVal={},rangemin={},rangemax={}:'.format(j,thresVal,rangeMin,rangeMax))
                predictedVals=stumpClassify(dataMatrix,i,thresVal,inequal)
                errArr=mat(ones((m,1)))
                errArr[predictedVals==labelMat]=0#计算误差率
                weightedError=D.T*errArr#计算加权误差，上一轮错误分类对目前加权影响较大
                #print("split: dim %d, thresh %.2f, thresh ineqal: %s, the weighted error is %.3f" % \(i, thresVal, inequal, weightedError))
                if(weightedError<minError):
                    minError=weightedError
                    bestClasEst=predictedVals.copy()#记录预测最佳分类
                    bestStump['dim']=i
                    bestStump['thresh']=thresVal
                    bestStump['ineq']=inequal
    return bestStump,minError,bestClasEst
def test():
    dataMat,classLabels=loadSimpData()
    D=ones((5,1))/5
    bestStump, minError, bestClasEst=buildStump(dataMat,classLabels,D)
    print('===========================================================')
    print(bestStump)
    print('===========================================================')
    print(minError)
    print('===========================================================')
    print(bestClasEst)
'''
本质是串行训练，先训练一颗决策树，调整权值，训练下面一颗
对每次迭代：
利用buildstump函数找到最佳的单层决策树
将最佳单层决策树加入到单层决策树数组
计算alpha
计算新的权重向量D
更新累计类别估计值
如果错误率等于0.0,则退出循环
'''
def adaBoostTrainDS(dataArr,classLabels,numIt=40):
    weakClassArr=[]
    m=shape(dataArr)[0]
    D=ones((m,1))/m
    aggClassEst=mat(zeros((m,1)))
    for i in range(numIt):
        bestStump,error,classEst=buildStump(dataArr,classLabels,D)
        #alpha要转化为float，不然后面运算会出错
        alpha=float(0.5*log((1-error)/max(error,1e-16)))#避免error太小或者为0的情况的出现，alpha为组合系数，代表在最终分类中的话语权
        bestStump['alpha']=alpha
        weakClassArr.append(bestStump)#加入决策数组
        #update D for the next iteration: 更新D，先计算准确率
        expon=multiply(-1*alpha*classEst,mat(classLabels).T)
        D=multiply(D,exp(expon))
        D=D/sum(D)#归一化
        #print('D=: ',D.T)
        #计算组合预测，为各个基本分类器的组合，calc training error of all classifiers,如果为0就quit for loop
        aggClassEst+=alpha*classEst
        #print('aggClassEst=:',aggClassEst.T)
        #若aggclassest>0,预测为正例，否则为负例
        aggErrors=multiply(sign(aggClassEst)!=mat(classLabels).T,ones((m,1)))#sign函数返回1，-1
        errRate=mean(aggErrors)
        #print('The total error rate is: ',errRate)
        if errRate==0.0: break
    return weakClassArr,errRate
'''
编写分类模块，分类模块调用决策树组每一个分类器对数据进行分类最后加权
'''
def adaClassify(dataMat,classifierArr):
    dataMatrix=mat(dataMat)
    m=shape(dataMatrix)[0]
    aggClassEst=mat(zeros((m,1)))
    for classItem in classifierArr:
        classEst=stumpClassify(dataMatrix,classItem['dim'],classItem['thresh'],classItem['ineq'])
        aggClassEst+=classItem['alpha']*classEst
        #print('aggClassEst=:',aggClassEst)#分类器分类结果逐渐增强
    return sign(aggClassEst)

if __name__ == '__main__':
    #test()
    dataMat, classLabels = loadSimpData()
    weakArr=adaBoostTrainDS(dataMat,classLabels,9)
    print(weakArr)
    print('=================================================================')
    predictedClass=adaClassify([1,2.1],weakArr)
    print(predictedClass)
    '''
    [1,2.1]的错误被调整过来了
    aggClassEst=: [[-0.69314718]]
    aggClassEst=: [[ 0.27980789]]
    aggClassEst=: [[ 1.17568763]]
    [[ 1.]]
    '''