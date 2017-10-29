import numpy as np
import math
#导入运算符模块
import operator
import matplotlib.pyplot as plt
import os

def createDataset():
    group=np.array([[1.0,1.1],[1.0,1.0],[0.0,0.0],[0,0.1]]);
    labels=['A','A','B','B']
    return (group,labels)
def classifyKNN0(inX,group,labels,k):
    (m,n)=np.shape(group)
    diff=np.tile(inX,(m,1))-group
    sqdiff=diff**2
    sqdist=np.sum(sqdiff,axis=1)#columns direction sum
    distances=sqdist**0.5
    sortedDisIndicies=distances.argsort()#对距离进行排序，从小到大返回索引
    classcountDict={}#dictionary字典
    for i in range(k):#0 - k-1
        ithDisIndex=sortedDisIndicies[i]
        ithLabel=labels[ithDisIndex]#第i个最近的训练样本对应的label
        classcountDict[ithLabel]=classcountDict.get(ithLabel,0)+1#default=0

    #排序，取出value值最大的key
    #sortedclassDict=sorted(classcountDict.items(),key=lambda item:item[1],reverse=True)
    sortedclassDict=sorted(classcountDict.items(),key=operator.itemgetter(1),reverse=True)
    return sortedclassDict[0][0]
def file2matrix(filename):
    fr=open(filename)
    lines=fr.readlines()
    fr.close()
    dataMatrix=np.zeros((np.shape(lines)[0],3))
    labelVector=[]
    index=0
    for line in lines:
        line=line.strip()#截取掉开头所有的字符
        listFromline=line.split('\t')
        dataMatrix[index,:]=listFromline[0:-1]
        labelVector.append(int(listFromline[-1]))
        index+=1
    return dataMatrix,labelVector
def readTxtData(filename,delimiterch):
    data=np.loadtxt(filename,delimiter=delimiterch)
    fearureMat=data[:,0:-1]
    labelVector=data[:,-1]
    return fearureMat,labelVector
def plotData(featureMat,labelVector):
    plt.figure(1)
    plt.xlabel('Percentage of time spending playing video game')
    plt.ylabel('Liters of ice cream consumed per week')
    plt.plot(featureMat[labelVector==1][:,1],featureMat[labelVector==1][:,2],'ro')
    plt.plot(featureMat[labelVector==2][:,1],featureMat[labelVector==2][:,2], 'y*')
    plt.plot(featureMat[labelVector==3][:,1],featureMat[labelVector==3][:,2], 'b^')
    plt.legend(('y=1','y=2','y=3'))
    plt.legend(('y=1', 'y=2', 'y=3'),loc='upper right')
    plt.show()
def scatterData(featureMat,labelVector):
    idx_1=np.where(labelVector==1)
    idx_2=np.where(labelVector==2)
    idx_3=np.where(labelVector==3)
    fig=plt.figure(1)
    ax1=fig.add_subplot(311)
    #plt.title("Scatter between feature 0 and feature 1")
    plt.xlabel('Num.of miles')
    plt.ylabel('Per.of time video game')
    plt.scatter(featureMat[idx_1,0],featureMat[idx_1,1],marker='o',color='r',label='y==1')
    plt.scatter(featureMat[idx_2, 0], featureMat[idx_2, 1], marker='*', color='y', label='y==2')
    plt.scatter(featureMat[idx_3, 0], featureMat[idx_3, 1], marker='^', color='b', label='y==3')
    plt.legend(loc='upper right')
    ax2 = fig.add_subplot(312)
    #plt.title("Scatter between feature 1 and feature 2")
    plt.xlabel('Per.of time on video game')
    plt.ylabel('Liters of ice cream ')
    plt.scatter(featureMat[idx_1, 1], featureMat[idx_1, 2], marker='o', color='r', label='y==1')
    plt.scatter(featureMat[idx_2, 1], featureMat[idx_2, 2], marker='*', color='y', label='y==2')
    plt.scatter(featureMat[idx_3, 1], featureMat[idx_3, 2], marker='^', color='b', label='y==3')
    plt.legend(loc='upper right')
    ax3 = fig.add_subplot(313)
    #plt.title("Scatter between feature 2 and feature 3")
    plt.xlabel('Num.of miles')
    plt.ylabel('Liters of ice cream')
    plt.scatter(featureMat[idx_1, 0], featureMat[idx_1, 2], marker='o', color='r', label='y==1')
    plt.scatter(featureMat[idx_2, 0], featureMat[idx_2, 2], marker='*', color='y', label='y==2')
    plt.scatter(featureMat[idx_3, 0], featureMat[idx_3, 2], marker='^', color='b', label='y==3')
    plt.legend(loc='upper right')
    plt.show()

def featureNormalize(dataSet):
    m=dataSet.shape[0]
    minVals=np.min(dataSet,axis=0)
    maxVals=np.max(dataSet,axis=0)
    ranges=maxVals-minVals
    normdataSet=np.zeros(np.shape(dataSet))
    normdataSet=dataSet-np.tile(minVals,(m,1))#行的方向扩展m倍，列的方向不扩展
    normdataSet=normdataSet/np.tile(ranges,(m,1))
    return normdataSet

def datingClassTest():
    testRatio=0.1#10 percent is used to be test set
    dataMat,labelVector=readTxtData('datingTestSet2.txt',delimiterch='\t')
    normMat=featureNormalize(dataMat)
    m=normMat.shape[0]
    numTestVecs=int(testRatio*m)

    #np.random.shuffle(normMat)#打散数据
    errorcount=0
    for i in range(numTestVecs):
        ithPreClass=classifyKNN0(normMat[i,:],normMat[numTestVecs:m],labelVector[numTestVecs:m],3)
        print("The classifier result came back with: {}, the real reuslt is: {}".format(ithPreClass,labelVector[i]))
        if(ithPreClass!=labelVector[i]):
            errorcount+=1
    errorcount=errorcount/numTestVecs
    print("The total error is: {}".format(errorcount))

def ClassifierErrorCount(TestSet,TestLabels,trainingSet,trainingLabels,k):
    errorcount=0
    numCrossValidation=TestSet.shape[0]
    for i in range(numCrossValidation):
        ithPreClass = classifyKNN0(TestSet[i], trainingSet, trainingLabels, k)
        if (ithPreClass !=TestLabels[i]):
            errorcount += 1
    return errorcount,errorcount/numCrossValidation

def crossValidation():
    dataMat, labelVector = readTxtData('datingTestSet2.txt', delimiterch='\t')
    normMat = featureNormalize(dataMat)
    m = normMat.shape[0]
    numCrossValidation=int(0.1*m)
    numTestSet = int(0.1 * m)
    CValidationSet=normMat[0:numCrossValidation]
    CValidationLabels=labelVector[0:numCrossValidation]
    trainingSet=normMat[numCrossValidation+numTestSet:m]
    trainingLabels=labelVector[numCrossValidation+numTestSet:m]
    testSet=normMat[numCrossValidation:numCrossValidation+numTestSet]
    testLabels=labelVector[numCrossValidation:numCrossValidation+numTestSet]
    kList=[ i for i in range(1,15)]
    errList=[]
    for k in kList:
        errcount,errorrate=ClassifierErrorCount(CValidationSet,CValidationLabels,trainingSet,trainingLabels,k)
        print('When k is equal to {}, The total number of errors is {}, total error rate is {}'.format(k,errcount,errorrate))
        errList.append(errorrate)

    plt.figure()
    plt.title("The error---k")
    plt.xlabel('k')
    plt.ylabel('error')
    plt.plot(kList,errList,lineWidth=2)
    plt.show()

    minErr=min(errList)
    k=errList.index(minErr)+1
    print(k)
    testerr,testerrrate=ClassifierErrorCount(testSet,testLabels,trainingSet,trainingLabels,k)
    print('The total error in testset is:{}, total error rate in testset is:{}'.format(testerr,testerrrate))
'''
手写字符识别
'''
def imgVector(filename):
    returnVector=np.zeros((1,1024))#32*32
    fr = open(filename)
    for i in range(32):
        strLine=fr.readline()#readline函数，每个元素
        for j in range(32):
            returnVector[0,32*i+j]=int(strLine[j])
    return returnVector

def handWrittingClassTest():
    trainingFileList=os.listdir('digits/trainingDigits')
    numtrainingSet=len(trainingFileList)
    trainingDataMat=np.zeros((numtrainingSet,1024))#1024 features
    trainingLabels=[]
    for i in range(numtrainingSet):
        fileNameStr=trainingFileList[i]
        trainingDataMat[i, :] = imgVector('digits/trainingDigits/%s'%fileNameStr)
        fileStr=fileNameStr.split('.')[0]
        #fileStr=fileStr.split('_')
        classNumber=int(fileStr.split('_')[0])#转化为整数
        trainingLabels.append(classNumber)

    testFileList=os.listdir('digits/testDigits')
    numtestSet=len(testFileList)
    testDataMat=np.zeros((numtestSet,1024))
    testLabels=[]#不需要
    errorCount=0
    for i in range(numtestSet):
        fileNameStr=testFileList[i]
        testDataMat[i,:]=imgVector('digits/testDigits/%s'%fileNameStr)
        fileStr=fileNameStr.split('.')[0]
        classNumber=int(fileStr.split('_')[0])
        predictClass=classifyKNN0(testDataMat[i,:],trainingDataMat,trainingLabels,3)
        print('预测的数字是：{}，实际的数字是：{}'.format(predictClass,classNumber))
        if(predictClass!=classNumber):
            errorCount+=1
    print('The total number of errors is: {}'.format(errorCount))
    print('The total error rate is: {}'.format(errorCount/numtestSet))
def main():
    #handWrittingClassTest()
    crossValidation()
    #datingClassTest()

if __name__ == '__main__':
    main()

