from AdaBoost import *
def loadDataSet(filename):
    fr=open(filename)
    num=len(fr.readline().split('\t'))
    dataMat=[]
    labelMat=[]
    fr.seek(0,0)
    for line in fr.readlines():
        lineArr=[]
        Stringline=line.strip().split('\t')
        for i in range(num-1):
            lineArr.append(float(Stringline[i]))
        dataMat.append(lineArr)
        labelMat.append(float(Stringline[-1]))
    fr.close()
    return dataMat,labelMat

if __name__ == '__main__':
    classifierNum=[1,10,50,100,500,1000,10000]
    traindataMat, trainlabelMat = loadDataSet('horseColicTraining2.txt')
    testDataMat, testLabelMat = loadDataSet('horseColicTest2.txt')
    for num in classifierNum:
        weakArr,trainErrRate=adaBoostTrainDS(traindataMat,trainlabelMat,num)
        predictedClass=adaClassify(testDataMat,weakArr)
        testErrRate=mean(predictedClass!=mat(testLabelMat).T)
        print('training error: {},test error :{}'.format(trainErrRate,testErrRate))