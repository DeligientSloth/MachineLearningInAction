import numpy as np
import matplotlib.pyplot as plt
import random
def loadTxtData(filename):
    data=np.loadtxt(filename,delimiter='\t')
    m,n=data.shape
    print('该数据集有{}个特征，{}条记录'.format(n-1,m))
    X=data[:,0:n-1]#通过tuple作为嵌套list的下标
    y=data[:,n-1]
    X=np.c_[np.ones((m,1)),X]
    return X,y

def sigmoid(inX):
    return 1.0/(1+np.exp(-inX))
def CostFunction(htheta,yMatrix):
    return -np.sum(np.multiply(yMatrix,np.log(htheta))+np.multiply(1-yMatrix,np.log(1-htheta)))

def gradAscent(X,y):
    XMatrix=np.mat(X)
    yMatrix=np.mat(y).transpose()
    m,n=np.shape(XMatrix)
    theta=np.ones((n,1))
    alpha=0.001
    MaxCycle=500
    J_history=[]
    theta_history=[]
    for k in range(MaxCycle):
        htheta=sigmoid(XMatrix*theta)
        error=htheta-yMatrix
        theta-=alpha*XMatrix.transpose()*error#梯度上升
        J_history.append(CostFunction(htheta,yMatrix))
        theta_history.append(theta)
    return theta,J_history
def stocGradientAscent(X,y):
    m,n=np.shape(X)
    alpha=0.01
    theta=np.ones(n)
    J_history=[]
    for k in range(m):
        htheta=sigmoid(np.sum(theta*X[k]))
        error=htheta-y[k]#此时error是一个实数
        theta-=alpha*error*X[k]
        J=-y[k]*np.log(htheta)-(1-y[k])*np.log(1-htheta)
        J_history.append(J)
    return theta,J_history

def stocGradAscent1(dataMatrix, classLabels, numIter=150):
    m,n = np.shape(dataMatrix)
    weights = np.ones(n)   #initialize to all ones
    for j in range(numIter):
        dataIndex = list(range(m))
        for i in range(m):
            alpha = 4/(1.0+j+i)+0.0001    #apha decreases with iteration, does not
            randIndex = int(random.uniform(0,len(dataIndex)))#go to 0 because of the constant
            h = sigmoid(sum(dataMatrix[randIndex]*weights))
            error = classLabels[randIndex] - h
            weights = weights + alpha*error*dataMatrix[randIndex]
            del(dataIndex[randIndex])
    return weights
def stocGradientAscentImprove(X,y,numIter=150):
    m,n=np.shape(X)
    theta=np.ones(n)
    for j in range(numIter):
        dataIndex=list(range(m))#数据的下标
        for i in range(m):
            alpha=4/(1.0+i+j)+0.0001
            randIndex=int(random.uniform(0,len(dataIndex)))
            htheta=sigmoid(sum(X[randIndex]*theta))
            error=htheta-y[randIndex]
            theta-=alpha*error*X[randIndex]
            del dataIndex[randIndex]
    return theta
def monitorCostFunction(J_history):
    plt.figure('The curve of CostFunction')
    plt.plot(J_history, color='r', label='The curve of CostFunction')
    plt.ylabel('Cost Function')
    plt.xlabel('No.of iterations')
    plt.legend()
    plt.show()
def plotBestFit(X,y,theta):
    index1=np.where(y==1)
    index0=np.where(y==0)
    plt.figure('Classifier Boundary')
    plt.ylabel('x2')
    plt.xlabel('x1')
    plt.scatter(X[index1][:,1],X[index1][:,2],marker='^', color='r', label='y==1')
    plt.scatter(X[index0][:, 1], X[index0][:, 2], marker='*', color='y', label='y==0')
    x1=np.arange(min(X[:,1]),max(X[:,1]),0.01)
    x2=(-theta[0]-theta[1]*x1)/theta[2]
    plt.plot(x1,x2,linewidth=2,label='Boundary Line')
    plt.legend(loc='upper right')

def classifyVector(inX,theta):
    prob=sigmoid(np.sum(inX*theta,axis=1))
    labels=np.where(prob>=0.5,1,0)
    return labels

def horseColicClassification(X,y,X_test,y_test):

    theta=stocGradientAscentImprove(X,y)

    #plotBestFit(X,y,theta)
    labels=classifyVector(X_test,theta)
    errorRate=np.mean(labels!=y_test)
    return errorRate
def multiTest():
    X, y = loadTxtData('horseColicTraining.txt')
    X_test, y_test = loadTxtData('horseColicTest.txt')
    numIter=10
    errSum=0.0
    for k in range(numIter):
        errRate=horseColicClassification(X,y,X_test,y_test)
        print("本次的错误率为：{}".format(errRate))
        errSum+=errRate
    print("总的错误率为：{}".format(errSum/numIter))

if __name__ == '__main__':
    multiTest()


