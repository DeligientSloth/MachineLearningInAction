from numpy import *
import matplotlib.pyplot as plt

def loadDataSet(filename):
    fr=open(filename)
    numFeat=len(fr.readline().split('\t'))-1
    fr.seek(0,0)
    dataMat=[];labelMat=[]
    for line in fr.readlines():
        curline=line.strip().split('\t')
        lineArr=[]
        for i in range(numFeat):
            lineArr.append(float(curline[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curline[-1]))#转化为float型
    return dataMat,labelMat

def standRegress(X,y):
    XMat=mat(X)
    yMat=mat(y).T
    xTx=XMat.T*XMat
    if linalg.det(xTx)==0.0: print('x matrix is singular,cannot do inverse!');return
    ws=linalg.inv(xTx)*(XMat.T*yMat)#也可以用x.I求逆
    return ws
def plotRegression(dataMat,labelMat,ws):
    xMat=mat(dataMat)
    yMat=mat(labelMat).T
    fig=plt.figure()
    ax=fig.add_subplot(111)
    ax.scatter(xMat[:,1].flatten().A[0], yMat[:, 0].flatten().A[0])  # 扁平化矩阵的方法
    xCopy=xMat.copy()
    xCopy.sort(axis=0)
    yHat=xCopy*ws
    ax.plot(xCopy[:, 1], yHat)
    plt.show()


#k越小，曲线越尖锐
def lwlr(point,X,y,k=1.0):
    xMat=mat(X);yMat=mat(y).T
    m=shape(xMat)[0]
    weights=mat(eye((m)))
    for i in range(m):
        diffMat=point-xMat[i,:]
        weights[i,i]=float(exp(-diffMat*diffMat.T/(2.0*k**2)))
    xTx=xMat.T*weights*xMat
    if linalg.det(xTx)==0:
        print('This matrix is singular,can not inverse')
        return
    ws=xTx.I*xMat.T*weights*yMat#训练得到ws
    return float(point*ws)

def lwlrTest(Xtest,X,y,k=1.0):
    m=shape(Xtest)[0]#
    yHat=zeros((m,1))
    for i in range(m):
        yHat[i]=lwlr(Xtest[i],X,y,k)#用第i个数据点训练样本，得到第i个样本预测值
    return yHat
def plotlwlrTest(X,y,k=1.0):
    m=shape(X)[0]
    Xcopy=copy(X)
    Xcopy.sort(axis=0)
    yHat=zeros((m, 1))
    for i in range(m):
        yHat[i]=lwlr(Xcopy[i],X,y,k)
    return Xcopy,yHat


def plotlwlr(dataMat,labelMat,k):
    xMat = mat(dataMat)
    strInd = xMat[:, 1].argsort(axis=0)  # x[strInd]会得到一个三维数组
    sortIndex = strInd.flatten().A[0]  # matrix转化为一维数组

    #对x,y排序
    xsort = xMat[sortIndex]
    '''k=1.0
    yHat=lwlrTest(X,X,y,k)
    yHatsort=yHat[sortIndex]'''
       # 画图
    fig=plt.figure()
    for i in range(len(k)):
        yHat = lwlrTest(X, X, y, k[i])
        yHatsort = yHat[sortIndex]
        ax=fig.add_subplot(311+i)
    # print(yHat)
    # yHatsort=yHat[strInd][:,0,:]
        plt.title('k={}'.format(k[i]))
        plt.xlabel('x')
        plt.ylabel('y')
        ax.plot(xsort[:,1],yHatsort)
        ax.scatter(xMat[:,1].flatten().A[0],mat(labelMat).T.flatten().A[0],s=2,c='red')
    plt.show()

def test():
    dataMat,labelMat=loadDataSet('ex0.txt')
    plotlwlr(dataMat,labelMat,k=[1.0,0.01,0.003])

def rssError(yArr,yHatArr): #yArr and yHatArr both need to be arrays
    return ((yArr-yHatArr)**2).sum()

#编写岭回归程序
def ridgeRegres(xMat,yMat,lam=0.2):
    m=shape(xMat)[0]
    denom=xMat.T*xMat+lam*eye(m)
    if linalg.det(denom)==0.0:
        print('This matrix in singular,can not do inverse!')
        return
    ws=denom.I*xMat.T*yMat
    return ws

def ridgeTest(xArr,yArr):
    xMat=mat(xArr)
    yMat=mat(yArr).T

    #由于x减去了均值，为了消除这种影响，y也减去均值
    yMean=mean(yMat,axis=0)
    yMat=yMat-yMean

    xMeans=mean(xMat,axis=0)
    xVar=mean(xMat,axis=0)
    xMat=(xMat-xMeans)/xVar
    numTestPts = 30
    wMat = zeros((numTestPts, shape(xMat)[1]))
    for i in range(numTestPts):
        ws=ridgeRegres(xMat, yMat,exp(i-10))
        wMat[i,:]=ws.T
    return wMat

'''#得到所有预测值
yHat=lwlrTest(dataMat,dataMat,labelMat,k=0.003)
xMat=mat(dataMat)

strInd=xMat[:,1].argsort(axis=0)#x[strInd]会得到一个三维数组

sortIndex=strInd.flatten().A[0]#matrix转化为一维数组
#对x,y排序
xsort=xMat[sortIndex]
yHatsort=yHat[sortIndex]

#print(xSort.shape)
#xsort=xMat[strInd][:,0,:]
#print(xsort[:,1])

#画图
fig=plt.figure()
ax=fig.add_subplot(111)
#print(yHat)
#yHatsort=yHat[strInd][:,0,:]

ax.plot(xsort[:,1],yHatsort)
ax.scatter(xMat[:,1].flatten().A[0],mat(labelMat).T.flatten().A[0],s=2,c='red')

plt.show()'''


