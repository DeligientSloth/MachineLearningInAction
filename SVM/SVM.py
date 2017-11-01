from somSimple import*
from PlattSMO import*
import matplotlib.pyplot as plt
def plotData(dataMat,labelMat,ws,b,alpha):
    plt.figure(1)
    dataMat=array(dataMat)
    labelMat=array(labelMat)
    index1=where(labelMat==1)
    index0=where(labelMat==-1)
    plt.xlabel('x0')
    plt.ylabel('x1')
    plt.scatter(dataMat[index1][:,0],dataMat[index1][:,1],c='r',marker='^',label='y==1')
    plt.scatter(dataMat[index0][:,0],dataMat[index0][:,1],c='y', marker='*',label='y==-1')
    X0=([min(dataMat[:,0]),max(dataMat[:,0])])
    X0=[2,6]
    X1=[(-ws[0,0]*X0[0]-b)/ws[0,1],(-ws[0,0]*X0[1]-b)/ws[0,1]]
    plt.plot(X0,X1)
    for i in range(shape(alpha)[0]):
        if alpha[i]>0:
            sv=dataMat[i]
            plt.text(sv[0],sv[1],'O')
            #plt.annotate('***',xy=(x,y),xytext=(x,y))
    plt.legend()
    plt.show()
if __name__ == '__main__':
    dataMat, labelMat = loadDataSet('testSet.txt')
    alpha, b =smoP(dataMat, labelMat, 0.6, 0.001, 40)
    ws=calcWs(alpha,dataMat,labelMat)
    plotData(dataMat,labelMat,ws,b,alpha)




