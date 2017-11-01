from somSimple import*
#这个类是为了传递常用的参数

def kernelTrans(X,A,kTup):
    m,n=shape(X)
    K=mat(zeros((m,1)))#表示A与数据集每一个向量的积
    if kTup[0]=='lin':
        K=X*A.T
    elif kTup[0]=='rbf':
        for j in range(m):
            deltaRow=X[j,:]-A
            K[j]=deltaRow*deltaRow.T
        K=exp(K/(-1*kTup[1]**2))
    else: raise NameError('Houston We Have a Problem -- \
    That Kernel is not recognized')
    return K
class optStruct:
    def __init__(self,dataMatIn,classLabels,C,toler,kTup):
        self.X=dataMatIn
        self.labelMat=classLabels
        self.C=C
        self.toler=toler
        self.m=shape(dataMatIn)[0]
        self.alphas=mat(zeros((self.m, 1)))
        self.b=0
        self.eCache=mat(zeros((self.m,2)))
        self.K = mat(zeros((self.m, self.m)))
        for i in range(self.m):
            self.K[:,i]=kernelTrans(self.X, self.X[i, :],kTup)

def calEKK(oS,k):
    #yi=(alphas.*y).T*(X*Xi.T)+b
    fk=multiply(oS.alphas,oS.labelMat).T*(oS.K[:,k])+oS.b
    Ek=float(fk-oS.labelMat[k])
    return Ek
def selectJK(i,oS,Ei):
    maxK=-1
    maxDeltaE=0;Ej=0
    oS.eCache[i]=[1,Ei]#Ei已经计算了，放到缓存里面去
    validEcacheList=nonzero(oS.eCache[:,0].A)[0]#表示沿着axis=0，列方向的非0值，这一项为1表示之前已经计算过了
    if len(validEcacheList)>1:
        for k in validEcacheList:
            if k==i:continue
            else:
                 Ek=calEKK(oS,k)
                 deltaE=abs(Ek-Ei)
                 if deltaE>maxDeltaE:
                     maxK=k;maxDeltaE=deltaE;Ej=Ek;
        return maxK,Ej
    else:
        j=selectJrandom(i,oS.m)
        Ej=calEKK(oS,j)
    return j,Ej

#每次alpha值更新都需要重新更新一次ecache列表
def updateEkK(oS,k):
    Ek=calEKK(oS,k)
    oS.eCache[k]=[1,Ek]

#对于第i个样本，选择j更新alphas的程序写成函数
def innerLK(oS,i):
    Ei=calEKK(oS,i)
    if (oS.labelMat[i]*Ei<-oS.toler and oS.alphas[i]<oS.C) or (oS.labelMat[i]*Ei>oS.toler and oS.alphas[i]>0):
        j,Ej=selectJK(i,oS,Ei)
        alphaIOld=oS.alphas[i].copy()
        alphaJ0ld=oS.alphas[j].copy()

        #计算上下界
        if oS.labelMat[i]==oS.labelMat[j]:
            L=max(0,alphaIOld+alphaJ0ld-oS.C)
            H=min(oS.C,alphaIOld+alphaJ0ld)
        else:
            L=max(0,alphaJ0ld-alphaIOld)
            H=min(oS.C,alphaJ0ld-alphaIOld+oS.C)

        if L==H: print('L==H');return 0

        eta=float(2.0*oS.K[i,j]-oS.K[i,i]-oS.K[j,j])
        if eta>=0: print('eta>=0');return 0

        #更新alpha
        oS.alphas[j]-=oS.labelMat[j]*(Ei-Ej)/eta
        oS.alphas[j]=clipAlpha(oS.alphas[j],H,L)

        #更新缓存
        updateEkK(oS,j)

        if abs(oS.alphas[j]-alphaJ0ld)<=0.00001:print('j not moving enough');return 0

        #更新alphai
        oS.alphas[i]+=oS.labelMat[i]*oS.labelMat[j]*(alphaJ0ld-oS.alphas[j])
        updateEkK(oS,i)

        #计算b
        b1=oS.b-Ei+oS.labelMat[i]*(alphaIOld-oS.alphas[i])*oS.K[i,i]+\
               oS.labelMat[j]*(alphaJ0ld-oS.alphas[j])*oS.K[i,j]
        b2=oS.b-Ej+oS.labelMat[i]*(alphaIOld-oS.alphas[i])*oS.K[i,j]+\
                 oS.labelMat[j]*(alphaJ0ld-oS.alphas[j])*oS.K[j,j]
        if oS.alphas[i]>0 and oS.alphas[i]<oS.C:
            oS.b=float(b1)
        elif oS.alphas[j]>0 and oS.alphas[j]<oS.C:
            oS.b=float(b2)
        else:
            oS.b=float(b1+b2)/2
        return 1#代表改变了一次
    else: return 0

def smoPK(dataMatIn,classLabels,C,toler,maxIter,kTup):
    oS=optStruct(mat(dataMatIn),mat(classLabels).transpose(),C,toler,kTup)
    iter=0
    entireSet=True
    alphaParisChanged=0
    while iter<maxIter and (entireSet or alphaParisChanged>0):
        alphaParisChanged=0#每次进入循环，改变次数都是0
        #两种遍历方式 1、整体遍历 2、遍历C的非边界点
        if entireSet:
            for i in range(oS.m):
                alphaParisChanged+=int(innerLK(oS,i))
                print('fullset iterations, iter=%d,i=%d,pairschanged=%d',iter,i,alphaParisChanged)
            iter+=1#迭代次数加1
        else:
            nonBounds=nonzero((oS.alphas.A>0)*(oS.alphas.A<oS.C))[0]
            for i in nonBounds:
                alphaParisChanged+=int(innerLK(oS,i))
                print('non-bounds iterations, iter=%d,i=%d,pairschanged=%d',iter,i,alphaParisChanged)
            iter+=1
        if entireSet: entireSet=False
        elif alphaParisChanged==0:entireSet=True
        print('iterations : %d is over',iter)
    return oS.alphas,oS.b

def calcWs(alphas,dataArr,classLabels):
    X = mat(dataArr); labelMat = mat(classLabels).transpose()
    m,n = shape(X)
    w = zeros((n,1))
    #for i in range(m):
        #w += multiply(alphas[i]*labelMat[i],X[i,:].T)
    w=multiply(alphas,labelMat).T*X
    return w
def testRbf(k1=1.3):
    dataArr,labelArr = loadDataSet('testSetRBF.txt')
    alphas,b = smoPK(dataArr, labelArr, 200, 0.0001, 10000, ('rbf', k1)) #C=200 important
    datMat=mat(dataArr); labelMat = mat(labelArr).transpose()
    svInd=nonzero(alphas.A>0)[0]
    sVs=datMat[svInd] #get matrix of only support vectors
    labelSV = labelMat[svInd];
    print("there are %d Support Vectors" % shape(sVs)[0])
    m,n = shape(datMat)
    errorCount = 0
    for i in range(m):
        kernelEval = kernelTrans(sVs,datMat[i,:],('rbf', k1))
        predict=kernelEval.T * multiply(labelSV,alphas[svInd]) + b
        if sign(predict)!=sign(labelArr[i]): errorCount += 1
    print("the training error rate is: %f" % (float(errorCount)/m))
    dataArr,labelArr = loadDataSet('testSetRBF2.txt')
    errorCount = 0
    datMat=mat(dataArr); labelMat = mat(labelArr).transpose()
    m,n = shape(datMat)
    for i in range(m):
        kernelEval = kernelTrans(sVs,datMat[i,:],('rbf', k1))
        predict=kernelEval.T * multiply(labelSV,alphas[svInd]) + b
        if sign(predict)!=sign(labelArr[i]): errorCount += 1
    print("the test error rate is: %f" % (float(errorCount)/m))

if __name__ == '__main__':
    testRbf(0.7)