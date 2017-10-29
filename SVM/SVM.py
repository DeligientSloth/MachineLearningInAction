from numpy import *
import random
#先写辅助函数
def loadDataSet(filename):
    dataMat=[];labelMat=[]
    fr=open(filename)
    for line in fr.readlines():
        lineArr=line.strip().split('\t')
        dataMat.append([float(lineArr[0]),float(lineArr[1])])
        labelMat.append(float(lineArr[2]))
    return dataMat,labelMat
def selectJrandom(i,m):
    j=i#必须产生一个随机数，注意不能初始化j=0或者其他数字
    while(i==j):
        #j=int(random.uniform(0,m))#这个随机数向下取整，uniform函数不包含右边，产生随机数范围0-m-1的整数
        #如果使用random.randint，产生的随机数包含右端点
        j=random.randint(0,m-1)
    return j
def clipAlpha(alpha,H,L):
    if alpha>H:
        alpha=H
    if alpha<L:
        alpha=L
    return alpha

def SMOSimple(dataMatIn,classLabels,C,toler,maxIter):
    #输入数据转换成矩阵
    dataMatrix=mat(dataMatIn)
    labelMat=mat(classLabels).transpose()#转换成列向量
    m,n=dataMatrix.shape
    alpha=mat(zeros((m,1)))#每条样本对应一个alpha，所以是m by 1的列向量
    b=0#b只有一个，实数
    #在每一次迭代过程，都会遍历整个数据集去取alpha_i,检查是否满足kkt条件，如果不满足，取一个alpha_j进行更新,更新次数+1
    #遍历完成后，检查更新次数，若发生了更新，就将迭代次数置零，重新开始迭代
    #只有当alpha不再更改，并且达到最大迭代次数的时候，才会退出while循环，如果alpha发生了更改，需要重新迭代
    iter=0
    while(iter<maxIter):
        alphaChangedCount=0#记录alpha对的改变次数
        for i in range(m):#遍历整个数据集
            fxi=multiply(alpha,labelMat).T*(dataMatrix*dataMatrix[i,:].T)+b#计算判别函数
            ei=fxi-labelMat[i]#计算误差，如果ei则分类正确
            #找出不满足kkt条件的alpha_i，没有则直接走出循环
            #这里是间隔边界上的支持向量
            '''
            这里的违反kkt条件是指间隔边界上的点，因为间隔边界上的点更有可能需要调整，对于边界点（alpha=C或者alpha=0）
            往往得不到调整：
            alphai=0 yi*fi-1>=0
            0<alphai<C yi*fi-1=0
            alphai=C yi*fi-1<=0
            对于非边界点而言，0<alphai<C，若违反kkt条件，则有，yi*fi-1!=0，实际上一般有容忍度tolerance
            即当0<alphai<C，有yi*fi-1<-toler 或者 yi*fi-1>toler
            但是yi*fi-1<-toler包含了alphai=C的情况，yi*fi-1>toler包含了alphai=0的情况
            所以排除这两种情况：yi*fi-1<-toler and alphai<C 或者 yi*fi-1>toler and alphai>0
            当某一次遍历发现没有非边界数据样本得到调整时，遍历所有数据样本，以检验是否整个集合都满足KKT条件。如果整个集合的检
            验中又有数据样本被进一步进化，则有必要再遍历非边界数据样本。这样，不停地在遍历所有数据样本和遍历非边界数据样本之间切换，
            直到整个样本集合都满足KKT条件为止。以上用KKT条件对数据样本所做的检验都以达到一定精度ε就可以停止为条件。如果要求十分精
            确的输出算法，则往往不能很快收敛。
            三种情况下，alpha对不会更新
            1、非边界点满足kkt条件
            2、得到的上下界一样
            3、alpha变化太小
            4、eta大于0（正定）
            '''
            if (labelMat[i]*ei<-toler and alpha[i]<C) or (labelMat[i]*ei>toler and alpha[i]>0):
                j=selectJrandom(i,m)#随机找不同的alpha_j
                fxj=multiply(alpha,labelMat).T*(dataMatrix*dataMatrix[j,:].T)+b  # 计算判别函数
                ej=fxj-labelMat[j]
                alpha_i_Old=alpha[i].copy()
                alpha_j_Old=alpha[j].copy()#先保存现在刚刚选择的alpha值，马上更新
                #先求约束条件
                if labelMat[i]==labelMat[j]:
                    H=min(C,alpha_i_Old+alpha_j_Old)
                    L=max(0,alpha_i_Old+alpha_j_Old-C)
                else:
                    H=min(C,C+alpha_j_Old-alpha_i_Old)
                    L=max(0,alpha_j_Old-alpha_i_Old)
                if H==L:print("L==H"); continue
                eta=2.0*dataMatrix[i,:]*dataMatrix[j,:].T-dataMatrix[i,:]*dataMatrix[i,:].T-\
                    dataMatrix[j,:]*dataMatrix[j,:].T
                if eta>=0: print('eta>=0,eta={}'.format(eta)); continue
                #先计算没有剪辑的解
                alpha[j]=alpha_j_Old-labelMat[j]*(ei-ej)/eta
                alpha[j]=clipAlpha(alpha[j],H,L)#对理想解加以剪辑
                if abs(alpha[j]-alpha_j_Old)<0.00001:
                    print("j not moving enough"); continue
                #往相反方向调整alphai
                alpha[i]=alpha_i_Old+labelMat[i]*labelMat[j]*(alpha_j_Old-alpha[j])
                #求b1new和b2new
                b1=b-ei-labelMat[i]*(dataMatrix[i,:]*dataMatrix[i,:].T)*(alpha[i]-alpha_i_Old)\
                   -labelMat[j]*(dataMatrix[i,:]*dataMatrix[j,:].T)*(alpha[j]-alpha_j_Old)
                b2=b-ej-labelMat[i]*(dataMatrix[i,:]*dataMatrix[j,:].T)*(alpha[i]-alpha_i_Old)\
                   -labelMat[j]*(dataMatrix[j,:]*dataMatrix[j,:].T)*(alpha[j]-alpha_j_Old)
                if alpha[i]>0 and alpha[i]<C: b=b1
                elif alpha[j]>0 and alpha[j]<C: b=b2
                else:
                    b=(b1+b2)/2.0
                alphaChangedCount+=1
                print("iter: %d i:%d, pairs changed %d" % (iter, i, alphaChangedCount))

            #判断
        if alphaChangedCount==0: iter+=1
        else: iter=0
        print("iteration number: %d" % iter)
    return alpha,b
class optStruct:
    def __init__(self,dataMatIn,classLabels,C,toler):
        self.X=dataMatIn
        self.labelMat=classLabels
        self.C=C
        self.tol=toler
        self.m=shape(dataMatIn)[0]
        #运算得到的
        self.alphas=mat(zeros((self.m,1)))
        self.b=0
        self.eCache=mat(zeros((self.m,2)))#第一列是是否有效的标志位
        self.K=mat(zeros((self.m,self.m)))

def calcEk(oS,k):
    #fk=multiply(os.alpha,os.labelMat).T*os.K[:,k]+os.b
    fk=multiply(oS.alpha,oS.labelMat).T*(oS.X*oS.X[k,:].T)+oS.b
    Ek=fk-oS.labelMat[k]
    return Ek

def selectJ(i, oS, Ei):         #this is the second choice -heurstic, and calcs Ej
    maxK = -1; maxDeltaE = 0; Ej = 0
    oS.eCache[i] = [1,Ei]  #set valid #choose the alpha that gives the maximum delta E，有效意味着计算好了
    validEcacheList = nonzero(oS.eCache[:,0].A)[0]#返回非零E值对应的alpha值
    if (len(validEcacheList)) > 1:
        for k in validEcacheList:   #loop through valid Ecache values and find the one that maximizes delta E
            if k == i: continue #don't calc for i, waste of time
            Ek = calcEk(oS, k)
            deltaE = abs(Ei - Ek)
            if (deltaE > maxDeltaE):
                maxK = k; maxDeltaE = deltaE; Ej = Ek
        return maxK, Ej
    else:   #in this case (first time around) we don't have any valid eCache values
        j = selectJrandom(i, oS.m)
        Ej = calcEk(oS, j)
    return j, Ej

def updateEk(oS, k):#after any alpha has changed update the new value in the cache
    Ek = calcEk(oS, k)
    oS.eCache[k] = [1,Ek]
def innerL(i, oS):
    Ei = calcEk(oS, i)
    if ((oS.labelMat[i]*Ei < -oS.tol) and (oS.alphas[i] < oS.C)) or ((oS.labelMat[i]*Ei > oS.tol) and (oS.alphas[i] > 0)):
        j,Ej = selectJ(i, oS, Ei) #this has been changed from selectJrand
        alphaIold = oS.alphas[i].copy(); alphaJold = oS.alphas[j].copy();
        if (oS.labelMat[i] != oS.labelMat[j]):
            L = max(0, oS.alphas[j] - oS.alphas[i])
            H = min(oS.C, oS.C + oS.alphas[j] - oS.alphas[i])
        else:
            L = max(0, oS.alphas[j] + oS.alphas[i] - oS.C)
            H = min(oS.C, oS.alphas[j] + oS.alphas[i])
        if L==H: print("L==H"); return 0
        eta = 2.0 * oS.K[i,j] - oS.K[i,i] - oS.K[j,j] #changed for kernel
        if eta >= 0: print("eta>=0"); return 0
        oS.alphas[j] -= oS.labelMat[j]*(Ei - Ej)/eta
        oS.alphas[j] = clipAlpha(oS.alphas[j],H,L)
        updateEk(oS, j) #added this for the Ecache
        if (abs(oS.alphas[j] - alphaJold) < 0.00001): print("j not moving enough"); return 0
        oS.alphas[i] += oS.labelMat[j]*oS.labelMat[i]*(alphaJold - oS.alphas[j])#update i by the same amount as j
        updateEk(oS, i) #added this for the Ecache                    #the update is in the oppostie direction
        b1 = oS.b - Ei- oS.labelMat[i]*(oS.alphas[i]-alphaIold)*oS.K[i,i] - oS.labelMat[j]*(oS.alphas[j]-alphaJold)*oS.K[i,j]
        b2 = oS.b - Ej- oS.labelMat[i]*(oS.alphas[i]-alphaIold)*oS.K[i,j]- oS.labelMat[j]*(oS.alphas[j]-alphaJold)*oS.K[j,j]
        if (0 < oS.alphas[i]) and (oS.C > oS.alphas[i]): oS.b = b1
        elif (0 < oS.alphas[j]) and (oS.C > oS.alphas[j]): oS.b = b2
        else: oS.b = (b1 + b2)/2.0
        return 1
    else: return 0

def smoP(dataMatIn, classLabels, C, toler, maxIter,kTup=('lin', 0)):    #full Platt SMO
    oS = optStruct(mat(dataMatIn),mat(classLabels).transpose(),C,toler, kTup)
    iter = 0
    entireSet = True; alphaPairsChanged = 0
    #当迭代次数达到或者数据集中没有alpha更新都会推出循环
    while (iter < maxIter) and ((alphaPairsChanged > 0) or (entireSet)):
        alphaPairsChanged = 0
        if entireSet:   #go over all
            for i in range(oS.m):
                alphaPairsChanged += innerL(i,oS)
                print("fullSet, iter: %d i:%d, pairs changed %d" % (iter,i,alphaPairsChanged))
            iter += 1
        else:#go over non-bound (railed) alphas
            nonBoundIs = nonzero((oS.alphas.A > 0) * (oS.alphas.A < C))[0]
            for i in nonBoundIs:
                alphaPairsChanged += innerL(i,oS)
                print("non-bound, iter: %d i:%d, pairs changed %d" % (iter,i,alphaPairsChanged))
            iter += 1
        if entireSet: entireSet = False #toggle entire set loop
        elif (alphaPairsChanged == 0): entireSet = True
        print("iteration number: %d" % iter)
    return oS.b,oS.alphas

if __name__ == '__main__':
    dataMat,labelMat=loadDataSet('testSet.txt')
    alpha,b=SMOSimple(dataMat,labelMat,0.6,0.001,40)
    print(alpha[alpha>0])
    print(b)


