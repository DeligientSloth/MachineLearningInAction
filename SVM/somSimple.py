import random
from numpy import*
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
            fxi=float(multiply(alpha,labelMat).T*(dataMatrix*dataMatrix[i,:].T))+b#计算判别函数
            ei=float(fxi-labelMat[i])#计算误差，如果ei则分类正确
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
            #不管是正间隔还是负间隔都会被测试，同时检查alphas保证其不能等于0或C（因为后面alpha小于0或大于C时被调整为0或C,所以
            一旦在该if语句中它们等于这两个值的话，它们就已经在边 界上了，因而不能再减少或增大）
            三种情况下，alpha对不会更新
            1、非边界点满足kkt条件
            2、得到的上下界一样
            3、alpha变化太小
            4、eta大于0（正定）
            '''
            if (labelMat[i]*ei<-toler and alpha[i]<C) or (labelMat[i]*ei>toler and alpha[i]>0):
                j=selectJrandom(i,m)#随机找不同的alpha_j
                fxj=float(multiply(alpha,labelMat).T*(dataMatrix*dataMatrix[j,:].T))+b  # 计算判别函数
                ej=float(fxj-labelMat[j])
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
                eta=float(2.0*dataMatrix[i,:]*dataMatrix[j,:].T-dataMatrix[i,:]*dataMatrix[i,:].T-\
                    dataMatrix[j,:]*dataMatrix[j,:].T)
                if eta>=0: print('eta>=0,eta={}'.format(eta)); continue
                #先计算没有剪辑的解
                alpha[j]=float(alpha_j_Old-labelMat[j]*(ei-ej)/eta)
                alpha[j]=clipAlpha(alpha[j],H,L)#对理想解加以剪辑
                if abs(alpha[j]-alpha_j_Old)<0.00001:
                    print("j not moving enough"); continue
                #往相反方向调整alphai
                alpha[i]=float(alpha_i_Old+labelMat[i]*labelMat[j]*(alpha_j_Old-alpha[j]))
                #求b1new和b2new
                b1=b-ei-labelMat[i]*(dataMatrix[i,:]*dataMatrix[i,:].T)*(alpha[i]-alpha_i_Old)\
                   -labelMat[j]*(dataMatrix[i,:]*dataMatrix[j,:].T)*(alpha[j]-alpha_j_Old)
                b2=b-ej-labelMat[i]*(dataMatrix[i,:]*dataMatrix[j,:].T)*(alpha[i]-alpha_i_Old)\
                   -labelMat[j]*(dataMatrix[j,:]*dataMatrix[j,:].T)*(alpha[j]-alpha_j_Old)
                if alpha[i]>0 and alpha[i]<C: b=float(b1)
                elif alpha[j]>0 and alpha[j]<C: b=float(b2)
                else:
                    b=float(b1+b2)/2.0
                alphaChangedCount+=1
                print("iter: %d i:%d, pairs changed %d" % (iter, i, alphaChangedCount))

            #判断
        if alphaChangedCount==0: iter+=1
        else: iter=0
        print("iteration number: %d" % iter)
    return alpha,b