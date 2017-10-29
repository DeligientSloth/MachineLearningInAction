import numpy as np
import random

def loadDataSet():
    postingList=[['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                 ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                 ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0,1,0,1,0,1]    #1 is abusive, 0 not,1代表侮辱性的言论，0代表不是
    return postingList,classVec
def createVocabList(dataSet):

    vocabSet=set([])#创建一个空集合
    for document in dataSet:
        #print(document)
        vocabSet=vocabSet|(set(document))#求并集，与|应该一样
    return list(vocabSet)
def setOfWords2Vec(vocabList,inputSet):
    word2Vec=len(vocabList)*[0]#创建与vocablist长度相同的向量，表示每个词是否出现
    for word in inputSet:
        if word in vocabList:#检查单词是否出现在单词表中
            word2Vec[vocabList.index(word)]=1
        else: print('The word : {} is not in my vocabulary List!'.format(word))
    return word2Vec

def bagOfWords2Vec(vocabList,inputSet):
    word2Vec=len(vocabList)*[0]#创建与vocablist长度相同的向量，表示每个词是否出现
    for word in inputSet:
        if word in vocabList:#检查单词是否出现在单词表中
            word2Vec[vocabList.index(word)]+=1
        else: print('The word : {} is not in my vocabulary List!'.format(word))
    return word2Vec
'''
训练一个简单的贝叶斯0-1模型
@trainMatrix: 嵌套list，每个元素list代表每篇文档的word2vec
@trainCategory：每篇文档的类别
'''
def trainNBayes0(trainMatrix,trainCategory):
    if isinstance(trainMatrix,np.ndarray)!=True:
        trainMatrix=np.array(trainMatrix)
    numTrainDoes=len(trainMatrix)#得到训练文档数量
    numWords=len(trainMatrix[0])#得到词汇表的长度
    pAbusive=sum(trainCategory)/float(len(trainCategory))
    #p0Num=np.zeros(numWords);p1Num=np.zeros(numWords)#初始化两个类别下的，词汇表中每个单词出现数量的矩阵
    #p0Denom=p1Denom=0.0
    p0Num=np.ones(numWords)
    p1Num=np.ones(numWords) #初始化两个类别下的，词汇表中每个单词出现数量的矩阵,设定每个单词出现概率至少为1
    p0Denom=2.0
    p1Denom=2.0
    for i in range(numTrainDoes):#遍历训练文档
        if(trainCategory[i]==1):
            p1Num+=trainMatrix[i]#得到一个矩阵，每一项代表类别为1的文档中，该单词出现数量
            p1Denom+=sum(trainMatrix[i])#此篇文章中，单词总数，结果是类别为1的文档的单词总数
        else:
            p0Num+=trainMatrix[i]  # 得到一个矩阵，每一项代表类别为0的文档中，该单词出现数量
            p0Denom+=sum(trainMatrix[i])  # 此篇文章中，单词总数，结果是类别为0的文档的单词总数
    p1Vect=np.log(p1Num/float(p1Denom))
    p0Vect=np.log(p0Num/float(p0Denom))
    return p1Vect,p0Vect,pAbusive

def classifyNBayes0(w2VecToClass,p1Vec,p0Vec,p1Class):
    p1=sum(w2VecToClass*p1Vec)+np.log(p1Class)
    p0=sum(w2VecToClass*p0Vec)+np.log(1.0-p1Class)
    if p1>p0: return 1
    else: return 0

def testingNB():
    dataSet, classVec = loadDataSet()
    vocabList = createVocabList(dataSet)
    trainMatrix = []  # 初始化
    for document in dataSet:
        word2Vec = setOfWords2Vec(vocabList, document)
        trainMatrix.append(word2Vec)
    p1Vect,p0Vect,pAbusive=trainNBayes0(np.array(trainMatrix),np.array(classVec))
    testEntry=['love','my','dalmation', 'stupid']
    word2VecTest=np.array(setOfWords2Vec(vocabList,testEntry))
    print(testEntry,'is classified as: ',classifyNBayes0(word2VecTest,p1Vect,p0Vect,pAbusive))

    testEntry=['stupid', 'garbage']
    word2VecTest=np.array(setOfWords2Vec(vocabList, testEntry))
    print(testEntry,'is classified as: ',classifyNBayes0(word2VecTest, p1Vect, p0Vect, pAbusive))


def textParse(bigString):  # input is big string, #output is word list
    import re
    listOfTokens = re.split(r'\W*', bigString)
    return [tok.lower() for tok in listOfTokens if len(tok) > 2]

def spamTest():
    docList=[]
    classList=[]
    fullText=[]
    for i in range(1,26):
        wordList1 = textParse(open('email/spam/%d.txt' % i).read())
        docList.append(wordList1)
        fullText.extend(wordList1)
        classList.append(1)
        wordList2 = textParse(open('email/ham/%d.txt' % i).read())
        docList.append(wordList2)
        fullText.extend(wordList2)
        classList.append(0)
        #构建词汇表
    vocabList=createVocabList(docList)
    trainingSetIndex=list(range(50))
    testSetIndex=[]#记录测试集的下标
    for index in range(10):
        randomIndex=int(random.uniform(0,len(trainingSetIndex)))
        #randomIndex=random.randint(0,len(trainingSetIndex)-1)#本程序直接采用这个函数,这是测试集的下标
        testSetIndex.append(randomIndex)
        del trainingSetIndex[randomIndex]#从训练集下标中删除
    #下面开始根据下标划分训练集以及交叉验证集合,在doclist中进行划分,分别转化为0-1向量
    trainingMat=[]
    trainingClass=[]
    for docIndex in trainingSetIndex:
        bagOfWord2Vec=bagOfWords2Vec(vocabList,docList[docIndex])
        trainingMat.append(bagOfWord2Vec)
        trainingClass.append(classList[docIndex])

    #开始训练
    p1Vec,p0Vec,pSpam=trainNBayes0(np.array(trainingMat),np.array(trainingClass))
    #corss validation阶段
    errorCount=0
    print(testSetIndex)

    for testdocIndex in testSetIndex:
        testbagOfWord2Vec=bagOfWords2Vec(vocabList,docList[testdocIndex])
        print(testbagOfWord2Vec)
        if classifyNBayes0(np.array(testbagOfWord2Vec),p1Vec,p0Vec,pSpam)!=classList[testdocIndex]:
            errorCount+=1
            print('出现分类错误，文件{}，属于{}，被分类错了'.format(docList[testdocIndex],classList[testdocIndex]))

    print('The error rate is: {}'.format(float(errorCount)/(len(testSetIndex))))






'''def spamTest():
    docList = [];
    classList = [];
    fullText = []
    for i in range(1, 26):
        wordList = textParse(open('email/spam/%d.txt' % i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)
        wordList = textParse(open('email/ham/%d.txt' % i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    vocabList = createVocabList(docList)  # create vocabulary
    trainingSet = range(50);
    testSet = []  # create test set
    for i in range(10):
        randIndex = int(random.uniform(0, len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del (trainingSet[randIndex])
    trainMat = [];
    trainClasses = []
    for docIndex in trainingSet:  # train the classifier (get probs) trainNB0
        trainMat.append(bagOfWords2VecMN(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0V, p1V, pSpam = trainNB0(array(trainMat), array(trainClasses))
    errorCount = 0
    for docIndex in testSet:  # classify the remaining items
        wordVector = bagOfWords2VecMN(vocabList, docList[docIndex])
        if classifyNB(array(wordVector), p0V, p1V, pSpam) != classList[docIndex]:
            errorCount += 1
            print
            "classification error", docList[docIndex]
    print
    'the error rate is: ', float(errorCount) / len(testSet)'''
if __name__ == '__main__':
    #testingNB()
    spamTest()
    '''dataSet,classVec=loadDataSet()
    vocabList=createVocabList(dataSet)
    trainMatrix=[]#初始化
    for document in dataSet:
        word2Vec=setOfWords2Vec(vocabList,document)
        trainMatrix.append(word2Vec)
    p1Vect, p0Vect, pAbusive=trainNBayes0(trainMatrix,classVec)
    np.set_printoptions(threshold=np.inf)
    print('词汇表如下：')
    print(vocabList)
    print('类别为1的单词出现概率向量如下：')
    print(p1Vect)
    print('类别为0的单词出现概率向量如下：')
    print(p0Vect)
    print('单词cute的概率如下：')
    print('类别1中，p1(cute)=: {}'.format(p1Vect[vocabList.index('cute')]))
    print('类别0中，p1(cute)=: {}'.format(p0Vect[vocabList.index('cute')]))#理论上为1/24=0.041666666666666664
    print('p1向量中，出现最频繁的单词是：%s'%vocabList[np.argmax(p1Vect)])'''
    '''print(trainMatrix[0])
    print(vocabList)
    print(p1Vect)
    print(p0Vect)
    print(pAbusive)'''
