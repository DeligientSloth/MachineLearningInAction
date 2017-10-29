def getNumLeafs(decisionTree):
    numLeafs = 0
    firstStr = list(decisionTree.keys())[0]#does not support indexing,所以转化为list处理,！！！返回的是迭代器
    secondDict=decisionTree[firstStr]#这里相当于把后面的子树都放到字典里：{0: 'no', 1: {'flippers': {0: 'no', 1: 'yes'}}}

    for featValue in secondDict.keys():#对上一层划分属性的取值进行遍历
        #print('This for--loop, fistFeat={},featValue={},numLeafs={}'.format(firstStr, featValue, numLeafs))
        if type(secondDict[featValue]).__name__=='dict':#判断是否到达叶子节点
            #print('进入特征 {}取值为 {} 的分出来的子树 :{}'.format(firstStr,featValue,secondDict[featValue]))
            numLeafs+=getNumLeafs(secondDict[featValue])
        else:#这个是递归返回条件了，遇到叶子节点+1返回
            #print('下一个就到达了叶子节点:{}'.format(secondDict[featValue]))
            numLeafs+=1
    return numLeafs


def getTreeDepth(decisionTree):
    maxDepth=0
    firstStr=list(decisionTree.keys())[0]#这里是最佳划分特征
    secondDict=decisionTree[firstStr]
    for featValue in secondDict.keys():
        #print('This for--loop, fistFeat={},featValue={},maxDepth={}'.format(firstStr, featValue, maxDepth))
        if type(secondDict[featValue]).__name__=='dict':
            depth=1+getTreeDepth(secondDict[featValue])#1+子树深度
        else:
            #print('下一个就到达了叶子节点:{},子树深度为1'.format(secondDict[featValue]))
            depth=1
        if depth>maxDepth:
            #print('遇到更深的了，最大深度要更新为:{}'.format(depth))
            maxDepth=depth
    return maxDepth