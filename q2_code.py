import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

class Node:
    def __init__(self, val=None, left=None, right=None, feature=None, threshold=None):
        self.val = val # 0 or 1
        self.left = left
        self.right = right
        self.feature = feature
        self.threshold = threshold
    def isLeafNode(self):
        if self.val != None:
            return True
        return False

def parseData(path):
    X=np.loadtxt(path,delimiter=" ", usecols=(0,1), dtype=float)
    Y=np.loadtxt(path, usecols=2, dtype=int)
    # returning [x1 x2],[y]
    return X,Y
# print(parseData("./Homework 2 data/D1.txt"))
# parseData("./Homework 2 data/D2.txt")
# parseData("./Homework 2 data/D3leaves.txt")
# parseData("./Homework 2 data/Dbig.txt")
# parseData("./Homework 2 data/Druns.txt")

class DecisionTree():
    def __init__(self):
        self.root = None
    def makeSubTree(self,x_vals,y_vals):
        # Determining Candidate Splits
        oneIndexes=np.where(y_vals==1)[0]
        zeroIndexes=np.where(y_vals==0)[0]
        # if stopping criteria is met
        if (x_vals.shape[0]==0 or len(oneIndexes)==0 or len(zeroIndexes)==0):
            # determining class label for new node
            if len(oneIndexes) >= len(zeroIndexes):
                return Node(1)
            else:
                return Node(0)
        else: # making an internal node
            # getting the best split feature and threshold
            fbsres = self.findBestSplit(x_vals,y_vals)
            leftBIndex, rightBIndex = self.split(x_vals[:, fbsres[0]], fbsres[1])
            left = self.makeSubTree(x_vals[leftBIndex, :], y_vals[leftBIndex])
            right = self.makeSubTree(x_vals[rightBIndex, :], y_vals[rightBIndex])
            return Node(None,left,right,fbsres[0],fbsres[1])
    def split(self,feature,threshold):
        leftIndex=np.where(feature>=threshold)[0]
        rightIndex=np.where(feature<threshold)[0]
        return leftIndex,rightIndex
    def findBestSplit(self,x_vals,y_vals):
        best = -1
        splitIndex, splitThreshold = None, None

        for featIndex in [0,1]:
            xCol = x_vals[:, featIndex]
            thresholds = np.unique(xCol)

            for threshold in thresholds:
                
                # calculating the information gain
                gain = self.infoGainRatio(y_vals, xCol, threshold)

                if gain > best:
                    best = gain
                    splitIndex = featIndex
                    splitThreshold = threshold
                    #print("best gain: ", best)
                    # print("split index: ", splitIndex, " split_threshold: ", splitThreshold)

        return splitIndex, splitThreshold
    def infoGainRatio(self,y_vals,feature,threshold):
        # parent and children
        parentEntropy = self.calcEntropy(y_vals)
        leftIndex, rightIndex = self.split(feature, threshold)

        if len(leftIndex) == 0 or len(rightIndex) == 0:
            return 0
        
        # weighted average of the children's entropy
        n_l, n_r = len(leftIndex), len(rightIndex)
        e_l, e_r = self.calcEntropy(y_vals[leftIndex]), self.calcEntropy(y_vals[rightIndex])
        splitEntropy = ((n_l) * e_l + (n_r) * e_r)/len(y_vals)
        featureEntropy = -((n_l/len(y_vals))*np.log2(n_l/len(y_vals)) + (n_r/len(y_vals))*np.log2(n_r/len(y_vals)))

        # calculate the infoGain
        infoGain = parentEntropy - splitEntropy
        # print(infoGain)
        return infoGain/featureEntropy
    def calcEntropy(self, y_vals):
        probability = np.unique(y_vals,return_counts=True)[1] / len(y_vals)
        arr = [e*np.log2(e) for e in probability]
        return -np.sum(arr)
    def traversal(self,node):
        if node:
            if node.isLeafNode():
                print("Leaf Node: " + str(node.val))
            else:
                print("Node with Feature: " + str(node.feature) + " Threshold: " + str(node.threshold))
                if node.left:
                    self.traversal(node.left)
                if node.right:
                    self.traversal(node.right)
    def predictionTraverse(self,x, node):
        if node.isLeafNode():
            return node.val

        if x[node.feature] >= node.threshold:
            return self.predictionTraverse(x, node.left)
        return self.predictionTraverse(x, node.right)
    def getNodes(self,node):
        if node is None:
            return 0
        return 1 + self.getNodes(node.left) + self.getNodes(node.right)
    
# q3 = parseData("./Homework 2 data/Druns.txt")
# d3 = DecisionTree()
# d3.makeSubTree(q3[0],q3[1])
# for e in [0,1]:
#         feature = q3[0][:, e]
#         thresholds = np.unique(feature)
#         print("hi")
#         for threshold in thresholds:
#             print("feature index: " + str(e) + " Threshold: " + str(threshold) + " Information Gain Ratio: " + str(d3.infoGainRatio(q3[1],feature,threshold)))

# q4 = parseData("./Homework 2 data/D3leaves.txt")
# d4 = DecisionTree()
# d4.root = d4.makeSubTree(q4[0],q4[1])
# d4.traversal(d4.root)

# q51 = parseData("./Homework 2 data/D1.txt")
# d51 = DecisionTree()
# d51.root = d51.makeSubTree(q51[0],q51[1])
# d51.traversal(d51.root)

# q52 = parseData("./Homework 2 data/D2.txt")
# d52 = DecisionTree()
# d52.root = d52.makeSubTree(q52[0],q52[1])
# d52.traversal(d52.root)

# plt.scatter(q51[0][:,0],q51[0][:,1],c=q51[1])
# plt.xlabel("x_1")
# plt.ylabel("x_2")
# plt.axhline(y=0.201829)
# plt.title("Scatter plot: D1.txt")
# plt.show()

# plt.scatter(q52[0][:,0],q52[0][:,1],c=q52[1])
# plt.xlabel("x_1")
# plt.ylabel("x_2")
# x = np.linspace(q52[0].min(), q52[0].max(), 50)
# y = -x + 1
# plt.plot(x, y)
# plt.title("Scatter plot: D2.txt")
# plt.show()

q7=parseData('./Homework 2 data/Dbig.txt')
#Generating splits of data (D32, D128, D512, D2048, D8192)
x8192,xtest,y8192,ytest=train_test_split(q7[0],q7[1],train_size=8192,stratify=q7[1])
x32,y32=x8192[:32,:],y8192[:32]
x128,y128=x8192[:128,:],y8192[:128]
x512,y512=x8192[:512,:],y8192[:512]
x2048,y2048=x8192[:2048,:],y8192[:2048]

everything={"32":(x32,y32),"128":(x128,y128),"512":(x512,y512),"2048":(x2048,y2048),"8192":(x8192,y8192)}

errors={}
numNodes={}

for size in everything:
    d7=DecisionTree()
    d7.root = d7.makeSubTree(everything[size][0],everything[size][1])
    ypred=np.array([d7.predictionTraverse(x,d7.root) for x in xtest])
    errors[size]=1-accuracy_score(ytest,ypred)
    numNodes[size]=d7.getNodes(d7.root)

nodeList=[numNodes[i] for i in numNodes]
attachedList = [i for i in errors]
errorList=[errors[i] for i in errors]

print(nodeList)
print(attachedList)
print(errorList)

plt.title('Number of Nodes vs Test Set Error')
plt.xlabel('Number of nodes')
plt.ylabel('Test Set Error')

plt.plot(attachedList,errorList)

plt.show()
 
for size in everything:
  d7=DecisionTree()
  d7.root = d7.makeSubTree(everything[size][0],everything[size][1])
  ypred=np.array([d7.predictionTraverse(x,d7.root) for x in xtest])

  plt.xlabel('x_1')
  plt.ylabel('x_2')
  plt.title(f'Scatter Plot D_{size}')
  plt.scatter(xtest[:,0],xtest[:,1],c=ypred)
  plt.show()