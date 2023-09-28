import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

def parseData(path):
    X=np.loadtxt(path,delimiter=" ", usecols=(0,1), dtype=float)
    Y=np.loadtxt(path, usecols=2, dtype=int)
    # returning [x1 x2],[y]
    return X,Y

q3 = parseData("./Homework 2 data/Dbig.txt")
x8192,xtest,y8192,ytest=train_test_split(q3[0],q3[1],train_size=8192,stratify=q3[1])
x32,y32=x8192[:32,:],y8192[:32]
x128,y128=x8192[:128,:],y8192[:128]
x512,y512=x8192[:512,:],y8192[:512]
x2048,y2048=x8192[:2048,:],y8192[:2048]

everything={"32":(x32,y32),"128":(x128,y128),"512":(x512,y512),"2048":(x2048,y2048),"8192":(x8192,y8192)}

errors={}
numNodes={}

for size in everything:
    d3=DecisionTreeClassifier(criterion='entropy')
    d3.fit(everything[size][0],everything[size][1])
    errors[size]=1-d3.score(xtest,ytest)
    numNodes[size]=d3.tree_.node_count

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