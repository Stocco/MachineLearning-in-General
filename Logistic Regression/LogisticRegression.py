from math import *
from numpy import *
from numpy import linalg as LA
# PARSING FILES
fileTrain = open("D7train.txt")
train = fileTrain.read()
yTr = []
xTr = []
yDev = []
xDev = []
betas = []
converged = False

def loadD7():
    for c,row in enumerate(train.split("\n")):
            yTr.append(int(row[0]))
            cur = []
            if(c == 0): betas.append(0.0)
            cur.append(1.0)
            for col in row[1:].split(" "):
                try:
                    cur.append(float(col))
                    if( c == 0): betas.append(1.0)
                except ValueError:
                     pass
            xTr.append(cur)


    fileDev = open("D7dev.txt")
    dev = fileDev.read()

    for row in dev.split("\n"):
            yDev.append(int(row[0]))
            cur = []
            cur.append(1.0)
            for col in row[1:].split(" "):
                try:
                    cur.append(float(col))
                except ValueError:
                     pass
            xDev.append(cur)

def sigma(value):
         return 1.0/(1+ exp(-value))

# until threshold:  < 0.00001
def checkConversion(old, new):
     result = 0.0
     global converged
     oldSum = LA.norm(old)
     newSum = LA.norm(new)
     result = math.fabs((oldSum - newSum)/oldSum)
     print(result)

     if(result < 0.00001):
         converged = True;
     else:
         converged = False;

def logistic():
    X = mat(xTr)
    Y = mat(yTr).transpose()
    m,n  = shape(X)
    weights = zeros((n,1))


    while(converged != True):
          h = sigma(X*weights)
          error = (Y - h)
          oldVal = weights.copy()
          weights = weights + (0.001 * X.transpose() * error) #gradient ascend formula, with step size hard coded to 0.000001
          checkConversion(oldVal,weights)


    return weights

def predict():
     count=0
     total = 0
     for n in range(xTr.__len__()):
        total += 1
        yVal = yDev[n]
        xhat = mat(xDev[n])
        prediction = sigma(sum(xhat*trueBeta))
        if(prediction > 0.5):
            if(yVal == 1): count = count + 1
            print("Value is 1 with prob ",prediction, " True value : ", yVal, " Error rate: ", count/total  )
        else:
            if(yVal == 0): count = count + 0
            print("Value is 0 with prob ",prediction, " True value : ", yVal, " Error rate: ", count/total  )

def plotBfist():
    import matplotlib.pyplot as plt
    arr = array(datamat)
    n = shape(arr)[0]
    xcord1 = []
    xcord2 = []
    xcord3 = []
    ycord1 = []
    ycord2 = []
    ycord3 = []
    for i in range(n):
         if(int(yTr[i]) == 0):
               xcord1.append(arr[i,0])
               ycord1.append(arr[i,1])
         elif(int(yTr[i] == 1)):
             xcord2.append(arr[i,0])
             ycord2.append(arr[i,1])
         else:
             xcord3.append(arr[i,0])
             ycord3.append(arr[i,1])

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1,ycord1, s=30,c='red', marker='s')
    ax.scatter(xcord2,ycord2,s=30,c='blue')
    ax.scatter(xcord3,ycord3,s=30,c='green')
    x = arange(-3.0,3.0,0.1)
    y1 = (-trueBeta[0].tolist()[0][0]-trueBeta[0].tolist()[1][0] * x)/trueBeta[0].tolist()[2][0]
    y2 = (-trueBeta[1].tolist()[0][0]-trueBeta[1].tolist()[1][0] * x)/trueBeta[1].tolist()[2][0]
    y3 = (-trueBeta[2].tolist()[0][0]-trueBeta[2].tolist()[1][0] * x)/trueBeta[2].tolist()[2][0]
    ax.plot(x,y1)
    ax.plot(x,y2)
    ax.plot(x,y3)
    plt.show()

fileAbalo = open("abalo.txt").read()

datamat  = []
for row in fileAbalo.split("\n"):
    row = row.split(",")
    if(row[0] == "M"): yTr.append(0)
    elif(row[0] == "F"): yTr.append(1)
    elif(row[0] == "I"): yTr.append(2)
    cur = []
    cur2 = []
    cur.append(0.0)
    cur.append(float(row[1]))
    cur.append(float(row[2]))
    xTr.append(cur)
    cur2 = cur.copy()
    del cur2[0]
    datamat.append(cur2)


def mLogistic():
    global converged
    allBetas=[]
    allX = [[],[],[]]
    allY = [[],[],[]]
    for n in range(0,xTr.__len__()):
        if(yTr[n] == 0):
            allX[0].append(xTr[n])
            allY[0].append(yTr[n])
        elif(yTr[n] == 1 ):
            allX[1].append(xTr[n])
            allY[1].append(yTr[n])
        elif(yTr[n] == 2):
            allX[2].append(xTr[n])
            allY[2].append(yTr[n])

    for m in range(0,3):
        X = mat(allX[m])
        Y = mat(allY[m]).transpose()
        m,n  = shape(X)
        weights = zeros((n,1))
        converged=False;
        while(converged != True):
              h = sigma(X*weights)
              error = (Y - h)
              oldVal = weights.copy()
              weights = weights + (0.00001 * X.transpose() * error) #gradient ascend formula, with step size hard coded to 0.000001
              checkConversion(oldVal,weights)
        allBetas.append(weights)

    return allBetas

trueBeta = logistic()
print(trueBeta)
plotBfist()
#predict()

