#!/usr/bin/env python
# coding: utf-8

# In[73]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

path_to_train_csv = "./IA2-train.csv"
path_to_dev_csv = "./IA2-dev.csv"

originalData = pd.read_csv(path_to_train_csv)
testData = pd.read_csv(path_to_dev_csv)


# In[74]:


originalData.head()


# ## Data prepping

# In[75]:


normalizeData = originalData.copy()
normalizeData['Age'] = (originalData['Age'] - originalData['Age'].mean()) / originalData['Age'].std()
normalizeData['Annual_Premium'] = (originalData['Annual_Premium'] - originalData['Annual_Premium'].mean()) / originalData['Annual_Premium'].std()
normalizeData['Vintage'] = (originalData['Vintage'] - originalData['Vintage'].mean()) / originalData['Vintage'].std()

normalizeData.head()


# In[76]:


validationData = testData
validationData['Age'] = (validationData['Age'] - originalData['Age'].mean()) / originalData['Age'].std()
validationData['Annual_Premium'] = (validationData['Annual_Premium'] - originalData['Annual_Premium'].mean()) / originalData['Annual_Premium'].std()
validationData['Vintage'] = (validationData['Vintage'] - originalData['Vintage'].mean()) / originalData['Vintage'].std()

validationData.head()


# ## Part 1

# In[ ]:


MAX_ITER = 5000
#MAX_ITER = 10000

#MIN_ERR = 0.05
MIN_ERR = 0.01
#MIN_ERR = 0.005

initialGuess = np.repeat(0.0, len(testData.columns) - 1)
## W0 = [0,0,...,0]

featureData = normalizeData.drop(['Response'], axis = 1)
featureValidationData = validationData.drop(['Response'], axis = 1)
## Get new dataset with only feature

featuresNumber = featureData.columns.size
def sigmoid(x):
    return 1/(1+np.exp(-x))
## define sigmoid function

def BGDLogic(inputData, resultData, initVector, maxIter, minErr, alphaRate, lambdaReg):
    iteration = 0
    dataSize = len(inputData)
    w = initVector.copy()
    gradientVector = ((alphaRate/dataSize)*((inputData.mul(resultData - sigmoid(inputData.mul(w).sum(axis = 1)), axis = 0)).sum(axis = 0))).values
    normGradient = np.linalg.norm(gradientVector)
    while (normGradient > minErr and not np.isinf(normGradient) and iteration < maxIter):
        w = w + gradientVector
        for index, w_j in enumerate(w):
            if (index >= 1):
                w_j = w_j - alphaRate*lambdaReg*w_j
            w[index] = w_j
        gradientVector = ((alphaRate/dataSize)*((inputData.mul(resultData - sigmoid(inputData.mul(w).sum(axis = 1)), axis = 0)).sum(axis = 0))).values
        normGradient = np.linalg.norm(gradientVector)
        iteration += 1
        print('Gradient: %9.7f, Iteration: [%d/%d]\r'%(normGradient,iteration,MAX_ITER), end="")
    return w
## Batch gradient descent L2

def getPrediction(W, features):
    predictionsArray = sigmoid(features.mul(W).sum(axis = 1))
    return list(map(lambda x: 1 if x >= 0.5 else 0, predictionsArray))

def getAccuracy(predictions, actual):
    return 1 - (np.count_nonzero(list(predictions - actual))/actual.size)

def roundingW(W):
    return list(map(lambda x: 0 if x < 1e-9 else x, W))
## Helper functions


# In[ ]:


wLambda0001 = BGDLogic(featureData, originalData['Response'], initialGuess, MAX_ITER, MIN_ERR, 0.1, 0.001)

trainAccWLambda0001 = getAccuracy(getPrediction(wLambda0001, featureData), originalData['Response'])
print('\n Training accuracy: %6.4f'%trainAccWLambda0001)

valAccWLambda0001 = getAccuracy(getPrediction(wLambda0001, featureValidationData), validationData['Response'])
print('\n Validating accuracy: %6.4f '%valAccWLambda0001)


# In[ ]:


wLambda001 = BGDLogic(featureData, originalData['Response'], initialGuess, MAX_ITER, MIN_ERR, 0.1, 0.01)

trainAccWLambda001 = getAccuracy(getPrediction(wLambda001, featureData), originalData['Response'])
print('\n Training accuracy: %6.4f'%trainAccWLambda001)

valAccWLambda001 = getAccuracy(getPrediction(wLambda001, featureValidationData), validationData['Response'])
print('\n Validating accuracy: %6.4f'%valAccWLambda001)


# In[ ]:


wLambda01 = BGDLogic(featureData, originalData['Response'], initialGuess, MAX_ITER, MIN_ERR, 0.1, 0.1)

trainAccWLambda01 = getAccuracy(getPrediction(wLambda01, featureData), originalData['Response'])
print('\n Training accuracy: %6.4f'%trainAccWLambda01)

valAccWLambda01 = getAccuracy(getPrediction(wLambda01, featureValidationData), validationData['Response'])
print('\n Validating accuracy: %6.4f'%valAccWLambda01)


# In[ ]:


wLambda1 = BGDLogic(featureData, originalData['Response'], initialGuess, MAX_ITER, MIN_ERR, 0.1, 1)

trainAccWLambda1 = getAccuracy(getPrediction(wLambda1, featureData), originalData['Response'])
print('\n Training accuracy: %6.4f'%trainAccWLambda1)

valAccWLambda1 = getAccuracy(getPrediction(wLambda1, featureValidationData), validationData['Response'])
print('\n Validating accuracy: %6.4f'%valAccWLambda1)


# In[ ]:


wLambda10 = BGDLogic(featureData, originalData['Response'], initialGuess, MAX_ITER, MIN_ERR, 0.1, 10)

trainAccWLambda10 = getAccuracy(getPrediction(wLambda10, featureData), originalData['Response'])
print('\n Training accuracy: %6.4f'%trainAccWLambda10)

valAccWLambda10 = getAccuracy(getPrediction(wLambda10, featureValidationData), validationData['Response'])
print('\n Validating accuracy: %6.4f'%valAccWLambda10)


# In[ ]:


wLambda100 = BGDLogic(featureData, originalData['Response'], initialGuess, MAX_ITER, MIN_ERR, 0.1, 100)

trainAccWLambda100 = getAccuracy(getPrediction(wLambda100, featureData), originalData['Response'])
print('\n Training accuracy: %6.4f'%trainAccWLambda100)

valAccWLambda100 = getAccuracy(getPrediction(wLambda100, featureValidationData), validationData['Response'])
print('\n Validating accuracy: %6.4f'%valAccWLambda100)


# In[ ]:


wLambda1000 = BGDLogic(featureData, originalData['Response'], initialGuess, MAX_ITER, MIN_ERR, 0.1, 1000)

trainAccWLambda1000 = getAccuracy(getPrediction(wLambda1000, featureData), originalData['Response'])
print('\n Training accuracy: %6.4f'%trainAccWLambda1000)

valAccWLambda1000 = getAccuracy(getPrediction(wLambda1000, featureValidationData), validationData['Response'])
print('\n Validating accuracy: %6.4f'%valAccWLambda1000)


# In[ ]:


f, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6), sharey=True)
plt.title("Accuracy at different regulation parameter")

x1 = np.array([0.001,0.01,0.1,1,10,100,1000])
y1 = np.array([trainAccWLambda0001, trainAccWLambda001, trainAccWLambda01, trainAccWLambda1, trainAccWLambda10, trainAccWLambda100, trainAccWLambda1000])
sns.barplot(x=x1, y=y1, ax=ax1)
ax1.set_ylabel("Accuracy with training data")

x2 = np.array([0.001,0.01,0.1,1,10,100,1000])
y2 = np.array([valAccWLambda0001, valAccWLambda001, valAccWLambda01, valAccWLambda1, valAccWLambda10, valAccWLambda100, valAccWLambda1000])
sns.barplot(x=x2, y=y2, ax=ax2)
ax2.set_ylabel("Accuracy with validating data")
plt.show()
plt.savefig("IA2P1A.png", format="png")
# Most important weight:

# In[ ]:


FinalWStar = dict(zip(featureData.columns.values, wLambda01))
dict(sorted(FinalWStar.items(), key=lambda item: np.absolute(item[1]), reverse = True))


# In[ ]:


FinalWDash = dict(zip(featureData.columns.values, wLambda0001))
dict(sorted(FinalWDash.items(), key=lambda item: np.absolute(item[1]), reverse = True))


# In[ ]:


FinalWPlus = dict(zip(featureData.columns.values, wLambda1))
dict(sorted(FinalWPlus.items(), key=lambda item: np.absolute(item[1]), reverse = True))


# Sparcity Report: (assuming near 0 weight is 0 at 1e-9)

# In[ ]:


plt.subplots(figsize=(12, 9))
plt.title("Sparcity at different regulization")
xaxis = np.array([0.001,0.01,0.1,1,10,100,1000])
sparcities = np.array([
    featuresNumber - np.count_nonzero(roundingW(wLambda0001)),
    featuresNumber - np.count_nonzero(roundingW(wLambda001)),
    featuresNumber - np.count_nonzero(roundingW(wLambda01)),
    featuresNumber - np.count_nonzero(roundingW(wLambda1)),
    featuresNumber - np.count_nonzero(roundingW(wLambda10)),
    featuresNumber - np.count_nonzero(roundingW(wLambda100)),
    featuresNumber - np.count_nonzero(roundingW(wLambda1000))
])
ax = sns.barplot(x = xaxis, y = sparcities)
ax.set(xlabel='Lambda', ylabel='Number of weight == 0')
plt.show()
plt.savefig("IA2P1C.png", format="png")


# ## Part 2

# In[ ]:


def BGDLogicL1(inputData, resultData, initVector, maxIter, minErr, alphaRate, lambdaReg):
    iteration = 0
    dataSize = len(inputData)
    w = initVector.copy()
    gradientVector = ((alphaRate/dataSize)*((inputData.mul(resultData - sigmoid(inputData.mul(w).sum(axis = 1)), axis = 0)).sum(axis = 0))).values
    normGradient = np.linalg.norm(gradientVector)
    while (normGradient > minErr and not np.isinf(normGradient) and iteration < maxIter):
        w = w + gradientVector
        for index, w_j in enumerate(w):
            if (index >= 1):
                if w_j >= 0:
                    w_j = max(np.absolute(w_j) - alphaRate*lambdaReg,0)
                else:
                    w_j = -max(np.absolute(w_j) - alphaRate*lambdaReg,0)
            w[index] = w_j
        gradientVector = ((alphaRate/dataSize)*((inputData.mul(resultData - sigmoid(inputData.mul(w).sum(axis = 1)), axis = 0)).sum(axis = 0))).values
        normGradient = np.linalg.norm(gradientVector)
        iteration += 1
        print('Gradient: %9.7f, Iteration: [%d/%d]\r'%(normGradient,iteration,MAX_ITER), end="")
    return w
## Batch gradient descent L1


# In[ ]:


wLambda0001L1 = BGDLogicL1(featureData, originalData['Response'], initialGuess, MAX_ITER, MIN_ERR, 0.1, 0.001)

trainAccWLambda0001L1 = getAccuracy(getPrediction(wLambda0001L1, featureData), originalData['Response'])
print('\n Training accuracy: %6.4f'%trainAccWLambda0001L1)

valAccWLambda0001L1 = getAccuracy(getPrediction(wLambda0001L1, featureValidationData), validationData['Response'])
print('\n Validating accuracy: %6.4f '%valAccWLambda0001L1)


# In[ ]:


wLambda001L1 = BGDLogicL1(featureData, originalData['Response'], initialGuess, MAX_ITER, MIN_ERR, 0.1, 0.01)

trainAccWLambda001L1 = getAccuracy(getPrediction(wLambda001L1, featureData), originalData['Response'])
print('\n Training accuracy: %6.4f'%trainAccWLambda001L1)

valAccWLambda001L1 = getAccuracy(getPrediction(wLambda001L1, featureValidationData), validationData['Response'])
print('\n Validating accuracy: %6.4f '%valAccWLambda001L1)


# In[ ]:


wLambda01L1 = BGDLogicL1(featureData, originalData['Response'], initialGuess, MAX_ITER, MIN_ERR, 0.1, 0.1)

trainAccWLambda01L1 = getAccuracy(getPrediction(wLambda01L1, featureData), originalData['Response'])
print('\n Training accuracy: %6.4f'%trainAccWLambda01L1)

valAccWLambda01L1 = getAccuracy(getPrediction(wLambda01L1, featureValidationData), validationData['Response'])
print('\n Validating accuracy: %6.4f '%valAccWLambda01L1)


# In[ ]:


wLambda1L1 = BGDLogicL1(featureData, originalData['Response'], initialGuess, MAX_ITER, MIN_ERR, 0.1, 1)

trainAccWLambda1L1 = getAccuracy(getPrediction(wLambda1L1, featureData), originalData['Response'])
print('\n Training accuracy: %6.4f'%trainAccWLambda1L1)

valAccWLambda1L1 = getAccuracy(getPrediction(wLambda1L1, featureValidationData), validationData['Response'])
print('\n Validating accuracy: %6.4f '%valAccWLambda1L1)


# In[ ]:


wLambda10L1 = BGDLogicL1(featureData, originalData['Response'], initialGuess, MAX_ITER, MIN_ERR, 0.1, 10)

trainAccWLambda10L1 = getAccuracy(getPrediction(wLambda10L1, featureData), originalData['Response'])
print('\n Training accuracy: %6.4f'%trainAccWLambda10L1)

valAccWLambda10L1 = getAccuracy(getPrediction(wLambda10L1, featureValidationData), validationData['Response'])
print('\n Validating accuracy: %6.4f '%valAccWLambda10L1)


# In[ ]:


wLambda100L1 = BGDLogicL1(featureData, originalData['Response'], initialGuess, MAX_ITER, MIN_ERR, 0.1, 100)

trainAccWLambda100L1 = getAccuracy(getPrediction(wLambda100L1, featureData), originalData['Response'])
print('\n Training accuracy: %6.4f'%trainAccWLambda100L1)

valAccWLambda100L1 = getAccuracy(getPrediction(wLambda100L1, featureValidationData), validationData['Response'])
print('\n Validating accuracy: %6.4f '%valAccWLambda100L1)


# In[ ]:


wLambda1000L1 = BGDLogicL1(featureData, originalData['Response'], initialGuess, MAX_ITER, MIN_ERR, 0.1, 1000)

trainAccWLambda1000L1 = getAccuracy(getPrediction(wLambda1000L1, featureData), originalData['Response'])
print('\n Training accuracy: %6.4f'%trainAccWLambda1000L1)

valAccWLambda1000L1 = getAccuracy(getPrediction(wLambda1000L1, featureValidationData), validationData['Response'])
print('\n Validating accuracy: %6.4f '%valAccWLambda1000L1)


# In[ ]:


f, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6), sharey=True)
plt.title("Accuracy at different regulation parameter")

x1 = np.array([0.001,0.01,0.1,1,10,100,1000])
y1 = np.array([trainAccWLambda0001L1, trainAccWLambda001L1, trainAccWLambda01L1, trainAccWLambda1L1, trainAccWLambda10L1, trainAccWLambda100L1, trainAccWLambda1000L1])
sns.barplot(x=x1, y=y1, ax=ax1)
ax1.set_ylabel("Accuracy with training data")

x2 = np.array([0.001,0.01,0.1,1,10,100,1000])
y2 = np.array([valAccWLambda0001L1, valAccWLambda001L1, valAccWLambda01L1, valAccWLambda1L1, valAccWLambda10L1, valAccWLambda100L1, valAccWLambda1000L1])
sns.barplot(x=x2, y=y2, ax=ax2)
ax2.set_ylabel("Accuracy with validating data")
plt.show()
plt.savefig("IA2P2A.png", format="png")


# Most important weight:

# In[ ]:


FinalWStarL1 = dict(zip(featureData.columns.values, wLambda01L1))
dict(sorted(FinalWStarL1.items(), key=lambda item: np.absolute(item[1]), reverse = True))


# In[ ]:


FinalWDashL1 = dict(zip(featureData.columns.values, wLambda001L1))
dict(sorted(FinalWDashL1.items(), key=lambda item: np.absolute(item[1]), reverse = True))


# In[ ]:


FinalWPlus1 = dict(zip(featureData.columns.values, wLambda1L1))
dict(sorted(FinalWPlus1.items(), key=lambda item: np.absolute(item[1]), reverse = True))


# Sparcity graph:

# In[ ]:


plt.subplots(figsize=(12, 9))
plt.title("Sparcity at different regulization")
xaxis = np.array([0.001,0.01,0.1,1,10,100,1000])
sparcities = np.array([
    featuresNumber - np.count_nonzero(roundingW(wLambda0001L1)),
    featuresNumber - np.count_nonzero(roundingW(wLambda001L1)),
    featuresNumber - np.count_nonzero(roundingW(wLambda01L1)),
    featuresNumber - np.count_nonzero(roundingW(wLambda1L1)),
    featuresNumber - np.count_nonzero(roundingW(wLambda10L1)),
    featuresNumber - np.count_nonzero(roundingW(wLambda100L1)),
    featuresNumber - np.count_nonzero(roundingW(wLambda1000L1))
])
ax = sns.barplot(x = xaxis, y = sparcities)
ax.set(xlabel='Lambda', ylabel='Number of weight == 0')
plt.show()
plt.savefig("IA2P2C.png", format="png")


# In[ ]:
