import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

path_to_train_csv = "./IA1_train.csv"
path_to_dev_csv = "./IA1_dev.csv"

originalData = pd.read_csv(path_to_train_csv)
testData = pd.read_csv(path_to_dev_csv)

## Data processing

dateformat = originalData["date"].str.split("/", expand = True)
originalData['year'] = dateformat[2].astype(int)
originalData['month'] = dateformat[1].astype(int)
originalData['day'] = dateformat[0].astype(int)
originalData['W0'] = 1
originalData['age_since_renovated'] = np.where(originalData['yr_renovated'] == 0, originalData['year'] - originalData['yr_built'], originalData['year'] - originalData['yr_renovated'])

normalizeData = (originalData - originalData.mean()) / originalData.std()
normalizeData['W0'] = originalData['W0']
normalizeData['waterfront'] = originalData['waterfront']
normalizeData['price'] = originalData['price']

trainingData = normalizeData.drop(['date', 'id', 'yr_renovated'], axis = 1)
trainingData.head()

## Part 1
initial_guess = np.repeat([0.5],22)
featureData = trainingData.drop(['price'], axis = 1)

def BatchGradientDescent(featureDataframe, actualValue, initialGuess, maxSteps, minErr, learningRate):
    iteration = 0
    dfSize = len(featureDataframe)
    nextW = initialGuess
    gradientLw = (2/dfSize)*featureDataframe.mul((featureDataframe.mul(initialGuess).sum(axis=1) - actualValue), axis = 0).sum().values
    normGradient = np.linalg.norm(gradientLw)
    MeanSquareError = (1/dfSize)*((featureDataframe.mul(nextW).sum(axis=1) - actualValue)**2).sum()
    MSEs = [MeanSquareError]
    while (normGradient > minErr and not np.isinf(normGradient) and iteration < maxSteps):
        nextW = nextW - learningRate*gradientLw
        gradientLw = (2/dfSize)*featureDataframe.mul((featureDataframe.mul(nextW).sum(axis=1) - actualValue), axis = 0).sum().values
        normGradient = np.linalg.norm(gradientLw)
        MeanSquareError = (1/dfSize)*((featureDataframe.mul(nextW).sum(axis=1) - actualValue)**2).sum()
        MSEs.append(MeanSquareError)
        iteration += 1
    return {'w': dict(zip(featureDataframe.columns.values, nextW)), 'gradientLw': normGradient, 'n': iteration, 'MSE': MSEs}

## Result:
Rate0 = BatchGradientDescent(featureData, trainingData['price'], initial_guess, 5000, 0.05, 10)
plt.subplots(figsize=(15, 12))
plt.title("MSE vs Iteration")
ax = sns.lineplot(x = list(range(0,Rate0['n']+1)), y = Rate0['MSE'])
ax.set(xlabel='n', ylabel='MSE')
plt.savefig("Part1Rate10.png", format="png")

Rate1 = BatchGradientDescent(featureData, trainingData['price'], initial_guess, 5000, 0.05, 0.1)
Rate2 = BatchGradientDescent(featureData, trainingData['price'], initial_guess, 5000, 0.05, 0.01)
Rate3 = BatchGradientDescent(featureData, trainingData['price'], initial_guess, 5000, 0.05, 0.001)
Rate4 = BatchGradientDescent(featureData, trainingData['price'], initial_guess, 5000, 0.05, 0.0001)
Rate5 = BatchGradientDescent(featureData, trainingData['price'], initial_guess, 5000, 0.05, 0.00001)
Rate6 = BatchGradientDescent(featureData, trainingData['price'], initial_guess, 5000, 0.05, 0.000001)

plt.subplots(figsize=(12, 9))
plt.title("MSE vs Iteration")
ax1 = sns.lineplot(x = list(range(0,Rate1['n']+1)), y = Rate1['MSE'], label='lambda = 0.1')
ax2 = sns.lineplot(x = list(range(0,Rate2['n']+1)), y = Rate2['MSE'], label='lambda = 0.01')
ax3 = sns.lineplot(x = list(range(0,Rate3['n']+1)), y = Rate3['MSE'], label='lambda = 0.001')
ax4 = sns.lineplot(x = list(range(0,Rate4['n']+1)), y = Rate4['MSE'], label='lambda = 0.0001')
ax5 = sns.lineplot(x = list(range(0,Rate5['n']+1)), y = Rate5['MSE'], label='lambda = 0.00001')
ax6 = sns.lineplot(x = list(range(0,Rate6['n']+1)), y = Rate6['MSE'], label='lambda = 0.000001')
ax6.set(xlabel='n', ylabel='MSE')
ax6.legend()
plt.savefig("Part1RateRest.png", format="png")

testdateformat = testData["date"].str.split("/", expand = True)
testData['year'] = testdateformat[2].astype(int)
testData['month'] = testdateformat[1].astype(int)
testData['day'] = testdateformat[0].astype(int)
testData['W0'] = 1
testData['age_since_renovated'] = np.where(testData['yr_renovated'] == 0, testData['year'] - testData['yr_built'], testData['year'] - testData['yr_renovated'])

normalizetestData = (testData - originalData.mean()) / originalData.std()
normalizetestData['W0'] = testData['W0']
normalizetestData['waterfront'] = testData['waterfront']
normalizetestData['price'] = testData['price']

validatingData = normalizetestData.drop(['date', 'id', 'yr_renovated'], axis = 1)
validatingFeatureData = validatingData.drop(['price'], axis = 1)

Rate1MSE = (1/len(validatingData))*((validatingFeatureData.mul(list(Rate1['w'].values())).sum(axis=1) - validatingData['price'])**2).sum()
print(f'MSE for learning rate = 0.1: {Rate1MSE}')

Rate2MSE = (1/len(validatingData))*((validatingFeatureData.mul(list(Rate2['w'].values())).sum(axis=1) - validatingData['price'])**2).sum()
print(f'MSE for learning rate = 0.01: {Rate2MSE}')

print(f'Learning rate = 0.1 final W: \n{Rate1["w"]} \n achieved at iteration: {Rate1["n"]}')

## Part 2:
anomalousTrainingData = originalData.drop(['date', 'id', 'yr_renovated'], axis = 1)
anomalousFeatureData = anomalousTrainingData.drop(['price'], axis = 1)

anomalousRate0 = BatchGradientDescent(anomalousFeatureData, anomalousTrainingData['price'], initial_guess, 5000, 0.05, 10)
anomalousRate1 = BatchGradientDescent(anomalousFeatureData, anomalousTrainingData['price'], initial_guess, 5000, 0.05, 0.1)
anomalousRate2 = BatchGradientDescent(anomalousFeatureData, anomalousTrainingData['price'], initial_guess, 5000, 0.05, 0.01)
anomalousRate3 = BatchGradientDescent(anomalousFeatureData, anomalousTrainingData['price'], initial_guess, 5000, 0.05, 0.001)
anomalousRate4 = BatchGradientDescent(anomalousFeatureData, anomalousTrainingData['price'], initial_guess, 5000, 0.05, 0.0001)
anomalousRate5 = BatchGradientDescent(anomalousFeatureData, anomalousTrainingData['price'], initial_guess, 5000, 0.05, 0.00001)
anomalousRate6 = BatchGradientDescent(anomalousFeatureData, anomalousTrainingData['price'], initial_guess, 5000, 0.05, 0.000001)

plt.subplots(figsize=(12, 9))
plt.title("MSE vs Iteration")
ax1 = sns.lineplot(x = list(range(0,anomalousRate0['n']+1)), y = np.log10(anomalousRate0['MSE']), label='lambda = 10')
ax1 = sns.lineplot(x = list(range(0,anomalousRate1['n']+1)), y = np.log10(anomalousRate1['MSE']), label='lambda = 0.1')
ax2 = sns.lineplot(x = list(range(0,anomalousRate2['n']+1)), y = np.log10(anomalousRate2['MSE']), label='lambda = 0.01')
ax3 = sns.lineplot(x = list(range(0,anomalousRate3['n']+1)), y = np.log10(anomalousRate3['MSE']), label='lambda = 0.001')
ax4 = sns.lineplot(x = list(range(0,anomalousRate4['n']+1)), y = np.log10(anomalousRate4['MSE']), label='lambda = 0.0001')
ax5 = sns.lineplot(x = list(range(0,anomalousRate5['n']+1)), y = np.log10(anomalousRate5['MSE']), label='lambda = 0.00001')
ax6 = sns.lineplot(x = list(range(0,anomalousRate6['n']+1)), y = np.log10(anomalousRate6['MSE']), label='lambda = 0.000001')

ax6.set(xlabel='n', ylabel='MSE log10')
ax6.legend()
plt.savefig("Part2RateEvery.png", format="png")

## Part 3
leanTrainingData = normalizetestData.drop(['date', 'id', 'yr_renovated', 'sqft_living15'], axis = 1)
leanFeatureData = leanTrainingData.drop(['price'], axis = 1)
originalGuess = Rate1['w'].copy()
originalGuess.pop('sqft_living15')
originalGuess = list(originalGuess.values())
leanRate1 = BatchGradientDescent(leanFeatureData, leanTrainingData['price'], originalGuess, 5000, 0.05, 0.1)

leanValidatingData = validatingData.drop(['sqft_living15'], axis = 1)
leanValidatingFeatureData = leanValidatingData.drop(['price'], axis = 1)

finalMSE = (1/len(leanValidatingData))*((leanValidatingFeatureData.mul(list(leanRate1['w'].values())).sum(axis=1) - leanValidatingData['price'])**2).sum()
print(f'MSE for learning rate = 0.1: {finalMSE}')

print(f'Final w for learning rate = 0.1: \n {leanRate1["w"]}')

## Part 4
p4_path_to_train_csv = "./PA1_train1.csv"
p4_path_to_dev_csv = "./PA1_test1.csv"

p4originalData = pd.read_csv(p4_path_to_train_csv)
p4testData = pd.read_csv(p4_path_to_train_csv)

## Drop low weight value (< 0.1) base on result of part 3
dateformat = p4originalData["date"].str.split("/", expand = True)
p4originalData['year'] = dateformat[2].astype(int)
p4originalData['W0'] = 1
p4originalData['age_since_renovated'] = np.where(p4originalData['yr_renovated'] == 0, p4originalData['year'] - p4originalData['yr_built'], p4originalData['year'] - p4originalData['yr_renovated'])

p4normalizeData = (p4originalData - p4originalData.mean()) / p4originalData.std()
p4normalizeData['W0'] = p4originalData['W0']
p4normalizeData['waterfront'] = p4originalData['waterfront']
p4normalizeData['price'] = p4originalData['price']

p4trainingData = p4normalizeData.drop(['date', 'id', 'yr_renovated', 'floors', 'sqft_lot', 'sqft_living15'], axis = 1)
p4trainingData.head()

initial_guess = np.repeat([0.5],17)
p4featureData = p4trainingData.drop(['price'], axis = 1)

p4originalGuess = Rate1['w'].copy()
for key in ['floors', 'sqft_lot', 'sqft_living15', 'month', 'day']:
    p4originalGuess.pop(key)
p4originalGuess = list(p4originalGuess.values())
p4Rate2 = BatchGradientDescent(p4featureData, p4trainingData['price'], p4originalGuess, 5000, 0.05, 0.01)

p4testdateformat = p4testData["date"].str.split("/", expand = True)
p4testData['year'] = p4testdateformat[2].astype(int)
p4testData['W0'] = 1
p4testData['price'] = 0
p4testData['age_since_renovated'] = np.where(p4testData['yr_renovated'] == 0, p4testData['year'] - p4testData['yr_built'], p4testData['year'] - p4testData['yr_renovated'])


p4normalizetestData = (p4testData - p4originalData.mean()) / p4originalData.std()
p4normalizetestData['W0'] = p4testData['W0']
p4normalizetestData['waterfront'] = p4testData['waterfront']
p4normalizetestData['price'] = p4testData['price']

p4validatingData = p4normalizetestData.drop(['date', 'id', 'yr_renovated', 'floors', 'sqft_lot', 'sqft_living15', 'price'], axis = 1)
p4validatingData.head()

resultDF = pd.DataFrame({ 'id': p4testData['id']})
resultDF['price'] = p4validatingData.mul(list(p4Rate2['w'].values())).sum(axis = 1)

resultDF.to_csv('Part4_Attempt1.csv')

## Further drop lat and long due to duplicated meaning with zip

p4trainingData = p4normalizeData.drop(['date', 'id', 'yr_renovated', 'floors', 'sqft_lot', 'sqft_living15', 'lat', 'long'], axis = 1)
p4trainingData.head()

p4featureData = p4trainingData.drop(['price'], axis = 1)

p4originalGuess = Rate1['w'].copy()
for key in ['floors', 'sqft_lot', 'sqft_living15', 'month', 'day', 'lat', 'long']:
    p4originalGuess.pop(key)
p4originalGuess = list(p4originalGuess.values())
p4Rate2 = BatchGradientDescent(p4featureData, p4trainingData['price'], p4originalGuess, 5000, 0.05, 0.01)

p4validatingData = p4normalizetestData.drop(['date', 'id', 'yr_renovated', 'floors', 'sqft_lot', 'sqft_living15', 'price', 'lat', 'long'], axis = 1)
p4validatingData.head()

resultDF = pd.DataFrame({ 'id': p4testData['id']})
resultDF['price'] = p4validatingData.mul(list(p4Rate2['w'].values())).sum(axis = 1)
resultDF.to_csv('Part4_Attempt2.csv', index=False)

## Further dropping zip instead

p4trainingData = p4normalizeData.drop(['date', 'id', 'yr_renovated', 'floors', 'sqft_lot', 'sqft_living15', 'zipcode'], axis = 1)
p4trainingData.head()
p4featureData = p4trainingData.drop(['price'], axis = 1)

p4originalGuess = Rate1['w'].copy()
for key in ['floors', 'sqft_lot', 'sqft_living15', 'month', 'day', 'zipcode']:
    p4originalGuess.pop(key)
p4originalGuess = list(p4originalGuess.values())
p4Rate2 = BatchGradientDescent(p4featureData, p4trainingData['price'], p4originalGuess, 5000, 0.05, 0.01)

p4validatingData = p4normalizetestData.drop(['date', 'id', 'yr_renovated', 'floors', 'sqft_lot', 'sqft_living15', 'price', 'zipcode'], axis = 1)
p4validatingData.head()
resultDF = pd.DataFrame({ 'id': p4testData['id']})
resultDF['price'] = p4validatingData.mul(list(p4Rate2['w'].values())).sum(axis = 1)

resultDF.to_csv('Part4_Attempt3.csv', index=False)
