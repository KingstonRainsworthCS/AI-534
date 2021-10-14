import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

path_to_csv = "./pa0(train-only).csv"

print(f'Reading data from {path_to_csv}')
rawDF = pd.read_csv(path_to_csv)
print("Viewing head: ")
print(rawDF.head())
# read and view head of csv file for quick check

print("-----------------------------------------------------------------------")
print("Question a")
print("Remove id from dataframe: ")
formattedDF = rawDF.drop(['id'], axis = 1)
print(formattedDF.head())

print("-----------------------------------------------------------------------")
print("Question b")
print("splitting date to their columns: ")
dateformat = formattedDF["date"].str.split("/", expand = True)
formattedDF["year"] = dateformat[2]
formattedDF["month"] = dateformat[0]
formattedDF["day"] = dateformat[1]
finalDF = formattedDF.drop(['date'], axis = 1)
print(finalDF.head())

print("-----------------------------------------------------------------------")
print("Question c")
plt.subplots(figsize=(15, 12))
plt.title("Price variation for each bedrooms type")
sns.boxplot(x="bedrooms", y="price", data=formattedDF, flierprops = dict(markerfacecolor = '0.50', markersize = 1))
plt.savefig("Q3P1.png", format="png")

plt.subplots(figsize=(15, 12))
plt.title("Price variation for each bathrooms type")
sns.boxplot(x="bathrooms", y="price", data=formattedDF, flierprops = dict(markerfacecolor = '0.50', markersize = 1))
plt.savefig("Q3P2.png", format="png")

plt.subplots(figsize=(15, 12))
plt.title("Price variation for each floors type")
sns.boxplot(x="floors", y="price", data=formattedDF, flierprops = dict(markerfacecolor = '0.50', markersize = 1))
plt.savefig("Q3P3.png", format="png")

print("-----------------------------------------------------------------------")
print("Question d")
covariance_matrix = finalDF[['sqft_living','sqft_lot','sqft_living15','sqft_lot15']].cov()
print('Covariance matrix for sqft_living, sqft_lot, sqft_living15 and sqft_lot15: ')
print(covariance_matrix)

plt.title("sqft_living15 vs sqft_living")
sns.relplot(x="sqft_living", y="sqft_living15", data=formattedDF)
plt.savefig("Q4P1.png", format="png")

plt.title("sqft_lot15 vs sqft_lot")
sns.relplot(x="sqft_lot", y="sqft_lot15", data=formattedDF)
plt.savefig("Q4P2.png", format="png")
