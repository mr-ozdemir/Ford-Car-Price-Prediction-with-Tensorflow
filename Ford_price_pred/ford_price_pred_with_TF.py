import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sbn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import r2_score




firstDataFrame = pd.read_csv("ford.csv")

#########Data Analyses


firstDataFrame.isnull().sum()

print(firstDataFrame.describe())

firstDataFrame.corr()["price"].sort_values()

sbn.displot(firstDataFrame["price"])
plt.plot()


model_counts = firstDataFrame['model'].value_counts()

print(model_counts)

# Model adedi 100'den az olan verileri seçelim
models_with_less_than_100_count = model_counts[model_counts < 100]

# Seçilen verilerin toplam adedini hesaplayalım
total_count = models_with_less_than_100_count.sum()
print("****************")
print("Model adedi 100'den az olan verilerin toplam adedi:", total_count)
print(models_with_less_than_100_count)

print("\n\n1000 tl den az arac sayısı:  ",len(firstDataFrame[firstDataFrame['price'] < 1000]))
print("\n 25 bin  tl den fazla arac sayısı:  ",len(firstDataFrame[firstDataFrame['price'] > 25000]))

print(firstDataFrame.groupby("year").mean()["price"])

print("2000 yılından eski araçlar = ",len(firstDataFrame[firstDataFrame['year'] < 2000]))


plt.figure(figsize=(15, 15))
sbn.lineplot(data=firstDataFrame, x='model', y='price')
plt.xticks(rotation=45)
plt.title('Model and Price Relationship')
plt.plot()



plt.figure(figsize=(15, 10))
sbn.barplot(data=firstDataFrame, x='model', y='price')
plt.xticks(rotation=45)
plt.title('Model and Price Relationship')
plt.plot()



plt.figure(figsize=(12, 6))
sbn.barplot(x=model_counts.index, y=model_counts.values)
plt.xticks(rotation=45)
plt.xlabel('Model')
plt.ylabel('quantity')
plt.title('Model quantity')
plt.plot()


#Delete models with less than 100 vehicles
new_models = model_counts[model_counts > 100].index
cleanDataframe = firstDataFrame[firstDataFrame['model'].isin(new_models)]

#delete erroneous data
cleanDataframe = cleanDataframe[cleanDataframe["year"] < 2024] 
cleanDataframe = cleanDataframe[cleanDataframe["year"] > 2000] 
cleanDataframe = cleanDataframe[cleanDataframe['price'] <= 25000]
cleanDataframe = cleanDataframe[cleanDataframe['price'] > 1000]


print(cleanDataframe.describe())
print("\n\n1000 tl den az arac sayısı:  ",len(cleanDataframe[cleanDataframe['price'] < 1000]))
print("\n 25 bin  tl den fazla arac sayısı:  ",len(cleanDataframe[cleanDataframe['price'] > 25000]))
print("2000 yılından eski araçlar = ",len(cleanDataframe[cleanDataframe['year'] < 2000]))
print(cleanDataframe.groupby("year").mean()["price"])


sbn.displot(cleanDataframe["price"])
plt.plot()


model_counts2 = cleanDataframe['model'].value_counts()

plt.figure(figsize=(12, 6))
sbn.barplot(x=model_counts2.index, y=model_counts2.values)
plt.xticks(rotation=45)
plt.xlabel('Model')
plt.ylabel('quantity')
plt.title('Model quantity')
plt.plot()




##cleanDataframe = cleanDataframe.drop("tax",axis=1)



data_encoded = pd.get_dummies(cleanDataframe, columns=['transmission'])
data_encoded = pd.get_dummies(data_encoded, columns=['model'])
data_encoded = pd.get_dummies(data_encoded, columns=['fuelType'])




y = data_encoded["price"].values #dependent variable
x = data_encoded.drop("price",axis = 1).values #independent variable

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.33, random_state = 10 )

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.fit_transform(x_test)

print(x_train.shape)

model = Sequential()


model.add(Dense(128,activation="relu"))

model.add(Dense(64,activation="relu"))

model.add(Dense(32,activation="relu"))


model.add(Dense(1))

model.compile(optimizer = "adam",loss = "mse")

earlyStopping = EarlyStopping(monitor = "val_loss", mode = "min",verbose = 1, patience = 25)

model.fit(x = x_train, y = y_train,validation_data = (x_test, y_test),batch_size = 100, epochs = 300, callbacks = [earlyStopping])
 
lossDataFrame = pd.DataFrame(model.history.history)

lossDataFrame.plot()


predArray = model.predict(x_test)

print(mean_absolute_error(y_test,predArray))


print(firstDataFrame.iloc[486])

newCar = data_encoded.drop("price",axis=1).iloc[486]

newCar = scaler.transform(newCar.values.reshape(-1,26))

print(model.predict(newCar))

import statsmodels.api as sm

model2 = sm.OLS(y,x).fit()

print(model2.summary())






