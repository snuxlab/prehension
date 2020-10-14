## local model trial -- version 0.2 with Keras
## created: 2020/10/13
## author: minjoon
## last edited: 2020/10/13

#%%
# import required libraries
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import sklearn

# import necessary modules
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from math import sqrt 
from scipy.stats import zscore
import joblib

# keras specific
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# %%
# Load dataset.
df = pd.read_csv('data/sensor_sampled_ma-2020-08-25T07-49-22.611Z.csv')
df.head()

# %%
# drop unused columns in dataframe
df.drop(columns=['Date', 'Time', 'Cam'], inplace=True)
df.columns

# %%
# need to change 'n' into 2 and make categories as 'int' type for classification
df['NoP'].replace(['n'], 2, inplace=True)

# %%
# set target column
target_column = ['NoP']
predictors = list(set(list(df.columns)) - set(target_column))

# scaler = StandardScaler()
# df[predictors] = scaler.fit_transform(df[predictors])

# df[predictors] = df[predictors]/df[predictors].max()
# df.describe()
# %%
# create training and testing datasets, normalize data
X = df[predictors].values
y = df[target_column].values

scaler = MinMaxScaler()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=43)

X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)

print(X_train.shape)
print(X_test.shape)

# %%
# one-hot encode outputs
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

count_classes = y_test.shape[1]
print(count_classes)

# %%
model = Sequential()
model.add(Dense(500, activation='relu', input_dim=10))
model.add(Dense(100, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(3, activation='softmax'))

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# %%
# fit the model
# build the model
model.fit(X_train, y_train, batch_size=32,epochs=100)

# %%
# perform prediction on test data and
# compute evaluation metrics
pred_train = model.predict(X_train)
scores = model.evaluate(X_train, y_train, verbose=0)
print('Accuracy on training data: {}% \n Error on training data: {}'.format(
    scores[1], 1 - scores[1]))

pred_test = model.predict(X_test)
scores2 = model.evaluate(X_test, y_test, verbose=0)
print('Accuracy on test data: {}% \n Error on test data: {}'.format(
    scores2[1], 1 - scores2[1]))

# %%
test = df.loc[4]
print(test['NoP'])

test.drop(['NoP'], inplace=True)
testinput = np.asarray(test).astype(np.float32)

# %%
probs = model.predict(np.array([testinput, ]))
answer = probs.argmax(axis=-1)

print(answer, probs)

# %%
# save model and scaler
model.save('model/prehension_v1')

scaler_filename = 'model/scaler.save'
joblib.dump(scaler, scaler_filename)

# %%
