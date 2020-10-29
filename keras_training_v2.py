## local model trial -- version 2.0 with Keras
## created: 2020/10/28
## author: minjoon
## last edited: 2020/10/28

# %%
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
# import data
df = pd.read_csv('data/sensor_sampled_ma-2020-08-25T07-49-22.611Z.csv')

# %%
# add Hour column (numerical category) to dataframe &
# change 'n' to 2 (numerical category) &
# drop unused columns 

df['Hour'] = pd.to_datetime(
    df['Time'], format='%H:%M:%S').dt.hour
df['NoP'].replace(['n'], 2, inplace=True)
df.drop(columns=['Date', 'Time', 'Cam'], inplace=True)

df.head()
# %%
# divide target and numerical predictors
predictors = df.columns.tolist()
predictors.remove('NoP')
predictors.remove('Hour')

# %%
# one-hot encode hours (numerical category)
hour_oh = to_categorical(df['Hour'])

# %%
# one-hot encode hour data, add to predictors
X = df[predictors].values
y = df['NoP'].values
X = np.concatenate((X, hour_oh), axis=1)

# %%
# train-test split of data

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=1337)

# %%
# scale training set
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print(X_train.shape)
print(X_test.shape)

# %%
# one-hot encode outputs
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

count_classes = y_test.shape[1]
print(count_classes)

# %%
# set up model
model = Sequential()
model.add(Dense(500, activation='relu', input_dim=34))
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
model.fit(X_train, y_train, batch_size=32, epochs=100)

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
test = df.loc[10100]
print(test['NoP'], test['Hour'])

test.drop(['NoP'], inplace=True)
testinput = np.asarray(test[:-1]).astype(np.float32)
testinput = np.concatenate((testinput, (to_categorical(test['Hour'],24))))

# %%
input_scaled = scaler.transform(testinput.reshape(1, -1))

input_scaled = tf.convert_to_tensor(input_scaled)

pred = model.predict(input_scaled)
result_class = pred.argmax(axis=-1)[0]
result_score = round(pred[0][result_class] * 100, 2)

print(result_class, result_score)

# %%
model.save('model/prehension_v3')

scaler_filename = 'model/scaler3.save'
joblib.dump(scaler, scaler_filename)

# %%
