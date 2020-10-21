# %%
# import required libraries
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import sklearn
import joblib

# import necessary modules
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt

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
# set target column, load and normalize data
target_column = ['NoP']
predictors = list(set(list(df.columns)) - set(target_column))

# load scaler
scaler = joblib.load('scaler.save')
df[predictors] = scaler.fit_transform(df[predictors])
df.describe()

# %%
# load model
model = keras.models.load_model('model/prehension_v2')

# %%
test = df.loc[20100]
print(test['NoP'])
test.drop(['NoP'], inplace=True)
testinput = np.asarray(test).astype(np.float32)

# %%
probs = model.predict(np.array([testinput,]))
answer = probs.argmax(axis=-1)
print(probs, answer)
# %%
