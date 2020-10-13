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
# set target column, and normalize data
target_column = ['NoP']
predictors = list(set(list(df.columns)) - set(target_column))
df[predictors] = df[predictors]/df[predictors].max()
df.describe()

# %%
# load model
model = keras.models.load_model('prehension_v1_e50')

# %%
test = df.loc[3561]
test.drop(['NoP'], inplace=True)
testinput = np.asarray(test).astype(np.float32)

# %%
probs = model.predict(np.array([testinput,]))
answer = probs.argmax(axis=-1)
# %%
