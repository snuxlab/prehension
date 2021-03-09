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
from tensorflow.keras.utils import to_categorical

# %%
# Load dataset.
df = pd.read_csv('data/test1.csv')
df.head()

# %%
# drop unused columns in dataframe
time = df['Time']
df.drop(columns=['Date', 'Time', 'Hour'], inplace=True)
df.columns

# %%
# need to change 'n' into 2 and make categories as 'int' type for classification
df['NoP'].replace(['n'], 2, inplace=True)

# %%
# set target column, load and normalize data
target_column = ['NoP']
predictors = list(set(list(df.columns)) - set(target_column))

# load scaler
scaler = joblib.load('model/scaler.save')
df[predictors] = scaler.fit_transform(df[predictors])
df.describe()

# %%
# load model
model = keras.models.load_model('model/prehension_v1')

# %%
test = df[predictors]
# print(test['NoP'])
# test.drop(['NoP'], inplace=True)
testinput = np.asarray(test).astype(np.float32)

# %%
answers = to_categorical(df['NoP'].values, 3)
pred_test = model.predict(testinput)
scores2 = model.evaluate(testinput, answers, verbose=0)
print('Accuracy on test data: {}% \n Error on test data: {}'.format(
    scores2[1], 1 - scores2[1]))

# %%
# labeled vs. predicted comparison
result_class = pred_test.argmax(axis=-1)

checkans = pd.DataFrame()
checkans['Time'] = time
checkans['Labeled'] = df['NoP']
checkans['Predicted'] = result_class

checkans.to_csv('data/test1-v1results.csv')

# %%
