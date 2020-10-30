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
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt

# %%
# Load dataset
df = pd.read_csv('data/test1.csv')
df.head()

# %%
# drop unused columns in dataframe, change 'n' to 2
df.drop(columns=['Date', 'Time'], inplace=True)
df['NoP'].replace(['n'], 2, inplace=True)
df.head()

# %%
# prepare test input
testprep = df.drop(columns=['NoP', 'Hour'])
testprep.head()

# %%
# one-hot encode hours (numerical category)
hour_oh = to_categorical(df['Hour'])
predictors = testprep.columns.tolist()

# %%
# load model, scaler & create test input
scaler = joblib.load('model/scaler3.save')
model =keras.models.load_model('model/prehension_v3')

testinput = np.concatenate((df[predictors].values, hour_oh), axis=1)
testinput = scaler.transform(testinput)

# %%
# run by model, check total accuracy
answers = to_categorical(df['NoP'].values, 3)
pred_test = model.predict(testinput)
scores2 = model.evaluate(testinput, answers, verbose=0)
print('Accuracy on test data: {}% \n Error on test data: {}'.format(
    scores2[1], 1 - scores2[1]))

# %%
# labeled vs. predicted comparison
result_class = pred_test.argmax(axis=-1)

checkans = pd.DataFrame()
checkans['Labeled'] = df['NoP']
checkans['Predicted'] = result_class
checkans['Hour'] = df['Hour']
checkans['Time'] = df['Time']

checkans.to_csv('data/test1-results.csv')

# %%
