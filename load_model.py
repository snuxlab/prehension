#%%
# import libraries
import pandas as pd
import numpy as np
import tensorflow as tf

#%%
# input function for predictions
# make sure to use this as an input on predict()
def make_input_fn(X, y, n_epochs=1):
    def input_fn():
        dataset = tf.data.Dataset.from_tensor_slices((dict(X), y))
        return dataset
    return input_fn


#%%
# load model from directory
checkpoint_dir = './model/'

ckpt = tf.train.Checkpoint()
ckpt.restore(tf.train.latest_checkpoint(checkpoint_dir))

#%%
# Load dataset for input testing
df = pd.read_csv('data/sensor_sampled_ma-2020-08-25T07-49-22.611Z.csv')
df.drop(columns=['Date', 'Time', 'Cam', 'NoP'], inplace=True)

df

#%%
pred_input_fn = make_input_fn(df[888:889], None)
pred_test = list(ckpt.predict(pred_input_fn))

# %%
