## local model trial -- version 0.1
## created: 2020/09/15
## author: minjoon
## last edited: 2020/10/13

#%%
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras

from numpy.random import RandomState
from IPython.display import clear_output
from matplotlib import pyplot as plt

#%%
# Load dataset.
df = pd.read_csv('data/sensor_sampled_ma-2020-08-25T07-49-22.611Z.csv')
df.head()

#%%
# drop unused columns in dataframe
df.drop(columns=['Date', 'Time', 'Cam'], inplace=True)
df.columns

#%%
# Split the dataset to training(70%) and test(30%)
rng = RandomState()

train_data = df.sample(frac=0.7, random_state=rng)
eval_data = df.loc[~df.index.isin(train_data.index)]

# need to change 'n' into 3 and make categories as 'int' type for classification
y_train = train_data.pop('NoP').replace('n', '2').astype(int)
y_eval = eval_data.pop('NoP').replace('n', '2').astype(int)

# y_train = pd.to_numeric(y_train)
# y_eval = pd.to_numeric(y_eval)

# reset index of new dataframes
train_data.reset_index(inplace=True)
train_data.drop('index', axis=1, inplace=True)
eval_data.reset_index(inplace=True)
eval_data.drop('index', axis=1, inplace=True)


#%%
# create numerical columns for model training
tf.random.set_seed(123)

NUMERIC_COLUMNS = train_data.columns.tolist()
feature_columns = []

for feature_name in NUMERIC_COLUMNS:
  feature_columns.append(tf.feature_column.numeric_column(feature_name, dtype=tf.float64))


# %%
# Use entire batch since this is such a small dataset.
NUM_EXAMPLES = len(y_train)

def make_input_fn(X, y, n_epochs=None, shuffle=True):
  def input_fn():
    dataset = tf.data.Dataset.from_tensor_slices((dict(X), y))
    if shuffle:
      dataset = dataset.shuffle(NUM_EXAMPLES)
    # For training, cycle thru dataset as many times as need (n_epochs=None).
    dataset = dataset.repeat(n_epochs)
    # In memory training doesn't use batching.
    dataset = dataset.batch(NUM_EXAMPLES)
    return dataset
  return input_fn

# Training and evaluation input functions.
train_input_fn = make_input_fn(train_data, y_train)
eval_input_fn = make_input_fn(eval_data, y_eval, shuffle=False, n_epochs=1)

#%%
# inputs = {'MSE': 132.09,
#           'SSIM': 0.03,
#           'Sound': 12.32525914,
#           'Radar': 66,
#           'PIR': 1,
#           'MSE_MA': 45.97,
#           'SSIM_MA': 0.02,
#           'Sound_MA': 12.38,
#           'Radar_MA': 62,
#           'PIR_MA': 0.25
#           }

pred_input_fn = make_input_fn(eval_data[888:889], None, shuffle=False, n_epochs=1)

#%%
# Train a linear classifier (logistic regression model) to establish a benchmark.
linear_est = tf.estimator.LinearClassifier(feature_columns, n_classes=3)

# Train model.
linear_est.train(train_input_fn, max_steps=100)

# Evaluation.
result = linear_est.evaluate(eval_input_fn)
clear_output()
print(pd.Series(result))

#%%
# train a boosted tree classifier
# Since data fits into memory, use entire dataset per layer. It will be faster.
# Above one batch is defined as the entire dataset.
n_batches = 1
est = tf.estimator.BoostedTreesClassifier(feature_columns, n_batches_per_layer=n_batches, n_classes=3, model_dir='./model')

# The model will stop training once the specified number of trees is built, not
# based on the number of steps.
est.train(train_input_fn, max_steps=300)


#%%
# Eval.
result = est.evaluate(eval_input_fn)
clear_output()
print(pd.Series(result))

#%% save model section
# tf.saved_model.save(est, './saved_model')
est.summary()


#%%
pred_test = list(est.predict(pred_input_fn))

#%%
# predictions from the trained model
pred_dicts = list(est.predict(eval_input_fn))

# predicted classes & their probabilities
# answer = pred_dicts[0]['class_ids'][0]
probs = pd.Series([pred['probabilities'][pred['class_ids'][0]] for pred in pred_dicts])

probs.plot(kind='hist', bins=20, title='predicted probabilities')
plt.show()

#%%
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

#%%
# calculate ROC for all classes
from sklearn.preprocessing import label_binarize

## the answer set
y_true = label_binarize(y_eval, classes=[0, 1, 2])

## predicted set
answers = pd.Series([pred['class_ids'][0] for pred in pred_dicts])
y_pred = label_binarize(answers, classes=[0, 1, 2])

## prediction scores
y_score = pd.Series([pred['probabilities'].tolist() for pred in pred_dicts])

#%%
# create arrays for each cl
y0 = []
y1 = []
y2 = []

for score in y_score:
  y0.append(score[0])
  y1.append(score[1])
  y2.append(score[2])

y0 = np.array(y0)
y1 = np.array(y1)
y2 = np.array(y2)


#%%
print('Overall Score: ', roc_auc_score(y_true, y_pred), '\n')
print('Class 0 Score: ', roc_auc_score(y_true[:, 0], y0))
print('Class 1 Score: ', roc_auc_score(y_true[:, 1], y1))
print('Class 2 Score: ', roc_auc_score(y_true[:, 2], y2))

#%%

fpr, tpr, _ = roc_curve(y_true[:, 0], y0)
plt.plot(fpr, tpr)
plt.title('ROC curve')
plt.xlabel('false positive rate')
plt.ylabel('true positive rate')
plt.xlim(0,)
plt.ylim(0,)
plt.show()


# %%
