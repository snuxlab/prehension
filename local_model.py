## local model demo trial with the titanic dataset on google.
## date: 2020/09/07
## author: minjoon

#%%
import pandas as pd
import numpy as np
from IPython.display import clear_output
from matplotlib import pyplot as plt

#%%
# Load dataset.
train_data = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/train.csv')
eval_data = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/eval.csv')
y_train = train_data.pop('survived')
y_eval = eval_data.pop('survived')

#%%
import tensorflow as tf
tf.random.set_seed(123)

#%%
train_data.head()

#%%
train_data.describe()

#%%
train_data.shape[0], eval_data.shape[0]

# %%
train_data.age.hist(bins=20)
plt.show()

# %%
train_data.sex.value_counts().plot(kind='barh')
plt.show()

# %%
train_data['class'].value_counts().plot(kind='barh')
plt.show()

# %%
train_data['embark_town'].value_counts().plot(kind='barh')
plt.show()

# %%
pd.concat([train_data, y_train], axis=1).groupby('sex').survived.mean().plot(kind='barh').set_xlabel('% survive')
plt.show()

# %%
CATEGORICAL_COLUMNS = ['sex', 'n_siblings_spouses', 'parch', 'class', 'deck', 'embark_town', 'alone']
NUMERIC_COLUMNS = ['age', 'fare']


def one_hot_cat_column(feature_name, vocab):
  return tf.feature_column.indicator_column(
    tf.feature_column.categorical_column_with_vocabulary_list(feature_name, vocab))


feature_columns = []
for feature_name in CATEGORICAL_COLUMNS:
  # Need to one-hot encode categorical features.
  vocabulary = train_data[feature_name].unique()
  feature_columns.append(one_hot_cat_column(feature_name, vocabulary))

for feature_name in NUMERIC_COLUMNS:
  feature_columns.append(tf.feature_column.numeric_column(feature_name, dtype=tf.float32))


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
# Train a linear classifier (logistic regression model) to establish a benchmark.
feature_columns
linear_est = tf.estimator.LinearClassifier(feature_columns)

# Train model.
linear_est.train(train_input_fn, max_steps=100)

# Evaluation.
result = linear_est.evaluate(eval_input_fn)
clear_output()
print(pd.Series(result))

# %%
# Since data fits into memory, use entire dataset per layer. It will be faster.
# Above one batch is defined as the entire dataset.
n_batches = 1
est = tf.estimator.BoostedTreesClassifier(feature_columns, n_batches_per_layer=n_batches)

# The model will stop training once the specified number of trees is built, not
# based on the number of steps.
est.train(train_input_fn, max_steps=100)

# Eval.
result = est.evaluate(eval_input_fn)
clear_output()
print(pd.Series(result))

#%%
# predictions from the trained model
pred_dicts = list(est.predict(eval_input_fn))
probs = pd.Series([pred['probabilities'][1] for pred in pred_dicts])

probs.plot(kind='hist', bins=20, title='predicted probabilities')
plt.show()

#%%
from sklearn.metrics import roc_curve

fpr, tpr, _ = roc_curve(y_eval, probs)
plt.plot(fpr, tpr)
plt.title('ROC curve')
plt.xlabel('false positive rate')
plt.ylabel('true positive rate')
plt.xlim(0,)
plt.ylim(0,)
plt.show()

