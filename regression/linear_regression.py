import json
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import tensorflow as tf

from IPython import display
from sklearn import metrics
from tensorflow.python.framework import dtypes
from scipy import stats

def get_train_test_np_arrays():
  # Load the preprocessed training data set 
  train_set_input_path = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    'preprocessed_data_sets',
    'train.npz')
  preprocessed = np.load(train_set_input_path)
  X_train = preprocessed['X']
  Y_train = preprocessed['Y']

  # Load the preprocessed testing data set 
  test_set_input_path = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    'preprocessed_data_sets',
    'test.npz')
  preprocessed = np.load(test_set_input_path)
  X_test = preprocessed['X']
  Y_test = preprocessed['Y']

  return X_train, Y_train, X_test, Y_test

### Learn about normalizing the input data.
def feature_normalize(dataset):
  mu = np.mean(dataset, axis=0)
  sigma = np.std(dataset, axis=0)
  return (dataset - mu) / sigma

### Learn about reshaping the input arrays.
def append_bias_reshape(features, labels):
  n_training_samples = features.shape[0] # number of rows
  n_dim = features.shape[1] # number of columns
  features_reshaped = np.reshape(np.c_[np.ones(n_training_samples), features], 
                                 [n_training_samples, n_dim + 1])
  labels_reshaped = np.reshape(labels, [n_training_samples, 1])

  return features_reshaped, labels_reshaped

def tf_linear_regression(X_train, Y_train, X_test, Y_test):
  learning_rate = 0.01
  training_epochs = 100
  cost_history = np.empty(shape=[1], dtype=float)

  ### Learn that the placeholder is used as a placeholder --> to be filled when we run the particular session
  X = tf.placeholder(tf.float32, [None, X_train.shape[1]])
  Y = tf.placeholder(tf.float32, [None, 1])
  ### Learn that Variable means they can be updated.Weights for each param + bias
  W = tf.Variable(tf.ones([X_train.shape[1], 1]))

  init = tf.global_variables_initializer()

  # Start training
  Y_ = tf.matmul(X, W)
  cost = tf.reduce_mean(tf.square(Y_ - Y))
  training_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

  sess = tf.Session()
  sess.run(init)

  for epoch in range(training_epochs):
    ### Learn that X_train is feed to the X placeholder for training, Y to the cost function step
    sess.run(training_step, feed_dict={X: X_train, Y: Y_train})
    cost_history = np.append(cost_history, sess.run(cost, feed_dict={X: X_train, Y: Y_train}))

  plt.plot(range(len(cost_history)), cost_history)
  plt.axis([0, training_epochs, 0, np.max(cost_history)])
  # plt.show()

  # Test
  pred_y = sess.run(Y_, feed_dict={X: X_test})

  correct_count = 0
  for i, prediction in enumerate(pred_y):
    if Y_test[i] == 1 and prediction > 0.5 or Y_test[i] == 0 and prediction <= 0.5:
      correct_count += 1

  print("Total number of tests: %d" % len(pred_y))
  print("Total correct predictions: %d" % correct_count)
  print("Accuracy: %.4f" % (correct_count * 1.0 / len(pred_y)))

  mse = tf.reduce_mean(tf.square(pred_y - Y_test))
  print("MSE: %.4f" % sess.run(mse))

  fig, ax = plt.subplots()
  ax.scatter(Y_test, pred_y)
  ax.plot([Y_test.min(), Y_test.max()], [Y_test.min(), Y_test.max()], 'k--', lw=3)
  ax.set_xlabel('Measured')
  ax.set_ylabel('Predicted')
  # plt.show()

  sess.close()

X_train, Y_train, X_test, Y_test = get_train_test_np_arrays()
X_reshaped_train, Y_reshaped_train = append_bias_reshape(X_train, Y_train)
X_reshaped_test, Y_reshaped_test = append_bias_reshape(X_test, Y_test)

tf_linear_regression(X_reshaped_train, Y_reshaped_train, X_reshaped_test, Y_reshaped_test)
# print stats.describe(X_train)

