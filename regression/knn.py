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
from tensorflow.contrib.metrics import streaming_accuracy

MODEL_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                         'model')

def preprocess_features(dataframe):
  output_features = dataframe.copy()
  return output_features

def preprocess_targets(dataframe):
  output_targets = dataframe.copy()
  return output_targets

def get_hero_names_sorted_by_id():
  heroes_txt_path = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    'misc',
    'heroes.txt')

  with open(heroes_txt_path) as data_file:
    data_json = json.load(data_file)

    data_json['heroes'] = sorted(data_json['heroes'], 
                                 key = lambda hero: (int(hero['id'])), 
                                 reverse = False)

    heroes = []
    for hero in data_json['heroes']:
      # print hero['id'], hero['name']
      heroes.append('radiant_' + hero['name'])

    for hero in data_json['heroes']:
      # print hero['id'], hero['name']
      heroes.append('dire_' + hero['name'])

    return heroes

def get_train_test_data_frames():
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

  # Get a list of sorted hero names to be used as column names 
  sorted_heroes = get_hero_names_sorted_by_id()

  # Convert to pandas dataframe
  features_train = preprocess_features(pd.DataFrame(X_train))
  features_train.columns = sorted_heroes
  targets_train = preprocess_targets(pd.DataFrame(Y_train))

  features_test = preprocess_features(pd.DataFrame(X_test))
  features_test.columns = sorted_heroes
  targets_test = preprocess_targets(pd.DataFrame(Y_test))

  print "Training data set summary: "
  display.display(features_train.describe())
  display.display(features_test.describe())
  display.display(targets_train.describe())
  display.display(targets_test.describe())

  return features_train, targets_train, features_test, targets_test

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

def train_knn_model(display_step,
                    X_train,
                    Y_train,
                    X_test,
                    Y_test,
                    K):
  # tf graph input
  xtr = tf.placeholder(tf.float32, [None, X_train.shape[1]])  # first param (None) is # of examples; then #of features
  xte = tf.placeholder(tf.float32, [X_train.shape[1]])  # placeholder for a single test case

  # NN calculation using L1 Distance by reduce_sum and negate the test set.
  neg_distance = tf.negative(tf.reduce_sum(tf.abs(tf.add(xtr, tf.negative(xte))), axis=1))

  # Prediction: min distance index (k = 1 here)
  pred_values, pred_indices = tf.nn.top_k(neg_distance, k=K, sorted=True)

  # Labels for nearest neighbors
  nearest_neighbors = []
  for i in range(K):
    nearest_neighbors.append(Y_train[i])
  # y is a list of unique values (0, 1), and count is the number of neighbors having that value
  y, _, count = tf.unique_with_counts(nearest_neighbors)

  # Get the slice of y of size 1 in the y list, depending on the begin argument
  pred = tf.slice(y, begin=[tf.argmax(count, 0)], size=tf.constant([1], dtype=tf.int64))[0]

  accuracy = 0

  # tf initialize all variables
  init = tf.global_variables_initializer()

  # tf launch graph
  with tf.Session() as sess:
    sess.run(init)

    # Loop over test data
    for i in range(len(X_test)):
      # Get predicted value
      Y_ = sess.run(pred, feed_dict={xtr: X_train, xte: X_test[i, :]})

      if Y_ == Y_test[i]:
        accuracy += 1. / len(X_test)

      if i % display_step == 0:
        print("Accuracy: ", accuracy * len(X_test) / (i + 1));

  print("Accuracy: ", accuracy);

def shuffle_train_test_sets(train, test):
  assert len(train) == len(test)
  permutation = np.random.permutation(len(a))
  return train[permutation], test[permutation]

X_train, Y_train, X_test, Y_test = get_train_test_np_arrays()

# Training variables:
display_step = 100
K = 3

# Try with different K values
train_knn_model(display_step,
                X_train, 
                Y_train,
                X_test, 
                Y_test,
                K)

