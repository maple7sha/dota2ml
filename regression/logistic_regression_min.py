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

def input_fn(df_features, df_targets):
  # www.tensorflow.org/tutorials/wide
  # Creates a dictionary mapping from each categorical feature column name (k)
  # to the values of that column stored in a tf.SparseTensor.
  categorical_cols = {k: tf.SparseTensor(
      indices = [[i, 0] for i in range(df_features[k].size)],
      values = df_features[k].values,
      dense_shape = [df_features[k].size, 1])
                      for k in df_features.columns}

  label = tf.constant(df_targets[0].values)

  return categorical_cols, label

def train_regressor_model(learning_rate, 
                          steps,
                          batch_size,
                          features_train,
                          targets_train):
  """Trains a linear regressor model.

  Returns: 
    The trained model.
  """
  ### Build the model
  print "Producing tensors."

  # At end of each period, the model is exported and metrics gathered.
  periods = 10 

  # Create an iterable(list) of feature columns.
  feature_columns = []
  for k in features_train.columns:
    feature_columns.append(tf.contrib.layers.sparse_column_with_integerized_feature(
        column_name = k, bucket_size = 2, combiner = "sum", dtype = dtypes.int32))

  print "Building linear regressor."
  # validation_monitor = tf.contrib.learn.monitors.ValidationMonitor(features_test, 
  #                                                                  targets_test)

  model = tf.contrib.learn.LinearRegressor(
    feature_columns = feature_columns,
    optimizer = tf.train.GradientDescentOptimizer(learning_rate),
    # gradient_clip_norm = 5.0, # TODO
    config = tf.contrib.learn.RunConfig(
      save_checkpoints_steps = 10,
      save_checkpoints_secs = None),
    gradient_clip_norm = 5.0,
    model_dir = os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        'model')
  )

  # # Hooks to record metrics periodically during training & testing
  # hook_train = 

  ### Train the model
  print "Training linear regressor."
  model.fit(input_fn = lambda: input_fn(features_train, targets_train),
            steps = steps,
            monitors = [])

  # for name in model.get_variable_names():
  #   print name
  #   print model.get_variable_value(name)
  # return

  print "Evaluating linear regressor."
  results = model.evaluate(input_fn = lambda: input_fn(features_test, targets_test), 
                           steps = 1)["accuracy"]

  print result

  for key in sorted(results):
    print("%s: %s" % (key, results[key]))

  return model

features_train, targets_train, features_test, targets_test = \
  get_train_test_data_frames()

# Training variables:
learning_rate = 0.001
steps = 200
batch_size = 20

model = train_regressor_model(learning_rate, steps, batch_size, 
                              features_train, targets_train)

# with tf.Session() as sess:
#   saver = tf.train.import_meta_graph(os.path.join(MODEL_DIR, 
#       'checkpoint'))
#   saver.restore(sess, MODEL_DIR)
#   print(sess.run(tf.global_variables_initializer()))
  

