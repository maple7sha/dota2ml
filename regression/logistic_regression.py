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
      # save_checkpoints_steps = 10,
      save_summary_steps=1,
      save_checkpoints_secs = 5),
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
            # metrics =  {
            #    'prediction accuracy':
            #        tf.contrib.learn.MetricSpec(
            #            metric_fn = streaming_accuracy,
            #            prediction_key = 'prediction_key',
            #            label_key = 'label_key',
            #            weight_key = 'input_key')

  # for name in model.get_variable_names():
  #   print name
  #   print model.get_variable_value(name)
  # return

  print "Evaluating linear regressor."
  results = model.evaluate(input_fn = lambda: input_fn(features_test, targets_test), 
                           steps = 10,
                           metrics =  {
                              'prediction accuracy':
                                  tf.contrib.learn.MetricSpec(
                                      metric_fn = streaming_accuracy,
                                      prediction_key = 'prediction_key',
                                      label_key = 'label_key',
                                      weight_key = 'input_key')
                           })

  print result

  # for key in sorted(results):
  #   print("%s: %s" % (key, results[key]))

  return model

def train_regressor_model_tf_session(learning_rate,
                                     steps,
                                     batch_size,
                                     training_epochs,
                                     display_step,
                                     X_train,
                                     Y_train,
                                     X_test,
                                     Y_test):
  # tf graph input
  X = tf.placeholder(tf.float32, [None, X_train.shape[1]]) # fist param (None) is # of examples; then #of features
  Y = tf.placeholder(tf.float32, [None, 1])
  # print X.shape
  # print Y.shape

  # tf model weights
  W = tf.Variable(tf.zeros([X_train.shape[1], 1]))
  b = tf.Variable(tf.zeros([1]))

  # tf predicted Y values
  Y_pred = tf.nn.sigmoid(tf.matmul(X, W) + b) # tf.nn.softmax(tf.matmul(X, W) + b)

  # tf minimize cross entropy (loss function?)
  #cost = tf.reduce_mean(-tf.reduce_sum(Y*tf.log(Y_pred), axis = 1))
  cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = Y_pred, labels = Y))

  # tf gradient descent optimizer
  optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

  # tf initializer variables 
  init = tf.global_variables_initializer()

  cost_history = np.empty(shape=[1], dtype=float)
  test_accuracy_history = np.empty(shape=[1], dtype=float)

  # tf launch graph
  with tf.Session() as sess:
    sess.run(init)

    # tf train
    for epoch in range(training_epochs):
      total_batch = int(X_train.shape[0] / batch_size)

      average_cost = 0
      for i in range(total_batch):
        if i + 1 == total_batch: 
          X_batch = X_train[i * batch_size :]
          Y_batch = Y_train[i * batch_size :]
        else:
          X_batch = X_train[i * batch_size : (i + 1) * batch_size]
          Y_batch = Y_train[i * batch_size : (i + 1) * batch_size]

        Y_batch = np.reshape(Y_batch, [len(Y_batch), 1])
        _, c = sess.run([optimizer, cost], feed_dict={X: X_batch, 
                                                      Y: Y_batch})
        average_cost += c / total_batch

      cost_history = np.append(cost_history, average_cost)

      if (epoch + 1) % display_step == 0:
        print("Epoch: %04d" % (epoch + 1), "cost={:.9f}".format(average_cost))

        # Test
        Y_hat = sess.run(Y_pred, feed_dict={X: X_test})
        results = np.empty(len(Y_hat), dtype=float)

        for i in range(len(Y_hat)):
          results[i] = 1 if np.absolute(Y_hat[i] - Y_test[i]) < 0.5 else 0
        accuracy = sess.run(tf.reduce_mean(results))
        print accuracy
        test_accuracy_history = np.append(test_accuracy_history, accuracy)

        
        # print "Accuracy: ", test_accuracy_history

    plt.plot(range(len(cost_history)), cost_history)
    plt.axis([0, training_epochs, 0, np.max(cost_history)])
    plt.show()

    plt.plot(range(len(test_accuracy_history)), test_accuracy_history)
    plt.axis([0, len(test_accuracy_history), 0, 1])
    plt.show()
    
    # Test model
    Y_hat = sess.run(Y_pred, feed_dict={X: X_test})

    correct_count = 0
    for i, prediction in enumerate(Y_hat):
      if Y_test[i] == 1 and prediction > 0.5 or Y_test[i] == 0 and prediction <= 0.5:
        correct_count += 1

    print("Total number of tests: %d" % len(Y_hat))
    print("Total correct predictions: %d" % correct_count)
    print("Accuracy: %.4f" % (correct_count * 1.0 / len(Y_hat)))

    print Y_hat.shape
    print Y_test.shape
    print Y_hat
    print Y_test
    mse = tf.reduce_mean(tf.square(Y_hat - np.reshape(Y_test, [len(Y_test), 1])))
    print("MSE: %.4f" % sess.run(mse))


def shuffle_train_test_sets(train, test):
  assert len(train) == len(test)
  permutation = np.random.permutation(len(a))
  return train[permutation], test[permutation]

X_train, Y_train, X_test, Y_test = get_train_test_np_arrays()

# Training variables:
learning_rate = 0.01
steps = 1000
batch_size = 1000
epoch = 5000
display_step = 100

# model = train_regressor_model(learning_rate, steps, batch_size, 
#                               features_train, targets_train)

train_regressor_model_tf_session(learning_rate, 
                                 steps, 
                                 batch_size, 
                                 epoch,
                                 display_step,
                                 X_train, 
                                 Y_train,
                                 X_test, 
                                 Y_test)


# with tf.Session() as sess:
#   saver = tf.train.import_meta_graph(os.path.join(MODEL_DIR, 
#       'checkpoint'))
#   saver.restore(sess, MODEL_DIR)
#   print(sess.run(tf.global_variables_initializer()))
  

