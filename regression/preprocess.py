from pymongo import MongoClient
from progressbar import ProgressBar, Bar, Percentage, FormatLabel, ETA
import numpy as np
import os

def preprocess(percent_training_set):
  client = MongoClient()
  db = client.dotabot
  matches = db.matches

  NUM_HEROES = 114
  NUM_FEATURES = NUM_HEROES * 2

  NUM_MATCHES = matches.count()

  # X is a matrix where each row represents a match and each column is a
  # feature indicating whether a specific hero is picked(1) or not(0).

  # Y is a bit vector indicating whether radiant won(1) or lost(-1).
  X = np.zeros((NUM_MATCHES, NUM_FEATURES), dtype = np.int32)
  Y = np.zeros(NUM_MATCHES, dtype=np.int32)

  widgets = [FormatLabel('Processed: %(value)d/%(max)d matches. '), ETA(), Percentage(), ' ', Bar()]
  pbar = ProgressBar(widgets = widgets, maxval = NUM_MATCHES).start()

  for i, record in enumerate(matches.find()):
    pbar.update(i)
    Y[i] = 1 if record['radiant_win'] else 0

    for player in record['players']:
      hero_id = player['hero_id'] - 1

      if player['player_slot'] >= 128:
        hero_id += NUM_HEROES

      X[i, hero_id] = 1

  pbar.finish()

  print "Generate permutation of training & testing sets."
  indices = np.random.permutation(NUM_MATCHES)
  train_indices = indices[0 : NUM_MATCHES * percent_training_set]
  test_indices = indices[NUM_MATCHES * percent_training_set : NUM_MATCHES]

  X_train = X[train_indices]
  Y_train = Y[train_indices]

  X_test = X[test_indices]
  Y_test = Y[test_indices]

  print "Compressing & saving training & testing sets."
  test_set_output_path = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    'preprocessed_data_sets',
    'test.npz')
  train_set_output_path = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    'preprocessed_data_sets',
    'train.npz')

  np.savez_compressed(test_set_output_path, X=X_test, Y=Y_test)
  np.savez_compressed(train_set_output_path, X=X_train, Y=Y_train)

if __name__ == "__main__":
  import sys
  preprocess(float(sys.argv[1]))