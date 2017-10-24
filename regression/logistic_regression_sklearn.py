import os
import matplotlib.pyplot as plt
import numpy as np
import pickle
import pylab

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

POSITIVE_LABEL = 1
NEGATIVE_LABEL = 0
NUM_HEROES = 114
NUM_FEATURES = 114 * 2

class D2LogisticRegression:
  '''Logistic regression algorithm using the given model.'''
  def __init__(self, model_path):
    with open(model_path, 'r') as input_file:
      self.model = pickle.load(input_file)

  def transform(self, current_team, opponent_team):
    '''Returns a feature vector indicating hero selections for both teams'''
    X = np.zeros(NUM_FEATURES, dtype=np.int8)

    for hero_id in current_team:
      X[hero_id - 1] = 1

    for hero_id in opponent_team:
      X[hero_id - 1 + NUM_HEROES] = 1

    return X

  def score(self, radiant_query):
    '''Score the query by averaging the winning possibility from both the radiant and dire sides'''
    # Reverse the position of the first and second five heroes.
    dire_query = np.concatenate((radiant_query[NUM_HEROES:NUM_FEATURES], 
                                 radiant_query[0:NUM_HEROES]))
    radiant_win_probability = self.model.predict_proba(radiant_query)[0][1]
    dire_win_probability = self.model.predict_proba(dire_query)[0][0]

    return (radiant_win_probability + dire_win_probability) / 2

  def predict(self, current_team, opponent_team):
    match = self.transform(current_team, opponent_team)
    return self.score(match)

  def recommend(self, current_team, opponent_team, hero_candidates):
    '''Returns a list of (hero, winning possibility) pair.
       The current team can be of any len. We will transform the possibilities
       into a feature vector and calculate the scores, and then recommend.
    '''
    options = [(candidate, current_team + [candidate]) for candidate in hero_candidates]

    prob_candidate_pairs = []
    for candidate, team in options:
      match = self.transform(team, opponent_team)
      prob = self.score(query)
      prob_candidate_pairs.append((prob, candidate))
    prob_candidate_pairs = sorted(prob_candidate_pairs, reverse=True)[0:5 - len(my_team)]
    return prob_candidate_pairs

class LearningCurve():
  '''Graph plotting utility.'''
  def score(self, model, radiant_query):
    '''Return the probability of the query being in the positive class.'''
    dire_query = np.concatenate((radiant_query[NUM_HEROES:NUM_FEATURES], radiant_query[0:NUM_HEROES]))
    rad_prob = model.predict_proba(radiant_query)[0][1]
    dire_prob = model.predict_proba(dire_query)[0][0]
    return (rad_prob + dire_prob) / 2

  def evaluate(self, model, X, Y):
    '''Returns the predication accuracy for the given set of examples.'''
    correct_predictions = 0.0
    for i, radiant_query in enumerate(X):
        overall_prob = self.score(model, radiant_query)
        prediction = 1 if (overall_prob > 0.5) else 0
        result = 1 if prediction == Y[i] else 0
        correct_predictions += result

    return correct_predictions / len(X)

  def plot_learning_curve(self, num_points, X_train, Y_train, X_test, Y_test):
    matches_count = len(X_train)
    training_set_sizes = []
    for div in list(reversed(range(1, num_points + 1))):
      training_set_sizes.append(matches_count / div)

    accuracy_list = []
    for training_set_size in training_set_sizes:
      model = train(X_train, Y_train, training_set_size)
      accuracy = self.evaluate(model, X_test, Y_test)
      accuracy_list.append(accuracy)
      print 'Accuracy for %d training examples: %f' % (training_set_size, accuracy)

    plt.plot(np.array(training_set_sizes), np.array(accuracy_list))
    plt.ylabel('Accuracy')
    plt.xlabel('Number of training samples')
    plt.title('Logistic Regression Learning Curve')
    pylab.show()

  def plot_learning_curves(self, num_points, X_train, Y_train, X_test, Y_test):
    matches_count = len(X_train)
    training_set_sizes = []
    for div in list(reversed(range(1, num_points + 1))):
      training_set_sizes.append(matches_count / div)

    test_errors = []
    training_errors = []
    for training_set_size in training_set_sizes:
        model = train(X_train, Y_train, training_set_size)
        test_error = self.evaluate(model, X_test, Y_test)
        training_error = self.evaluate(model, X_train, Y_train)
        test_errors.append(test_error)
        training_errors.append(training_error)
        print 'Accuracy for %d testing examples: %f' % (training_set_size, test_error)
        print 'Accuracy for %d training examples: %f' % (training_set_size, training_error)

    plt.plot(training_set_sizes, training_errors, 'bs-', label='Training accuracy')
    plt.plot(training_set_sizes, test_errors, 'g^-', label='Test accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Number of training samples')
    plt.title('Logistic Regression Learning Curve')
    plt.legend(loc='lower right')
    pylab.show()

def train(X, Y, num_samples):
  print 'Training logistic regression model using %d matches.' % num_samples
  return LogisticRegression().fit(X[0:num_samples], Y[0:num_samples])

def predict_match_result(algo, match):
  '''Given the ML algorithm and a feature vector of the given match hero selections
     Return predicted result.
  '''
  prob = algo.score(match)
  return POSITIVE_LABEL if prob > 0.5 else NEGATIVE_LABEL

def calculate_f1score(algo, X, Y):
  print 'Calculating f1 score.'
  Y_true = Y
  num_matches = len(Y_true)

  Y_predict = np.zeros(num_matches)

  for i, match in enumerate(X):
    Y_predict[i] = predict_match_result(algo, match)

  prec, recall, f1, support = precision_recall_fscore_support(Y_true, Y_predict, average='macro')
  accuracy = accuracy_score(Y_true, Y_predict)

  print 'Precision: ',prec
  print 'Recall: ',recall
  print 'F1 Score: ',f1
  print 'Support: ',support
  print 'Accuracy: ',accuracy

def main():
  # Train the model.
  train_set_input_path = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    'preprocessed_data_sets',
    'train.npz')

  preprocessed_train = np.load(train_set_input_path)
  X_train = preprocessed_train['X']
  Y_train = preprocessed_train['Y']

  model = train(X_train, Y_train, len(X_train))

  # Output the model.
  model_output_path = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    'model_sklearn',
    'model.pkl')

  with open(model_output_path, 'w') as output_file:
    pickle.dump(model, output_file)

  # Load the tests.
  test_set_input_path = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    'preprocessed_data_sets',
    'test.npz')

  preprocessed_test = np.load(test_set_input_path)
  X_test = preprocessed_test['X']
  Y_test = preprocessed_test['Y']

  algo = D2LogisticRegression(model_path=model_output_path)

  # Calculate & print f1 score (precision, recall, f1, support)
  calculate_f1score(algo, X_test, Y_test)

  plotter = LearningCurve()
  plotter.plot_learning_curve(20, X_train, Y_train, X_test, Y_test)
  plotter.plot_learning_curves(20, X_train, Y_train, X_test, Y_test)

if __name__ == "__main__":
  main()