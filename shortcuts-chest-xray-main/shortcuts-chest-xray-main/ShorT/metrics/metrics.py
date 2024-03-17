#@title Define the decision threshold by maximizing the F1-score on validation data

from sklearn.metrics import precision_recall_curve

def f1_curve(truth, prediction_scores, e=1e-6):
  precision, recall, thresholds = precision_recall_curve(truth, prediction_scores)
  f1 = 2*recall*precision/(recall+precision+e)
  return thresholds, f1[:-1]

def threshold_at_max_f1_score(truth, prediction_scores):
  thresholds, f1 = f1_curve(truth, prediction_scores)
  peak_idx = np.argmax(f1)
  return thresholds[peak_idx]