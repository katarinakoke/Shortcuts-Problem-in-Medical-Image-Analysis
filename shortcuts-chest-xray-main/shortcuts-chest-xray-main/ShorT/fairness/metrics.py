#@title Fairness metrics
import numpy as np
# As per the work of Alabdulmohsin et al., 2021

def fairness_metrics(y_pred, y_true, sens_attr):
  eps = 1e-5
  groups = np.unique(sens_attr).tolist()

  max_error = 0
  min_error = 1

  max_mean_y = 0
  min_mean_y = 1

  max_mean_y0 = 0  # conditioned on y = 0
  min_mean_y0 = 1

  max_mean_y1 = 0
  min_mean_y1 = 1

  for group in groups:
    yt = y_true[sens_attr == group].astype('int32')
    ypt = (y_pred[sens_attr == group]).astype('int32')
    err = -np.mean(yt * np.log(ypt+eps) + (1-yt)*np.log(1-ypt+eps))
    mean_y = np.mean(y_pred[sens_attr == group])
    neg = np.logical_and(sens_attr == group, y_true == 0)
    pos = np.logical_and(sens_attr == group, y_true == 1)
    mean_y0 = np.mean(y_pred[neg])
    mean_y1 = np.mean(y_pred[pos])

    if err > max_error:
      max_error = err
    if err < min_error:
      min_error = err

    if mean_y > max_mean_y:
      max_mean_y = mean_y
    if mean_y < min_mean_y:
      min_mean_y = mean_y

    if mean_y0 > max_mean_y0:
      max_mean_y0 = mean_y0
    if mean_y0 < min_mean_y0:
      min_mean_y0 = mean_y0

    if mean_y1 > max_mean_y1:
      max_mean_y1 = mean_y1
    if mean_y1 < min_mean_y1:
      min_mean_y1 = mean_y1
  
  eo = 0.5*(max_mean_y0 - min_mean_y0 + max_mean_y1 - min_mean_y1)
  dp = max_mean_y - min_mean_y
  err_parity = max_error - min_error

  return eo, dp, err_parity