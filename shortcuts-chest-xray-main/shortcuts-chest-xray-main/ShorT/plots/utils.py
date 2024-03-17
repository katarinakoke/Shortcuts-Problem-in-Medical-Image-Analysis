#@title Plot utils
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib import cm

SMALL_SIZE = 10
MEDIUM_SIZE = 14
BIGGER_SIZE = 18

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

palette = sns.color_palette('tab20b')

def plot_scale_gradients_encoding(scale_gradients, encoding_a, 
                                  upper_bound, lower_bound):
  """Plots the intervention compared to attribute encoding.

  Inputs:
  scale_gradients: numpy array of scale gradients used
  encoding_a: numpy array of attribute encoding values, as measure by transfer learning
  upper_bound: list or numpy array of maximum attribute encoding
  lower_bound: list or numpy array of minimum attribute encoding
  """
  
  sg_mean = np.mean(encoding_a, axis=1)
  sg_std = np.std(encoding_a, axis=1)

  fig = plt.figure(figsize=(5,4))
  ax = fig.add_axes([0,0,1,1])
  ax.errorbar(scale_gradients, sg_mean, 
                yerr=sg_std, 
                fmt='x', 
                color='tab:blue',
                ecolor='tab:blue')
  plt.hlines(np.mean(upper_bound),np.min(scale_gradients), np.max(scale_gradients),
            colors=[0.4,0.4,0.4],linestyles='dashed')
  plt.hlines(np.mean(lower_bound),np.min(scale_gradients), np.max(scale_gradients),
            colors=[0.4,0.4,0.4],linestyles='dashed')
  ax.set_xlabel('Scale Gradient')
  ax.set_ylabel('Attribute encoding')
  plt.show

def plot_fairness_encoding(encoding_m, fair_m, perf_m, perf_thresh = 0):
  """Plots fairness results vs attribute encoding.
  
  Inputs:
  encoding_m: encoding metric result, as numpy array
  fair_m: fairness metric result, as numpy array
  perf_m: model performance on the output label, as numpy array
  perf_thresh: what minimum performance to consider
  """

  filt = perf_m <= perf_thresh

  fig = plt.figure(figsize=(5,4))
  plt.scatter(encoding_m, fair_m,color=[0.6,0,0.2],alpha=0.5)
  plt.scatter(encoding_m[filt], fair_m[filt],color=[0.2,0.2,0.2])
  plt.xlabel('Attribute Accuracy')
  plt.ylabel('Fairness')
  plt.show()

def point_z_order(c, midpoint):
  deviation = np.zeros_like(c)
  for i in range(c.shape[0]):
    if c[i] > midpoint:
      deviation[i] = (c[i]-midpoint) / (np.max(c)-midpoint)
    else:
      deviation[i] = (midpoint-c[i]) / (midpoint-np.min(c))
  return np.argsort(deviation)

def performance_fairness_age_frontier_plot(encoding_m, fair_m, perf_m,
                                           scale_gradients, cmap='PRGn'):
  
  class MidpointNormalize(mpl.colors.Normalize):
    """
    class to help renormalize the color scale
    """
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        mpl.colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))

  baseline_models = np.argwhere(np.array(scale_gradients)==0.0)[0]
  if baseline_models.shape[0] == 0:
    baseline_models = int(len(scale_gradients)/2)
  midpoint = encoding_m[baseline_models,:].mean()
  baseline_model_perf = perf_m[baseline_models,:].mean()
  baseline_model_fair = fair_m[baseline_models,:].mean()

  print(f'Baseline model Attribute encoding: {midpoint:.2f}')
  print(f'Baseline model Performance: {baseline_model_perf:.4f}')
  print(f'Baseline model Fairness: {baseline_model_fair:.4f}')

  norm =  MidpointNormalize(midpoint = midpoint)

  attr = encoding_m.flatten()
  z_order = point_z_order(attr, midpoint)

  fair =fair_m.flatten()
  perf = perf_m.flatten()
  fig = plt.figure(figsize=(5,4))
  ax = fig.add_axes([0,0,1,1])
  plt.scatter(fair[z_order], perf[z_order], s=30, c=attr[z_order], cmap=cmap, norm=norm)
  # overplot the baseline models in red
  plt.scatter(fair_m[baseline_models,:], perf_m[baseline_models,:], s=30,
                   color=(0.8, 0.2, 0.2))
  plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap),label='Attribute Encoding')
  plt.ylabel('Performance')
  plt.xlabel('Fairness')
  plt.show()