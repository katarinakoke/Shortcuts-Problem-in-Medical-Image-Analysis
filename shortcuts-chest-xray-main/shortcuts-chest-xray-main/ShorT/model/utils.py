#@title Model utils
import tensorflow as tf
import ml_collections as mlc
from ml_collections import ConfigDict
import numpy as np

K = tf.keras.backend

class GradientReversal(tf.keras.layers.Layer):
    @tf.custom_gradient
    def grad_reverse(self, x):
        y = tf.identity(x)
        def custom_grad(dy):
          return self.hp_lambda * dy
        return y, custom_grad

    def __init__(self, hp_lambda, **kwargs):
        super(GradientReversal, self).__init__(**kwargs)
        self.hp_lambda = K.variable(hp_lambda, dtype='float', name='hp_lambda')

    def call(self, x, mask=None):
        return self.grad_reverse(x)

    def set_hp_lambda(self,hp_lambda):
        K.set_value(self.hp_lambda, hp_lambda)
    
    def increment_hp_lambda_by(self,increment):
        new_value = float(K.get_value(self.hp_lambda)) +  increment
        K.set_value(self.hp_lambda, new_value)

    def get_hp_lambda(self):
        return float(K.get_value(self.hp_lambda))


class BaselineArch():
  """Superclass for multihead training."""

  def __init__(self, main="y", aux=None, dtype=tf.float32, pos=None):
    """Initializer.

    Args:
      main: name of variable for the main task
      aux: nema of the variable for the auxiliary task
      dtype: desired dtype (e.g. tf.float32).
      pos: ConfigDict that specifies the index of x, y, c, w, u in data tuple.
        Default: data is of the form (x, y, c, w, u).
    """
    self.model = None
    self.inputs = "x"
    self.main = main
    self.aux = aux
    self.dtype = dtype
    if pos is None:
      pos = mlc.ConfigDict()
      pos.x, pos.y, pos.a = 0, 1, 2
    self.pos = pos

  def get_input(self, *batch):
    """Fetch model input from the batch."""
    # first input
    stack = tf.cast(batch[self.pos[self.inputs[0]]], self.dtype)
    # fetch remaining ones
    for c in self.inputs[1:]:
      stack = tf.concat([stack, tf.cast(batch[self.pos[c]], self.dtype)],
                        axis=1)
    return stack

  def get_output(self, *batch):
    """Fetch outputs from the batch."""
    if self.aux:
      return (tf.cast(batch[self.pos[self.main]],self.dtype),
              tf.cast(batch[self.pos[self.aux]], self.dtype))
    else:
      return (tf.cast(batch[self.pos[self.main]],self.dtype))

  def split_batch(self, *batch):
    """Split batch into input and output."""
    return self.get_input(*batch), self.get_output(*batch)

  def fit(self, data: tf.data.Dataset, **kwargs):
    """Fit model on data."""
    ds = data.map(self.split_batch)
    self.model.fit(ds, **kwargs)
    
  def predict(self, model_input, **kwargs):
    """Predict target Y given the model input. See also: predict_mult()."""
    y_pred = self.model.predict(model_input, **kwargs)
    return y_pred

  def predict_mult(self, data: tf.data.Dataset, num_batches: int, **kwargs):
    """Predict target Y from the TF dataset directly. See also: predict()."""
    y_true = []
    y_pred = []
    ds_iter = iter(data)
    for _ in range(num_batches):
      batch = next(ds_iter)
      model_input, y = self.split_batch(*batch)
      y_true.extend(y)
      y_pred.extend(self.predict(model_input, **kwargs))
    return np.array(y_true), np.array(y_pred)

  def score(self, data: tf.data.Dataset, num_batches: int, 
            metric: tf.keras.metrics.Metric , **kwargs):
    """Evaluate model on data.

    Args:
      data: TF dataset.
      num_batches: number of batches fetched from the dataset.
      metric: which metric to evaluate (schrouf not be instantiated).
      **kwargs: arguments passed to predict() method.

    Returns:
      score: evaluation score.
    """
    y_true, y_pred = self.predict_mult(data, num_batches, **kwargs)
    return metric()(y_true, y_pred).numpy()


class MultiHead(BaselineArch):
  """Multihead training."""

  def __init__(self, cfg, main, aux, dtype=tf.float32, pos=None): 
    """Initializer.

    Args:
      cfg: A config that describes the MLP architecture.
      main: variable for the main task
      aux: variable for the auxialiary task
      dtype: desired dtype (e.g. tf.float32) for casting data.
    """
    super(MultiHead, self).__init__(main, aux, dtype, pos)
    self.main = "y"
    self.aux = "a"
    self.cfg = cfg
    # build architecture
    self.model, self.feat_extract = self.build()

  def build(self):
    """Build model."""
    cfg = self.cfg
    input_shape = cfg.model.x_dim

    # set config params to defaults if missing
    use_bias = cfg.model.get("use_bias", True)
    activation = cfg.model.get("activation", "relu")
    output_activation = cfg.model.get("output_activation", "sigmoid")

    model_input = tf.keras.Input(shape=input_shape)
    flatten_input = tf.keras.layers.Flatten()(model_input)
    if cfg.model.depth:
      x = tf.keras.layers.Dense(cfg.model.width, use_bias=use_bias,
                                activation=activation,
                                kernel_regularizer=cfg.model.regularizer)(flatten_input)
      for _ in range(cfg.model.depth - 1):
        x = tf.keras.layers.Dense(cfg.model.width, use_bias=use_bias,
                                  activation=activation,
                                  kernel_regularizer=cfg.model.regularizer)(x)
    else:
      x = flatten_input
    feature_extractor = tf.keras.models.Model(inputs=flatten_input,
                                              outputs=x)
    # output layer - a single linear layer
    y = tf.keras.layers.Dense(cfg.model.output_dim,
                              use_bias=cfg.model.use_bias,
                              name="output",
                              activation=output_activation,
                              kernel_regularizer=cfg.model.regularizer)(x)
    # attribute layer - an extra dense layer is required for gradients to flow back
    attr_activation = cfg.model.get("attr_activation", "sigmoid")
    input_branch_a = GradientReversal(hp_lambda=cfg.model.attr_grad_updates)(x)
    a_branch = tf.keras.layers.Dense(cfg.model.branch_dim,
                    use_bias=cfg.model.use_bias,
                    name="attr_branch",
                    activation=activation,
                    kernel_regularizer=cfg.model.regularizer)(input_branch_a)
    a = tf.keras.layers.Dense(cfg.model.attr_dim,
                        use_bias=cfg.model.use_bias,
                        name="attribute",
                        activation=attr_activation,
                        kernel_regularizer=cfg.model.regularizer)(a_branch)
    


    # choose optimizer
    if cfg.opt.name == "sgd":
      opt = tf.keras.optimizers.SGD(learning_rate=cfg.opt.learning_rate,
                                    momentum=cfg.opt.get("momentum", 0.9))
    elif cfg.opt.name == "adam":
      opt = tf.keras.optimizers.Adam(learning_rate=cfg.opt.learning_rate)
    else:
      raise ValueError("Unrecognized optimizer type."
                       "Please select either 'sgd' or 'adam'.")

    # define losses
    losses = {
        "output": cfg.model.get("output_loss", "binary_crossentropy"),
        "attribute": cfg.model.get("attribute_loss", "binary_crossentropy")
    }
    loss_weights = {"output": 1.0,
                    "attribute": cfg.get("attr_loss_weight", 1.0)}
    metrics = {"output": tf.keras.metrics.AUC(),
               "attribute": tf.keras.metrics.AUC()}

    # build model
    model = tf.keras.models.Model(inputs=model_input, outputs=[y,a])
    model.build(input_shape)
    # model.compile(optimizer=opt, loss=tf.keras.losses.BinaryCrossentropy(),
    #               metrics=tf.keras.metrics.BinaryAccuracy())
    model.compile(optimizer=opt, loss=losses, loss_weights=loss_weights,
                  metrics=metrics)
    return model, feature_extractor

  def predict_mult(self, data: tf.data.Dataset, num_batches: int, **kwargs):
    """Predict from the TF dataset directly. See also: predict()."""
    # infer dimensions
    pos = self.pos
    batch = next(iter(data))
    y_dim = batch[pos.y].shape[1]
    a_dim = batch[pos.a].shape[1]

    # begin
    data_iter = iter(data)
    a_true_all = np.array([]).reshape((0, a_dim))
    a_pred_all = np.array([]).reshape((0, a_dim))
    y_true_all = np.array([]).reshape((0, y_dim))
    y_pred_all = np.array([]).reshape((0, y_dim))

    for _ in range(num_batches):
      batch = next(data_iter)
      x, y_true, a_true = batch[pos.x], batch[pos.y], batch[pos.a]
      a_true = tf.reshape(a_true, [-1, a_true.shape[-2]])
      y_true = tf.reshape(y_true, [-1, y_true.shape[-2]])
      y_pred, a_pred = self.predict(x, **kwargs)
      a_true_all = np.append(a_true_all, a_true, axis=0)
      a_pred_all = np.append(a_pred_all, a_pred, axis=0)
      y_true_all = np.append(y_true_all, y_true, axis=0)
      y_pred_all = np.append(y_pred_all, y_pred, axis=0)

    return (y_true_all, a_true_all), (y_pred_all, a_pred_all)

  def score(self, data: tf.data.Dataset, num_batches: int, 
            metric: tf.keras.metrics.Metric, **kwargs):
    """Evaluate model on data.

    Args:
      data: TF dataset.
      num_batches: number of batches fetched from the dataset.
      metric: which metric to evaluate (should not be instantiated).
      **kwargs: arguments passed to predict() method.

    Returns:
      score: evaluation score.
    """
    out_true, out_pred = self.predict_mult(data, num_batches, **kwargs)
    scores = []
    for head in range(len(out_true)):
      score = metric()(out_true[head], out_pred[head])
      scores.append(score.numpy())
    return scores


# Can be used as a single task model fully trained or from a pre-trained
# feature extractor

class SingleHead(BaselineArch):
  """Singlehead training."""

  def __init__(self, cfg, main, dtype=tf.float32, pos=None, feat_extract=None): 
    """Initializer.

    Args:
      cfg: A config that describes the MLP architecture.
      main: variable for the main task
      aux: variable for the auxialiary task
      dtype: desired dtype (e.g. tf.float32) for casting data.
    """
    super(SingleHead, self).__init__(main, None, dtype, pos)
    self.main = "a"
    self.cfg = cfg
    # build architecture
    self.model = self.build(feat_extract)

  def build(self, feat_extract=None):
    """Build model."""
    cfg = self.cfg
    input_shape = cfg.model.x_dim

    # set config params to defaults if missing
    use_bias = cfg.model.get("use_bias", True)
    activation = cfg.model.get("activation", "relu")
    output_activation = cfg.model.get("output_activation", "sigmoid")

    model_input = tf.keras.Input(shape=input_shape)
    flatten_input = tf.keras.layers.Flatten()(model_input)
    if not feat_extract:
      if cfg.model.depth:
        x = tf.keras.layers.Dense(cfg.model.width, use_bias=use_bias,
                                  activation=activation,
                                  kernel_regularizer=cfg.model.regularizer)(flatten_input)
        for _ in range(cfg.model.depth - 1):
          x = tf.keras.layers.Dense(cfg.model.width, use_bias=use_bias,
                                    activation=activation,
                                    kernel_regularizer=cfg.model.regularizer)(x)
      else:
        x = flatten_input
      feature_extractor = x
    else:
      feat_extract.trainable = False
      feature_extractor = feat_extract(flatten_input, training=False)
  
    # output layer
    y = tf.keras.layers.Dense(cfg.model.output_dim,
                              use_bias=cfg.model.use_bias,
                              name="output",
                              activation=output_activation,
                              kernel_regularizer=cfg.model.regularizer)(feature_extractor)  

    # choose optimizer
    if cfg.opt.name == "sgd":
      opt = tf.keras.optimizers.SGD(learning_rate=cfg.opt.learning_rate,
                                    momentum=cfg.opt.get("momentum", 0.9))
    elif cfg.opt.name == "adam":
      opt = tf.keras.optimizers.Adam(learning_rate=cfg.opt.learning_rate)
    else:
      raise ValueError("Unrecognized optimizer type."
                       "Please select either 'sgd' or 'adam'.")

    # build model
    model = tf.keras.models.Model(inputs=model_input, outputs=y)
    model.build(input_shape)
    model.compile(optimizer=opt,
                  loss=cfg.model.get("output_loss", "binary_crossentropy"),
                  metrics=tf.keras.metrics.AUC())

    return model