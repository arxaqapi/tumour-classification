
from keras.layers import (
  Layer,
  Conv2D,
  Lambda,
  BatchNormalization,
  Add,
  ReLU,
)


class ResidualLayer(Layer):
  def __init__(self, n_filters, kernel_size=(5, 1), strides=(1, 1)):
    super().__init__() # name="ResidualLayer"
    self.conv2D_1 = Conv2D(n_filters, kernel_size, strides=(1, 1), padding='same', use_bias=False)
    self.batch_norm_1 = BatchNormalization()
    # ReLU
    self.batch_norm_2 = BatchNormalization()
    self.conv2D_2 = Conv2D(n_filters, kernel_size, strides=strides, padding='same', use_bias=False)
    # ReLU

    # Dans le cas ou les données sont sous-échantillonnées (strides > 1, ...)
    # il faut s'assurer que la dimension de sortie de la skip-connection est la même
    # que celle a la sortie de la dernière couche convolutive
    # if n_filters != self.conv2D_2.output_shape[-1] or (strides != (1, 1)):
    if strides != (1, 1):
      self.skip_connection = Conv2D(n_filters, kernel_size, strides=strides, padding='same', use_bias=False)
    else:
      self.skip_connection = Lambda(lambda x : x)

    self.activation = ReLU()
  def call(self, inputs):
    x_skip = self.skip_connection(inputs)

    x = self.conv2D_1(inputs)
    x = self.batch_norm_1(x)
    x = self.activation(x)
    x = self.batch_norm_2(x)
    x = self.conv2D_2(x)
    x = self.activation(x)
    output = Add()([x, x_skip])
    return output # self.activation(output)