from Layers import ResidualLayer
from keras.models import Model, Sequential
from keras.layers import (
  Dense,
  Flatten,
  GlobalAveragePooling2D,
)


class ResidualNet(Model):
  def __init__(self, n_classes: int = 2):
    super(ResidualNet, self).__init__(name="ResidualNet")

    self.res_block = Sequential([
      ResidualLayer(n_filters=16, strides=(1, 1)),

      ResidualLayer(n_filters=32, strides=(3, 1)),
      ResidualLayer(n_filters=32, strides=(1, 1)),

      ResidualLayer(n_filters=64, strides=(3, 1)),
      ResidualLayer(n_filters=64, strides=(1, 1)),

      ResidualLayer(n_filters=128, strides=(3, 1)),
      ResidualLayer(n_filters=128, strides=(1, 1)),
      ResidualLayer(n_filters=128, strides=(3, 1)),
      ResidualLayer(n_filters=128, strides=(1, 1)),
      ResidualLayer(n_filters=128, strides=(3, 1)),
      ResidualLayer(n_filters=128, strides=(1, 1)),
      ResidualLayer(n_filters=128, strides=(3, 1)),
      ResidualLayer(n_filters=128, strides=(1, 1)),
      ResidualLayer(n_filters=128, strides=(3, 1)),
      ResidualLayer(n_filters=128, strides=(1, 1)),

      ResidualLayer(n_filters=128, strides=(3, 1)),
      ResidualLayer(n_filters=256, strides=(3, 1)),
    ])

    self.output_bloc = Sequential([
      GlobalAveragePooling2D(),
      Flatten(),
      Dense(n_classes, activation='softmax', kernel_initializer='he_normal')
    ])
  def call(self, inputs):
    x = self.res_block(inputs)
    output = self.output_bloc(x)
    return output

