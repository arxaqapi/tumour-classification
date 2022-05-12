from Layers import ResidualLayer
from keras.models import Model, Sequential
from keras.layers import (
  LocallyConnected2D,
  Dense,
  Flatten,
  BatchNormalization,
  Dropout,
  ReLU,
)


class IsotopeNet(Model):
  def __init__(self, n_classes: int = 2):
    super(IsotopeNet, self).__init__(name="IsotopeNet")
    self.res_bloc_1 = ResidualLayer(n_filters=8, kernel_size=(3, 1))
    self.res_bloc_2 = ResidualLayer(n_filters=8, kernel_size=(3, 1), strides=(5, 1))
    self.res_bloc_3 = ResidualLayer(n_filters=8, kernel_size=(3, 1))
    self.res_bloc_4 = ResidualLayer(n_filters=1, kernel_size=(3, 1), strides=(3, 1))

    self.output_bloc = Sequential([
      ReLU(),
      Dropout(rate=.3),
      LocallyConnected2D(filters=1, kernel_size=(5, 1)),
      BatchNormalization(),
      Flatten(),
      Dense(n_classes, activation='softmax', kernel_initializer='he_normal')
    ])

  def call(self, inputs):
    x = self.res_bloc_1(inputs)
    x = self.res_bloc_2(x)
    x = self.res_bloc_3(x)
    x = self.res_bloc_4(x)
    output = self.output_bloc(x)
    return output



"""
Start from here:
  - Clean repo
  x Put model into class
  - Create bigger model and compile it
  - Train on dummy data
  - Extract data from TMA
  - Make a tf dataset out of it

Resources:
  - https://raw.githubusercontent.com/raghakot/keras-resnet/master/images/architecture.png
  - https://keras.io/api/layers/locally_connected_layers/locall_connected2d/
"""