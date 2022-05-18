# from IsotopeNet import IsotopeNet
# from ResidualNet import ResidualNet
from AutoEncoder import train_ae, train_convae
from DatasetGeneration import TMADataset
import numpy as np
# from keras.losses import SparseCategoricalCrossentropy
# from tensorflow.keras.optimizers import Adam


# OPTIM = Adam(learning_rate=0.01)
# LOSS = SparseCategoricalCrossentropy()


if __name__ == '__main__':
  # input_shape = (None, 4672, 1, 1)
  # resnet_model = ResidualNet()
  # resnet_model.build(input_shape=input_shape)
  # resnet_model.summary()

  # resnet_model.compile(
  #   optimizer=OPTIM,
  #   loss=LOSS
  # )

  dataset = TMADataset()
  dataset.bounding_boxes_of_tissue_samples()
  dataset.manually_assign_labels()
  dataset.set_all_spectrums(overwrite=False)

  X_train, X_test, y_train, y_test = dataset.get_train_test_data(
    overwrite=False
  )

  print(X_train.shape)
  print(X_test.shape)
  print(y_train[:10])

  # train_ae(X_train, X_test, y_train, y_test)
  train_convae(
    X_train,
    X_test,
    y_train,
    y_test)
