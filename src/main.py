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
  dataset.set_all_spectrums()

  # arr = dataset.set_spectrums_of_tissue(dataset.tissue_samples[0])
  # dataset.manually_assign_labels()


  X_train, X_test, y_train, y_test = dataset.get_train_test_data()
  # X_train, X_test, y_train, y_test = np.ones((10, 32320)), np.ones((10, 32320)), None, None

  # train_ae(X_train, X_test, y_train, y_test)
  train_convae(
    X_train,
    X_test,
    y_train,
    y_test)
