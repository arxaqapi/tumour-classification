from keras.models import Model, Sequential
from keras.layers import (
  Dense,
  Conv2D,
  MaxPooling2D,
  UpSampling2D)
from keras.losses import mean_squared_error
from tensorflow.keras.optimizers import Adadelta


class AutoEncoder(Model):
  def __init__(self, input_dim: int = 32320, latent_dims: list[int] = [30000, 15000, 7500, 1000]):
    # 32320
    super(AutoEncoder, self).__init__(name="AutoEncoder")

    self.input_dim = input_dim
    self.latent_dims = latent_dims

    self.encoder = Sequential([
      Dense(latent_dims[0], activation='relu'),
      Dense(latent_dims[1], activation='relu'),
      Dense(latent_dims[2], activation='relu'),
      Dense(latent_dims[3], activation='relu'),
    ])
    self.decoder = Sequential([
      Dense(latent_dims[2], activation='sigmoid'),
      Dense(latent_dims[1], activation='sigmoid'),
      Dense(latent_dims[0], activation='sigmoid'),
      Dense(self.input_dim, activation='sigmoid'),
    ])
    
  def call(self, inputs):
    encoded = self.encoder(inputs)
    decoded = self.decoder(encoded)
    return decoded


class AutoEncoder(Model):
  def __init__(self, input_dim: int = 32320, latent_dims: list[int] = [30000, 15000, 7500, 1000]):
    # 32320
    super(AutoEncoder, self).__init__(name="AutoEncoder")

    self.input_dim = input_dim
    self.latent_dims = latent_dims

    self.encoder = Sequential([
      Dense(latent_dims[0], activation='relu'),
      Dense(latent_dims[1], activation='relu'),
      Dense(latent_dims[2], activation='relu'),
      Dense(latent_dims[3], activation='relu'),
    ])
    self.decoder = Sequential([
      Dense(latent_dims[2], activation='sigmoid'),
      Dense(latent_dims[1], activation='sigmoid'),
      Dense(latent_dims[0], activation='sigmoid'),
      Dense(self.input_dim, activation='sigmoid'),
    ])
    
  def call(self, inputs):
    encoded = self.encoder(inputs)
    decoded = self.decoder(encoded)
    return decoded

class ConvolutionalAutoEncoder(Model):
  def __init__(self, input_dim: int = (32320, 1, 1), latent_dims: int = [32, 16, 8]):
    super(ConvolutionalAutoEncoder, self).__init__()

    self.input_dim = input_dim
    self.latent_dims = latent_dims
    ksize = (12, 1)

    self.encoder = Sequential([
        # Block 1
        Conv2D(filters=latent_dims[0], kernel_size=ksize, padding='same', activation='relu'),
        MaxPooling2D(pool_size=(2, 1), padding='same'),
        # Block 2
        Conv2D(filters=latent_dims[1], kernel_size=ksize, padding='same', activation='relu'),
        MaxPooling2D(pool_size=(2, 1), padding='same'),
        # Block 3
        Conv2D(filters=latent_dims[2], kernel_size=ksize, padding='same', activation='relu'),
        MaxPooling2D(pool_size=(2, 1), padding='same'),

    ], name="Conv-encoder")

    self.decoder = Sequential([
        # Block 1
        UpSampling2D(size=(2, 1)),
        Conv2D(filters=latent_dims[2], kernel_size=ksize, padding='same', activation='relu'),
        # Block 2
        UpSampling2D(size=(2, 1)),
        Conv2D(filters=latent_dims[1], kernel_size=ksize, padding='same', activation='relu'),
        # Block 2
        UpSampling2D(size=(2, 1)),
        Conv2D(filters=latent_dims[0], kernel_size=ksize, padding='same', activation='relu'),
        # Final reconstruction layer
        Conv2D(filters=1, kernel_size=ksize, padding='same', activation='sigmoid'),
    ], name="Conv-decoder")

  def call(self, x):
    encoded = self.encoder(x)
    decoded = self.decoder(encoded)
    return decoded



def train_ae(X_train, X_test, y_train, y_test):
  batch_size = 8
  epochs = 100

  model = AutoEncoder()
  model.build((None, 32320))
  model.summary()

  model.compile(
    optimizer=Adadelta(learning_rate=10.0),
    loss=mean_squared_error)
  
  # history = model.fit(
  #   x=X_train,
  #   y=X_train,
  #   batch_size=batch_size,
  #   epochs=epochs,
  #   validation_split=0.2,
  #   validation_data=(X_test, X_test),
  #   shuffle=True)


def train_convae(X_train, X_test, y_train, y_test):
  X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1, 1))
  X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1, 1))

  batch_size = 8
  epochs = 100
  
  model = ConvolutionalAutoEncoder()
  model.build((None, 32320 ,1 , 1))
  model.summary()

  model.compile(
    optimizer=Adadelta(learning_rate=1.0),
    loss=mean_squared_error)
  
  history = model.fit(
    x=X_train,
    y=X_train,
    batch_size=batch_size,
    epochs=epochs,
    validation_split=0.2,
    validation_data=(X_test, X_test),
    shuffle=True)