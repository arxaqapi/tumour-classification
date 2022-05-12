import numpy as np
from pathlib import Path

import cv2
import pickle
from pyimzml.ImzMLParser import ImzMLParser
from sklearn.model_selection import train_test_split


class TissueSample():
  def __init__(self, x1, y1, x2, y2):
    assert x1 >= 0 and x1 < 416
    assert y1 >= 0 and y1 < 311
    self.x1 = x1
    self.y1 = y1 
    self.x2 = x2 
    self.y2 = y2
    self.label = ''
  
  def set_values(self, array) -> None:
    self.values: np.ndarray = array

  def set_label(self, label: str) -> None:
    self.label = label

  def __repr__(self) -> str:
    return f'TissueSample [{self.x1=} {self.y1=} {self.x2=} {self.y2=}]'



class TMADataset:
  def __init__(self) -> None:
    self.imzml_file = Path('../data/20200729_tma_ctrl_cc-chc_sans_normalisation.imzML')
    self.ibd_file = Path('../data/20200729_tma_ctrl_cc-chc_sans_normalisation.ibd')
    
    if not self.imzml_file.exists() or (not self.ibd_file.exists()):
      raise FileNotFoundError(f"ImzMl file ({self.imzml_file.name}) or .ibd file ({self.ibd_file.name}) not found")
    
    self.parser = ImzMLParser(self.imzml_file)

    if self.parser.imzmldict['max count of pixels z'] != 1:
      raise ValueError(f'Expected a 2D array with z dimension = 1, instead got z = {self.parser.imzmldict["max count of pixels z"]}')
    # {'x': 416, 'y': 311}
    self.tma_size = {'x': self.parser.imzmldict['max count of pixels x'], 'y': self.parser.imzmldict['max count of pixels y']}
    self.mz_values = self.parser.getspectrum(0)[0]

    # Coordonnées valides (matrice creuse, 255 = valeur atteignable)
    self.valid_coordinates = self.create_index_coverage_mat(True)
    
  def is_valid_coordinate(self, x: int, y: int) -> bool:
    """
    Vérifie si une paire de coordonnées est valide, permettant de récupérer un spectre
    """
    assert x >= 0 and x < self.tma_size['x']
    assert y >= 0 and y < self.tma_size['y']
    for valid_x, valid_y, _ in self.parser.coordinates:
      
      assert valid_x - 1 >= 0 and valid_x - 1 < self.tma_size['x']
      assert valid_y - 1 >= 0 and valid_y - 1 < self.tma_size['y']

      if x == (valid_x - 1) and y == (valid_y - 1):
        return True
    return False
  
  def idx_of_mz_value(self, value: float):
    """
    Récupère l'index de la valeur m/z dans le tableau self.mz_values
    - dataset.mz_values[2389], dataset.idx_of_mz_value(810.800048828125)
    """
    return np.where(self.mz_values == value)[0][0]
  
  def create_index_coverage_mat(self, create_image_file: bool = False):
    """
    Créer la matrice avec les pixels ayant pour valeur 255 sont les pixels 
    atteignable.
    Cette fonction permet de récuperer les indexs des valeurs atteignables 
    (416, 311)
    """
    valid_coordinates = np.zeros((tuple(self.tma_size.values()))) # 416, 311
    for x, y, _ in self.parser.coordinates:
      valid_coordinates[x - 1, y - 1] = 255
    if create_image_file:
      cv2.imwrite('index_coverage.png', valid_coordinates.T)
    return valid_coordinates

  def bounding_boxes_of_tissue_samples(self):
    # (416, 311)
    assert self.valid_coordinates.shape == (416, 311)
    gray_image = self.valid_coordinates.astype(dtype=np.uint8) 
    edges = cv2.Canny(gray_image, 50, 200)
    contours, hierarchy = cv2.findContours(
      edges,
      cv2.RETR_EXTERNAL,
      cv2.CHAIN_APPROX_NONE)
    empty_im = np.zeros(self.valid_coordinates.shape + (3,))

    hulls = []
    for cnt in contours:
      hull = cv2.convexHull(cnt)
      hulls.append(hull)
      cv2.drawContours(empty_im, [hull], 0, (0, 255, 0), 1)
    
    self.tissue_samples = []
    for hull in hulls:
      coordinates = hull.reshape(hull.shape[0], hull.shape[2])
      if coordinates.shape[0] > 8:
        # NOTE: xy yx
        y1, x1 = coordinates.min(axis=0)
        y2, x2 = coordinates.max(axis=0)
        # x1, y1  = coordinates.min(axis=0)
        # x2, y2 = coordinates.max(axis=0)
        self.tissue_samples.append(TissueSample(x1=x1,y1=y1,x2=x2,y2=y2))
        # NOTE: xy yx 
        # cv2.rectangle(empty_im, (x1, y1), (x2, y2), (0, 0, 255), 1)
        cv2.rectangle(empty_im, (y1, x1), (y2, x2), (0, 0, 255), 1)
        # empty_im[x1+1:x2, y1+1:y2] = 127
        # point: 
        cv2.rectangle(empty_im, (y1, x1), (y1, x1), (0, 255, 255), 1)
    cv2.imwrite('bounding_boxes.png', empty_im)
    self.test_image = empty_im
  
  def _get_idx_in_coords(self, x, y) -> int:
    """
    Récupère l'index unique des coordonnées dans le tableau des coordonnées
    afin de récupérer le spectre
    """
    assert x >= 0 and x < self.tma_size['x']
    assert y >= 0 and y < self.tma_size['y']
    for i, (x_, y_, _) in enumerate(self.parser.coordinates):
      if (x == (x_ - 1))  and (y == (y_ - 1)):
        return i
    return -1

  def set_spectrums_of_tissue(self, sample: TissueSample):
    """
    Récupère chaque spectre de chaque point de relevé valide du tissue
    """
    intensities = []
    for x in range(sample.x1, sample.x2):
      for y in range(sample.y1, sample.y2):
        if self.is_valid_coordinate(x, y):
          _, intensity_array = self.parser.getspectrum(
            self._get_idx_in_coords(x , y))
          intensities.append(intensity_array)
          # self.test_image[x, y, :] = 255
    intensities = np.array(intensities)
    sample.set_values(intensities)
    # cv2.imwrite("hm.png", self.test_image)
    return sample.values
  
  def set_all_spectrums(self, overwrite: bool = True):
    """
    Pour chaque sample, leurs spectres sont récupérées
    """
    # NOTE: too much data in cache 
    print('Get spectrums of all samples')
    temp_path = Path('../data/temp.pkl')

    if temp_path.exists():
      with open(temp_path, 'rb') as temp_f:
        self.tissue_samples = pickle.load(temp_f)
    else:
      for sample in self.tissue_samples:
        self.set_spectrums_of_tissue(sample)
      with open(temp_path, 'wb') as temp_f:
        pickle.dump(self.tissue_samples, temp_f)

  def manually_assign_labels(self):
    # (416, 311)
    for sample in self.tissue_samples:
      # if sample.x1 > 5 and sample.x1 < 40 and sample.y1 > 250 and 
      # print(sample)
      if sample.y2 >= 308:
        print(sample)
        self.test_image[sample.x1:sample.x2+1, sample.y1:sample.y2+1] = 255   
        cv2.imwrite("hm.png", self.test_image)
        exit(1)
  
  def data_generator(self):
    for sample in self.tissue_samples:
      for point in sample.values:
        yield point, sample.label

  def get_train_test_data(self):
    dataset_path = Path('../data/dataset_extracted_20200729_tma_ctrl_cc-chc_sans_normalisation.pkl')

    if dataset_path.exists():
      print('loading dataset')
      with open(dataset_path, 'rb') as data_pkl_f:
        X_train, X_test, y_train, y_test = pickle.load(data_pkl_f)
      return X_train, X_test, y_train, y_test
  
    else:
      print('creating dataset')
      X_ = []
      y_ = []
      for x, y in self.data_generator():
        X_.append(x)
        y_.append(y)
      # print(np.array(X_).shape)
      X_train, X_test, y_train, y_test = train_test_split(X_, y_, test_size=0.2)
      X_train, X_test, y_train, y_test = np.array(X_train), np.array(X_test), np.array(y_train), np.array(y_test)
      with open(dataset_path, 'wb') as data_pkl_f:
        pickle.dump((X_train, X_test, y_train, y_test), data_pkl_f)
      return X_train, X_test, y_train, y_test


# def show_im(p, mz_value=2400):
#   im = getionimage(p, mz_value)
#   plt.imshow(im, cmap='viridis', interpolation ='nearest')
#   plt.show()

