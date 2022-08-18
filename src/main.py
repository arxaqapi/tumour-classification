from DatasetGeneration import TMADataset


if __name__ == '__main__':

  dataset = TMADataset()
  dataset.bounding_boxes_of_tissue_samples()
  dataset.manually_assign_labels()
  dataset.set_all_spectrums(overwrite=False)

  X_train, X_test, y_train, y_test = dataset.get_train_test_data(
    overwrite=False
  )

