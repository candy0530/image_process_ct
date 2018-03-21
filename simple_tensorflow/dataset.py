import cv2
import os
import glob
from sklearn.utils import shuffle
import numpy as np


def load_train(train_path, image_size, classes):
    images = []
    labels = []
    img_names = []
    cls = []

    minus_number = {'01': 143, '02': 145, '03': 144, '04': 143, '05': 139, '06': 140, '07': 162, '08': 161, '09': 162, '10':145 , '11': 169, '12': 171, '13': 129, '14': 150, '15': 150, '16': 149, '17': 160, '18': 160, '19': 155, '20': 146, '21': 142, '22': 139, '23': 142, '24': 135, '25': 134, '26': 155, '27': 147, '28': 150, '29': 142, '30': 144}

    print('Going to read training images')
    for fields in classes:
        index = classes.index(fields)
        print('Now going to read {} files (Index: {})'.format(fields, index))
        path = os.path.join(train_path, fields, '*g')
        files = glob.glob(path)
        for fl in files:
            image = cv2.imread(fl, 0)
            image = cv2.resize(image, (image_size, image_size), 0, 0, cv2.INTER_LINEAR)
            image = image.astype(np.float32)
            # minus gray average of 14
            image = np.subtract(image, minus_number['14'])
            image = np.multiply(image, 1.0 / 255.0)
            image = np.reshape(image, (15, 15, 1))
            images.append(image)
            label = np.zeros(len(classes))
            label[index] = 1.0
            labels.append(label)
            flbase = os.path.basename(fl)
            img_names.append(flbase)
            cls.append(fields)
    images = np.array(images)
    labels = np.array(labels)
    img_names = np.array(img_names)
    cls = np.array(cls)

    return images, labels, img_names, cls


class DataSet(object):

  def __init__(self, images, labels, img_names, cls):
    self._num_examples = images.shape[0]

    self._images = images
    self._labels = labels
    self._img_names = img_names
    self._cls = cls
    self._epochs_done = 0
    self._index_in_epoch = 0

  @property
  def images(self):
    return self._images

  @property
  def labels(self):
    return self._labels

  @property
  def img_names(self):
    return self._img_names

  @property
  def cls(self):
    return self._cls

  @property
  def num_examples(self):
    return self._num_examples

  @property
  def epochs_done(self):
    return self._epochs_done

  def next_batch(self, batch_size):
    """Return the next `batch_size` examples from this data set."""
    start = self._index_in_epoch
    self._index_in_epoch += batch_size

    if self._index_in_epoch > self._num_examples:
      # After each epoch we update this
      self._epochs_done += 1
      start = 0
      self._index_in_epoch = batch_size
      assert batch_size <= self._num_examples
    end = self._index_in_epoch

    return self._images[start:end], self._labels[start:end], self._img_names[start:end], self._cls[start:end]


def read_train_sets(train_path, image_size, classes):
  class DataSets(object):
    pass
  data_sets = DataSets()

  images, labels, img_names, cls = load_train(train_path, image_size, classes)
  images, labels, img_names, cls = shuffle(images, labels, img_names, cls)

  train_images = images[:]  # len(images)*80//100]  # validation_size:]
  train_labels = labels[:]  # len(images)*80//100]  # validation_size:]
  train_img_names = img_names[:]  # len(images)*80//100]  # validation_size:]
  train_cls = cls[:]  # len(images)*80//100]  # validation_size:]

  data_sets.train = DataSet(train_images, train_labels, train_img_names, train_cls)

  return data_sets
