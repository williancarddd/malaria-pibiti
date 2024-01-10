import tensorflow as tf
import csv
import numpy as np
from tensorflow.keras.utils import to_categorical
from PIL import Image
from pathlib import Path
from classes.paths import Paths
from config.config import Config

class DataSetProcessor:
    def __init__(self, paths: Paths, config: Config):
        self.config = config
        self.path = paths.get_project_paths()

    def load_dataset(self, partition: str, data_set_name:str, n_classes: int):
        """
        Load and preprocess dataset.
        """
        TrainImages, TestImages, TrainLabels, TestLabels = list(), list(), list(), list()

        fName = self.path['partitions'] / f"{partition}.csv"
        try:
            with open(fName, 'r') as file:
                csvreader = csv.reader(file)
                next(csvreader)  # Skip header if present
                for row in csvreader:
                    if row[2] == 'True':   # Train sample
                        TrainImages.append(row[0])
                        TrainLabels.append(int(row[1]))
                    elif row[3] == 'True': # Test sample
                        TestImages.append(row[0])
                        TestLabels.append(int(row[1]))

            TrainImages = [self.select_image(p, data_set_name) for p in TrainImages]
            TestImages = [self.select_image(p, data_set_name) for p in TestImages]
            print(sum(TrainLabels))
            print(sum(TestLabels))
            TrainLabels = to_categorical(np.array(TrainLabels), num_classes=n_classes)
            TestLabels = to_categorical(np.array(TestLabels), num_classes=n_classes)

            return np.array(TrainImages)/255.0, TrainLabels, np.array(TestImages)/255.0, TestLabels

        except Exception as e:
            print(f"Error processing dataset: {e}")
            return None, None, None, None

    def select_image(self, filename, data_set_name):
        """
        Select and preprocess a single image.
        """
        filename = Path(self.path['project']) / '1_entrada' / f"{data_set_name}/{str(filename)}"
        try:
            image = Image.open(filename)
            image = np.asarray(image.convert('RGB'))
            image = tf.image.resize_with_crop_or_pad(image, self.config.get_cnn_config()['input_shape_crop_or_pad'][0], self.config.get_cnn_config()['input_shape_crop_or_pad'][1])
            image = tf.image.resize(image, [self.config.get_cnn_config()['input_shape'][0], self.config.get_cnn_config()['input_shape'][1] ])
            return np.asarray(image) 
        except IOError:
            print(f"Error opening {filename}.")
            return None
  