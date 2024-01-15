import math
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.losses import BinaryCrossentropy
from config.config import Config
from classes.paths import Paths
from classes.metrics import Metrics

class ModelBuilding:
    @staticmethod
    def create_model(nClasses: int,
                      methodName: str, 
                      denseNum: int, 
                      dropOut: int, 
                      config: Config, 
                      paths: Paths, 
                      partition: str, 
                      metrics: Metrics,
                      freezePercentage: int,
                      indiceDataSet: int):
        # Map methodName to the corresponding model constructor
        model_constructors = {
            "MobileNetV2": tf.keras.applications.MobileNetV2,
            "DenseNet201": tf.keras.applications.DenseNet201,
            "InceptionV3": tf.keras.applications.InceptionV3,
            # Add other models as needed
        }
        if methodName not in model_constructors:
            raise ValueError(f"Model {methodName} not recognized")

        base_model = model_constructors[methodName](weights='imagenet', include_top=False, input_shape=config.get_cnn_config()['input_shape'])

        numLayersFreeze = math.floor(len(base_model.layers) * freezePercentage)
        for layer in base_model.layers[:numLayersFreeze]:
            layer.trainable = False

        model = tf.keras.models.Sequential([
            base_model,
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(denseNum, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dropout(dropOut),
            tf.keras.layers.Dense(nClasses, activation='sigmoid')
        ])

        model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9),
                      loss=BinaryCrossentropy(),
                      metrics=metrics.get_metrics())

        filepath = paths.get_nets_path(partition)
        checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=0, save_best_only=True, mode='max')

        return model, checkpoint
