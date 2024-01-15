import os
import tensorflow as tf

class Config:
    def __init__(self):
        # CNN CONFIG
        self.batch_size = 16
        self.input_shape = (128, 128, 3)
        self.input_shape_crop_or_pad = (128, 128, 3)
        self.alpha = 1e-6
        self.epoch = 50

        # METHODS
        self.methods_names = ['MobileNetV2', 'DenseNet201', 'InceptionV3']

        #Make model
        self.freeze_percentage = [0.3]

        #dense num
        self.dense_num = [128]

        #drop out
        self.drop_out = [0.3]

        # datasets images percentage
        self.data_set_names = ['Dataset01_100', 'Dataset01_95.0',  'Dataset01_90.0',  'Dataset01_85.0' , 'Dataset01_80.0',
                                 'Dataset01_75.0',  'Dataset01_70.0',  'Dataset01_65.0',  'Dataset01_60.0',  'Dataset01_55.0',
                                  'Dataset01_50.0',  'Dataset01_45.0',  'Dataset01_40.0',  'Dataset01_35.0',  'Dataset01_30.0',
                                   'Dataset01_25.0',  'Dataset01_20.0',  'Dataset01_15.0',  'Dataset01_10.0',  'Dataset01_5.0']
        
        #nClasses
        self.n_classes = 2

    def get_cnn_config(self) -> dict:
        """Returns the configuration for the CNN."""
        return {
            "batch_size": self.batch_size,
            "input_shape": self.input_shape,            
            "input_shape_crop_or_pad": self.input_shape_crop_or_pad,
            "alpha": self.alpha,              
            "epoch": self.epoch       
        }
    
    @staticmethod
    def start_gpu_config():
        """Configures the GPU settings for TensorFlow."""
        config = tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(allow_growth=True))
        sess = tf.compat.v1.Session(config=config)
        tf.config.list_physical_devices('GPU')
        os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'false'
        os.environ['CUDA_VISIBLE_DEVICES'] = "0"
        return config, sess

    def get_methods_run(self) -> list[str]:
        """Returns the list of methods to run."""
        return self.methods_names
    
    def get_freeze_percentage(self):
        return self.freeze_percentage

    def get_dense_num(self):
        return self.dense_num
    
    def get_drop_out(self):
        return self.drop_out
    
    def get_data_sets(self):
        return self.data_set_names
    
    def get_n_classes(self):
        return self.n_classes