from operator import index
from this import d
import tensorflow as tf
from keras import backend as K
import numpy as np
from PIL import Image
import pandas as pd
import csv
import glob
import time
import datetime
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
import math
from collections import Counter

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.utils import to_categorical
#pip install imbalanced-learn
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

from sklearn.metrics import roc_curve, auc, roc_auc_score, confusion_matrix
from tensorflow.keras.models import load_model

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

tf.config.list_physical_devices('GPU')

# gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.7)

# tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))

import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'false'
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

##Variaveis globais
save_metrics_path = "C:/malaria/Jonathan/Metrics/Dataset01/RGB/csvs/"
save_csvs_path = "C:/malaria/Jonathan/Metrics/Dataset01/RGB/csvs/"
save_nets_path = "C:/malaria/Jonathan/Metrics/Dataset01/RGB/nets/"
base_path_parts = "C:/malaria/Jonathan/Partitions/Dataset01/RGB/"
files_parts = os.listdir(base_path_parts)
runtimeTrain = 0.0
runtimeTest = 0.0

# , 'VGG16', 'VGG19', 'ResNet50', 'ResNet50V2', 'ResNet101', 'ResNet101V2', 'ResNet152', 'ResNet152V2', \
#     , 'InceptionResNetV2', , 'MobileNetV2', 'DenseNet121','DenseNet169', , 'EfficientNetB0', 'EfficientNetB1', 

#    ,  \
# 'EfficientNetV2B0', 'EfficientNetV2B1', 'EfficientNetV2B2', 'EfficientNetV2B3', 'EfficientNetV2S', 'EfficientNetV2M', 'EfficientNetV2L', \
# 'EfficientNetV2B0', 'EfficientNetB2', 'EfficientNetB3', , 'EfficientNetB5', 'EfficientNetB6', 'EfficientNetB7'



# Refinar:  
# 'DenseNet201':    Dense 356 Dropout 0.4  Freeze 0.2
# 'InceptionV3':    Dense 256 Dropout 0.2  Freeze 0.3 
# 'MobileNetV2':    Dense 512 Dropout 0.3  Freeze 0.35
# 'MobileNetV3':    Dense 512 Dropout 0.3  Freeze 0.35
# 'Xception':       Dense 512 Dropout 0.4  Freeze 0.2
# 'ResNet101V2':    Dense 128 Dropout 0.3  Freeze 0.2 
# 'EfficientNetB4': Dense 128 Dropout 0.2  Freeze 0.4


# 'ResNet50V2':     Dense     Dropout   Freeze 
# 'MobileNet':      Dense    Dropout   Freeze 
# 'ResNet152V2':    Dense     Dropout   Freeze 


methodsNames = ["MobileNetV2"] 
# 'Resnet101V2', 'ResNet50V2', 'ResNet152V2', 'MobileNet'
# 'VGG16', 'VGG19', 'ResNet50', 'ResNet50V2', 'ResNet101', 'ResNet101V2', 'ResNet152', 'ResNet152V2', 'DenseNet201', 'Xception', 'EfficientNetB4'


##Parametros da CNNs
batch_size              = 32
input_shape             = (128, 128, 3)
input_shape_crop_or_pad = (128, 128, 3)
alpha                   = 1e-5
epoch                   = 100

lr_reduce = ReduceLROnPlateau(monitor='val_accuracy', factor=0.1, min_delta=alpha, patience=3, verbose=0)
early = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=5, mode='max')

def recall(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall_keras = true_positives / (possible_positives + K.epsilon())
    return recall_keras


def precision(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision_keras = true_positives / (predicted_positives + K.epsilon())
    return precision_keras

def specificity(y_true, y_pred):
    tn = K.sum(K.round(K.clip((1 - y_true) * (1 - y_pred), 0, 1)))
    fp = K.sum(K.round(K.clip((1 - y_true) * y_pred, 0, 1)))
    return tn / (tn + fp + K.epsilon())

def f1_score(y_true, y_pred):
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    return 2 * ((p * r) / (p + r + K.epsilon()))

# Netavie Predictive Error
def npv(y_true, y_pred):
    tn = K.sum(K.round(K.clip((1 - y_true) * (1 - y_pred), 0, 1)))
    fn = K.sum(K.round(K.clip(y_true * (1 - y_pred), 0, 1)))
    return tn / (tn + fn + K.epsilon())

# Matthews Correlation_Coefficient
def mcc(y_true, y_pred):
    tp = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    tn = K.sum(K.round(K.clip((1 - y_true) * (1 - y_pred), 0, 1)))
    fp = K.sum(K.round(K.clip((1 - y_true) * y_pred, 0, 1)))
    fn = K.sum(K.round(K.clip(y_true * (1 - y_pred), 0, 1)))

    num = tp * tn - fp * fn
    den = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    return num / K.sqrt(den + K.epsilon())



METRICS = [
    "accuracy",
    tf.keras.metrics.Precision(name='precision'),
    tf.keras.metrics.Recall(name='recall'),
    specificity,
    f1_score,
    npv,
    mcc,
    # tf.keras.metrics.AUC(name='auc',  curve='ROC'),
    tf.keras.metrics.TruePositives(name='tp'),
    tf.keras.metrics.FalsePositives(name='fp'),
    tf.keras.metrics.TrueNegatives(name='tn'),
    tf.keras.metrics.FalseNegatives(name='fn'), 
]


   
#Funções importantes
def calculateMeasures(history_net, folder, methodName):
    metrics = pd.DataFrame()
    idx = np.argmax(history_net.history['val_accuracy'])

    metrics['folder'] = [folder]
    metrics['network'] = [methodName]
    
    # TEST RESULTS
    metrics['val_accuracy'] = history_net.history['val_accuracy'][idx]
    metrics['val_precision'] = history_net.history['val_precision'][idx]
    metrics['val_sensitivity'] = history_net.history['val_recall'][idx]
    metrics['val_specificity'] = history_net.history['val_specificity'][idx]
    metrics['val_f1_score'] = history_net.history['val_f1_score'][idx]
    metrics['val_npv'] = history_net.history['val_npv'][idx]
    metrics['val_mcc'] = history_net.history['val_mcc'][idx]
    metrics['val_auc'] = history_net.history['val_auc'][idx]
    metrics['val_tn'] = history_net.history['val_tn'][idx]
    metrics['val_fp'] = history_net.history['val_fp'][idx]
    metrics['val_fn'] = history_net.history['val_fn'][idx]
    metrics['val_tp'] = history_net.history['val_tp'][idx]
    metrics['val_runtime'] = [runtimeTest]

    print(bcolors.FAIL + 'ACC: %.2f' %(100*metrics['val_accuracy'][0]) + ' AUC: %.2f' %(100*metrics['val_auc'][0]) + bcolors.ENDC)

    refined = '_NovosCasos'

    if os.path.exists(os.path.join(save_csvs_path, methodName + refined + '.csv')):
        metrics.to_csv(os.path.join(save_csvs_path, methodName + refined + '.csv'), sep=',', mode='a', index=False, header=False)
    else:
        metrics.to_csv(os.path.join(save_csvs_path, methodName + refined + '.csv'), sep=',', index=False)  


def select_image(filename):
    image = Image.open(filename) # load image from file
    image = np.asarray(image.convert('RGB')) # convert to RGB, if this option needed
    image = tf.image.resize_with_crop_or_pad(image, input_shape_crop_or_pad[0],input_shape_crop_or_pad[1]) # DEixa imagem quadrada 
    image = tf.image.resize(image, [input_shape[0], input_shape[1]]) #resize image to 
    return np.asarray(image)

def load_dataset(base_path):
    base_path.replace("D:/", "C:/")
    imagens, labels = list(), list()
    classes = os.listdir(base_path)
    for c in classes:
        for p in glob.glob(os.path.join(base_path, c, '*.bmp')):
            imagens.append(p)
            labels.append(c)
    
    return np.asarray(imagens), labels

def load_dataset_part(folder_name):
    test_x, test_y = load_dataset(os.path.join(folder_name, "test"))
    
    return test_x, test_y


def laod_balance_class_parts(folder_name):
    test_x, test_y = load_dataset_part(folder_name)

    
    ##Balanceamento dos dados de test
    undersample = RandomUnderSampler(sampling_strategy='majority')
    test_under_X, test_under_Y = undersample.fit_resample(test_x.reshape(-1,1), test_y)
    test_under_X = [select_image(p[0]) for p in test_under_X]
    test_under_X = np.array(test_under_X)/255.0
    
    lb = LabelBinarizer()
    test_under_Y = lb.fit_transform(test_under_Y)
    test_under_Y = to_categorical(test_under_Y)
    
    return test_under_X, test_under_Y 

def load_dataset(partition):
    TestImages, TestLabels = list(), list()

    fName = os.path.join(base_path_parts, partition + '.csv')

    # Read Partition Details
    with open(fName, 'r') as file:
        csvreader = csv.reader(file,delimiter=',')
        header = next(csvreader)
        #         IDX         0        1        2        3
        # print(header) # ['Image', 'Class', 'Train', 'Test']
        for row in csvreader:
            # Test Sample
            if (row[3] == '1'):
                TestImages.append(str(row[0]))
                TestLabels.append(row[1])

    TestImages = np.asarray(TestImages)


    #Balanceamento dos dados de test
    TestImages = [select_image(p) for p in TestImages]
    TestImages = np.array(TestImages)/255.0
    # print(TestImages)
    
    lb = LabelBinarizer()
    TestLabels  = to_categorical( lb.fit_transform( np.array(TestLabels ) ), num_classes=2)
    
    return TestImages , TestLabels

def makemodel(folder, methodName, denseNum, dropOut, freezePercentage):
    # create the base pre-trained model
    base_model = eval("tf.keras.applications." + methodName + "(weights='imagenet', include_top=False, input_shape=input_shape)")
    
    # Congela 50% do total de layers da rede
    numLayersFreeze = math.floor(len(base_model.layers)*freezePercentage)
    for layer in base_model.layers[:numLayersFreeze]:
        layer.trainable =  False
    
    model = tf.keras.models.Sequential()
    model.add(base_model)
    model.add(tf.keras.layers.GlobalAveragePooling2D())
    model.add(tf.keras.layers.Dense(denseNum, activation='relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dropout(dropOut))
    model.add(tf.keras.layers.Dense(2, activation='sigmoid'))
    model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9), loss='binary_crossentropy', metrics=METRICS) # 
    
    for layer in model.layers[numLayersFreeze:]:
        layer.trainable =  True

    # folder + '_' +
    fname = methodName + '/' +  methodName + '_weights' + partition + '.hdf5'
    filepath= os.path.join(save_nets_path, fname)
    checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=0, save_best_only=True, mode='max')
    return model, checkpoint




def predict_test(pred, test_under_Y, idx, folders, methodName):
    #y_pred_inception = np.array([np.argmax(p) for p in pred_inception.ravel()])
    y_pred_inception = np.argmax(pred, axis=1)
    y_true = np.argmax(test_under_Y, axis=1)
    y_pred_proba_inception = np.array([np.max(p) for p in pred.ravel()])
    
    #fpr_inception, tpr_inception, _ = roc_curve(test_under_Y.ravel(), y_pred_proba_inception.ravel())
    data_inception = pd.DataFrame()
    print(y_true.ravel())
    print(y_pred_inception.ravel())
    
    tp, fn, fp, tn = confusion_matrix(y_true.ravel(), y_pred_inception.ravel(), labels=[0,1]).ravel()

    data_inception = pd.DataFrame(columns=["Partition", "IDX", "TP", "TN", "FN", "FP", "Accuracy", "Precision", "Recall", "Specificity"])
    # auc_incp = roc_auc_score(test_under_Y.ravel(), y_pred_proba_inception.ravel())
    
    data_inception["Partition"] = [folders]
    data_inception["IDX"] = [idx]
    data_inception["TP"] = [tp]
    data_inception["TN"] = [tn]
    data_inception["FN"] = [fn]
    data_inception["FP"] = [fp]

    data_inception["Recall"] = [tp/(tp+fn)]
    data_inception["Precision"] = [tp/(tp+fp)]
    data_inception["Accuracy"] = [(tn+tp)/(tn+fp+tp+fn)]
    data_inception["Specificity"] = [tn/(tn+fp)]
    
    
    # data_inception["AUC"] = [auc_incp]
    
    if os.path.exists(os.path.join(save_csvs_path, methodName, folders + '_cm.csv')):
        data_inception.to_csv(os.path.join(save_csvs_path, methodName,  folders + '_cm.csv'), sep=',', mode='a', index=False, header=False)
    else:
        data_inception.to_csv(os.path.join(save_csvs_path, methodName,  folders + '_cm.csv'), sep=',', index=False)
        
    # data_mobile_fpr_tpr = pd.DataFrame(columns=["FPR", "TPR"])
    # data_mobile_fpr_tpr["FPR"], data_mobile_fpr_tpr["TPR"], _ = roc_curve(test_under_Y.ravel(), y_pred_proba_inception.ravel())
    
    # data_mobile_fpr_tpr.to_csv(os.path.join(save_csvs_path, methodName, '%s_fpr_tpr_NovosCasos.csv'%(str(folders))), sep=',', index=False)



if __name__ == '__main__':
    for method in range(0, len(methodsNames)):
        methodName = "MobileNetV2"
        parts = {"casos10101", "casos10102", "casos10103", "casos10104", "casos10105", "casos10106", "casos10107", "casos10108",
        "casos10109", "casos10110", "casos10120221215_085616", "casos10120221215_090507", "casos10120221215_090701", "casos10120221215_090939",
         "casos10120221215_091150", "casos10120221215_091812", "casos10120221215_091901", "casos10120221215_092447", "casos10120221215_093242",
         "casos10120221215_093352", "casos10120221215_093643", "casos10120221215_093750", "casos10120221215_093818",
         "casos10120221215_094154", "casos10120221215_094408", "casos10120221215_094550", "casos10120221215_094626", "casos10120221215_094746"}

        for partition in parts:
            if not os.path.exists(os.path.join(save_csvs_path, methodName)):
                os.mkdir(os.path.join(save_csvs_path, methodName))

            log_dir = save_csvs_path + "/" + methodName + "/" + "log_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
            
            test_under_X, test_under_Y = load_dataset(partition)

            print(bcolors.OKCYAN + "Testing " + methodName + bcolors.ENDC)

            dependencies = {
                                'precision': precision,
                                'recall': recall,
                                'f1_score': f1_score,
                                'specificity': specificity,
                                'npv': npv,
                                'mcc': mcc
                            }

            for idx in range(1, 101):
                print(idx)
                fname = methodName + '/' +  methodName + '_weights' + str(idx) + '.hdf5'
                filepath= os.path.join(save_nets_path, fname)
                model = load_model(filepath, custom_objects=dependencies)
                start_test = time.time()
                pred = model.predict(test_under_X)
                predict_test(pred, test_under_Y, idx, partition, methodName)