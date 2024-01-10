import tensorflow as tf
import pandas as pd
import csv
import time
import datetime
import math
import numpy as np
import os
from PIL import Image
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix
from tensorflow.keras.models import load_model
from classes.colors import bcolors
from classes.paths  import Paths
from classes.metrics import Metrics

config = tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(allow_growth=True))
sess = tf.compat.v1.Session(config=config)
tf.config.list_physical_devices('GPU')
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'false'
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

# Variaveis globais
runtimeTrain = 0.0
runtimeTest = 0.0
paths = Paths.get_project_paths()
metrics = Metrics


methodsNames = ['MobileNetV2', 'DenseNet201', 'InceptionV3']


# Parametros da CNNs
batch_size              = 32
input_shape             = (128, 128, 3)
input_shape_crop_or_pad = (128, 128, 3)
alpha                   = 1e-6
epoch                   = 50

lr_reduce = ReduceLROnPlateau(monitor='val_accuracy', factor=0.1, min_delta=alpha, patience=3, verbose=0)
early = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=5, mode='max')


METRICS = [
    "accuracy",
    tf.keras.metrics.TruePositives(name='TP',thresholds=0.5),
    tf.keras.metrics.FalsePositives(name='FP',thresholds=0.5),
    tf.keras.metrics.TrueNegatives(name='TN',thresholds=0.5),
    tf.keras.metrics.FalseNegatives(name='FN',thresholds=0.5), 
    tf.keras.metrics.Precision(name='precision'),
    tf.keras.metrics.Recall(name='recall'),
    metrics.pecificity,
    metrics.f1_score,
    metrics.npv,
    metrics.mcc,
    tf.keras.metrics.AUC(name='auc',  curve='ROC'),
]

#Calculations meansures
def calculateMeasures(history_net, folder, methodName, denseNum, dropOut, freezePercentage, batchsize):
    metrics = pd.DataFrame()
    idx = np.argmax(history_net.history['val_accuracy'])

    metrics['folder'] = [folder]
    metrics['network'] = [methodName]
    metrics['DenseNum'] = [denseNum]
    metrics['DropOut'] = [dropOut]
    metrics['FreezePercentage'] = [freezePercentage]
    metrics['BatchSize'] = [batchsize]

    # TEST RESULTS
    metrics['epoch'] = [idx]
    metrics['accuracy'] = history_net.history['accuracy'][idx]
    metrics['precision'] = history_net.history['precision'][idx]
    metrics['sensitivity'] = history_net.history['recall'][idx]
    metrics['specificity'] = history_net.history['specificity'][idx]
    metrics['f1_score'] = history_net.history['f1_score'][idx]
    metrics['npv'] = history_net.history['npv'][idx]
    metrics['mcc'] = history_net.history['mcc'][idx]
    metrics['auc'] = history_net.history['auc'][idx]
    metrics['TP'] = history_net.history['TP'][idx]
    metrics['TN'] = history_net.history['TN'][idx]
    metrics['FP'] = history_net.history['FP'][idx]
    metrics['FN'] = history_net.history['FN'][idx]
    metrics['runtime'] = [runtimeTrain]

    # TRAIN RESULTS
    metrics['val_accuracy'] = history_net.history['val_accuracy'][idx]
    metrics['val_precision'] = history_net.history['val_precision'][idx]
    metrics['val_sensitivity'] = history_net.history['val_recall'][idx]
    metrics['val_specificity'] = history_net.history['val_specificity'][idx]
    metrics['val_f1_score'] = history_net.history['val_f1_score'][idx]
    metrics['val_npv'] = history_net.history['val_npv'][idx]
    metrics['val_mcc'] = history_net.history['val_mcc'][idx]
    metrics['val_auc'] = history_net.history['val_auc'][idx]
    metrics['val_TP'] = history_net.history['val_TP'][idx]
    metrics['val_TN'] = history_net.history['val_TN'][idx]
    metrics['val_FP'] = history_net.history['val_FP'][idx]
    metrics['val_FN'] = history_net.history['val_FN'][idx]
    metrics['val_runtime'] = [runtimeTest]

    print(bcolors.FAIL + 'ACC: %.2f' %(100*metrics['val_accuracy'][0]) + ' AUC: %.2f' %(100*metrics['val_auc'][0]) + bcolors.ENDC)

    refined = '_refined'

    if os.path.exists(os.path.join(paths['csvs'], methodName + refined + '.csv')):
        metrics.to_csv(os.path.join(paths['csvs'], methodName + refined + '.csv'), sep=',', mode='a', index=False, header=False)
    else:
        metrics.to_csv(os.path.join(paths['csvs'], methodName + refined + '.csv'), sep=',', index=False)  

# load image
def select_image(filename, data_set_name):
    # print(filename)
    filename =  paths['project'] / '1_entrada' / f"{data_set_name }/{str(filename)}"
    image = Image.open(filename) # load image from file
    image = np.asarray(image.convert('RGB')) # convert to RGB, if this option needed
    image = tf.image.resize_with_crop_or_pad(image, input_shape_crop_or_pad[0],input_shape_crop_or_pad[1]) # Deixa imagem quadrada 
    image = tf.image.resize(image, [input_shape[0], input_shape[1]]) #resize image to 128x128
    #print(image.shape)
    return np.asarray(image)



def load_dataset(partition, data_set_name):
    TrainImages, TestImages, TrainLabels, TestLabels = list(), list(), list(), list()

    fName = partition['partitions'] / f"{partition}.csv"
    print(fName)

    # Read Partition Details
    with open(fName, 'r') as file:
        csvreader = csv.reader(file)
        header = next(csvreader)
        #         IDX         0        1        2        3
        # print(header) # ['Image', 'Class', 'Train', 'Test']
        # print(header)
        for row in csvreader:
            # print(row)
            # image = np.asarray(np.array(select_image(row[0]))/255.0)
            # print(len(image))
            # Train sample
            if (row[2] == 'True'):
                TrainImages.append(row[0])
                TrainLabels.append(int(row[1]))
            # Test Sample
            elif (row[3] == 'True'):
                TestImages.append(row[0])
                TestLabels.append(int(row[1]))

    TrainImages = np.asarray(TrainImages)
    TestImages = np.asarray(TestImages)

    TrainImages = [select_image(p, data_set_name) for p in TrainImages]
    TrainImages = np.array(TrainImages)/255.0


    TestImages = [select_image(p, data_set_name) for p in TestImages]
    TestImages = np.array(TestImages)/255.0

    print(sum(TrainLabels))
    print(sum(TestLabels))
    
    TrainLabels = to_categorical( np.array(TrainLabels), num_classes=7)
    TestLabels  = to_categorical( np.array(TestLabels ), num_classes=7)
    
   
    return TrainImages, TrainLabels, TestImages , TestLabels

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
    model.add(tf.keras.layers.Dense(7, activation='softmax'))
    model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9), loss='binary_crossentropy', metrics=METRICS)

    for layer in model.layers[numLayersFreeze:]:
        layer.trainable =  True

    filepath = paths['nets'] / methodName /  '_weights' / f"{partition}.hdf5"
    print(filepath)
    checkpoint = ModelCheckpoint(str(filepath), monitor='val_accuracy', verbose=0, save_best_only=True, mode='max')
    return model, checkpoint


def predict_test(pred, test_under_Y, folders, methodName):
    y_pred_inception = np.argmax(pred, axis=1)
    y_true = np.argmax(test_under_Y, axis=1)
    y_pred_proba_inception = np.array([np.max(p) for p in pred.ravel()])
    
    data_inception = pd.DataFrame()
    
    tn, fp, fn, tp = confusion_matrix(y_true.ravel(), y_pred_inception.ravel(), labels=[0,1]).ravel()

    data_inception = pd.DataFrame(columns=["TN", "FP", "FN", "TP"])
    auc_incp = roc_auc_score(test_under_Y.ravel(), y_pred_proba_inception.ravel())
    
    data_inception["TN"] = [tn]
    data_inception["FP"] = [fp]
    data_inception["FN"] = [fn]
    data_inception["TP"] = [tp]
    data_inception["AUC"] = [auc_incp]
    
    if os.path.exists(os.path.join(paths['csvs'], methodName, '_refined_cm.csv')):
        data_inception.to_csv(os.path.join(paths['csvs'], methodName,  '%s_cm_refined.csv'%(str(folders))), sep=',', mode='a', index=False, header=False)
    else:
        data_inception.to_csv(os.path.join(paths['csvs'], methodName, '%s_cm_refined.csv'%(str(folders))), sep=',', index=False)
        
    data_mobile_fpr_tpr = pd.DataFrame(columns=["FPR", "TPR"])
    data_mobile_fpr_tpr["FPR"], data_mobile_fpr_tpr["TPR"], _ = roc_curve(test_under_Y.ravel(), y_pred_proba_inception.ravel())
    
    data_mobile_fpr_tpr.to_csv(os.path.join(paths['csvs'], methodName, '%s_fpr_tpr_refined.csv'%(str(folders))), sep=',', index=False)



if __name__ == '__main__':
    for method in range(0, len(methodsNames)):
        methodName = methodsNames[method]

        if not os.path.exists(paths['csvs'] / methodName):
            os.mkdir(paths['csvs'] / methodName) 
            os.makedirs(paths['csvs']/ methodName /  '_weights') 

        cont = 1

        for partition in range(1, 101):
            for denseNum in [128]: # range(128,128, 128):
                for dropOut in [0.3]: #0.2,0.3,0.4,0.5
                    for freezePercentage in [0.3]: # 
                        print(bcolors.OKGREEN + f"{methodName}: Partition {partition} DenseNum {denseNum}, dropout {dropOut}, freezePercentage {freezePercentage} " + bcolors.ENDC)

                        log_dir = paths['csvs']  /  methodName /  "log_" / datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
                        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

                        csv_logger = tf.keras.callbacks.CSVLogger(paths['csvs'] /  methodName / f"{partition}-epoch-results.csv", separator=',', append=True)
                    
                        
                        initModel, checkpoint = makemodel(partition, methodName, denseNum, dropOut, freezePercentage)
                        
                        # train_under_X, train_under_Y, test_under_X, test_under_Y = laod_balance_class_parts(os.path.join(base_path_parts, partition))
                        train_under_X, train_under_Y, test_under_X, test_under_Y = load_dataset(partition, 'Dataset01_95.0')

                        
                        print(bcolors.OKCYAN + "Trainning " + methodName + bcolors.ENDC)
                        start_train = time.time()
                        with tf.device('/device:GPU:0'):
                            history_net = initModel.fit(train_under_X,
                                                    train_under_Y,
                                                    steps_per_epoch=(len(train_under_X) // batch_size),
                                                    validation_steps = (len(test_under_X) // batch_size),
                                                    batch_size = batch_size,
                                                    epochs=epoch, 
                                                    validation_data=(test_under_X, test_under_Y), 
                                                    callbacks=[early, checkpoint, lr_reduce,csv_logger])
                        
                        runtimeTrain = time.time() - start_train
                        print(bcolors.OKCYAN + methodName + " Trained in %2.2f seconds"%(runtimeTrain) + bcolors.ENDC)
                        

                        dependencies = {
                            'precision': metrics.precision,
                            'recall': metrics.recall,
                            'f1_score': metrics.f1_score,
                            'specificity': metrics.specificity,
                            'npv': metrics.npv,
                            'mcc': metrics.mcc
                        }

                        print(bcolors.OKCYAN + "Testing " + methodName + bcolors.ENDC)
                        # 
                
                        filepath = paths['csvs'] /  methodName / '_weights' / f"{partition}.hdf5"
                        model = load_model(filepath, custom_objects=dependencies)
                        start_test = time.time()
                        pred = model.predict(test_under_X)
                        runtimeTest = time.time() - start_test
                        print(bcolors.OKCYAN + methodName + " Tested in %2.2f seconds"%(runtimeTest) + bcolors.ENDC)
                        print("Completed: %s Percent"%(partition) + bcolors.ENDC)

                        predict_test(pred, test_under_Y, partition, methodName)
                    
                        calculateMeasures(history_net, partition, methodName, denseNum, dropOut, freezePercentage, batch_size)
                        cont = cont + 1