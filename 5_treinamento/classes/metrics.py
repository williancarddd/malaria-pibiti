import tensorflow as tf
import pandas as pd
import numpy as np
import os
from keras import backend as K
from classes.paths import Paths
from classes.colors import bcolors


"""
Verdadeiro Positivo (TP): Valores que são realmente positivos e positivos previstos .
Falso Positivo (FP): Valores que são realmente negativos , mas previstos como positivos .
Falso Negativo (FN): Valores que são realmente positivos , mas previstos como negativos.
Verdadeiro Negativo (TN): Valores que são realmente negativos e previstos como negativos.

Taxa de verdadeiro positivo (TPR): verdadeiro positivo/positivo

Taxa de falso positivo (FPR): falso positivo/negativo

Taxa de falso negativo (FNR): falso negativo/positivo

Taxa Verdadeira Negativa (TNR): Verdadeiro Negativo/Negativo

"""
class Metrics:
  #functions metrics personalized
    def __init__(self, paths: Paths, bgColors: bcolors) -> None:
        self.paths = paths
        self.bColors = bgColors

        self.METRICS = [
            'accuracy',
            tf.keras.metrics.TruePositives(name='TP', thresholds=0.5),
            tf.keras.metrics.FalsePositives(name='FP', thresholds=0.5),
            tf.keras.metrics.TrueNegatives(name='TN', thresholds=0.5),
            tf.keras.metrics.FalseNegatives(name='FN',thresholds=0.5),
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall'),
            self.specificity,
            self.f1_score,
            self.npv,
            self.mcc,
            tf.keras.metrics.AUC(name='auc', curve='ROC')
        ]
        
    def recall(self, y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall_keras = true_positives / (possible_positives + K.epsilon())
        return recall_keras


    def precision(self, y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision_keras = true_positives / (predicted_positives + K.epsilon())
        return precision_keras

    def specificity(self, y_true, y_pred):
        tn = K.sum(K.round(K.clip((1 - y_true) * (1 - y_pred), 0, 1)))
        fp = K.sum(K.round(K.clip((1 - y_true) * y_pred, 0, 1)))
        return tn / (tn + fp + K.epsilon())

    def f1_score(self, y_true, y_pred):
        p = self.precision(y_true, y_pred)
        r = self.recall(y_true, y_pred)
        return 2 * ((p * r) / (p + r + K.epsilon()))

    # Netavie Predictive Error
    def npv(self, y_true, y_pred):
        tn = K.sum(K.round(K.clip((1 - y_true) * (1 - y_pred), 0, 1)))
        fn = K.sum(K.round(K.clip(y_true * (1 - y_pred), 0, 1)))
        return tn / (tn + fn + K.epsilon())

    # Matthews Correlation_Coefficient
    def mcc(self, y_true, y_pred):
        tp = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        tn = K.sum(K.round(K.clip((1 - y_true) * (1 - y_pred), 0, 1)))
        fp = K.sum(K.round(K.clip((1 - y_true) * y_pred, 0, 1)))
        fn = K.sum(K.round(K.clip(y_true * (1 - y_pred), 0, 1)))

        num = tp * tn - fp * fn
        den = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
        return num / K.sqrt(den + K.epsilon())
    
    def calculateMeasures(self, history_net, folder, methodName, denseNum, dropOut, freezePercentage, batchsize, runtimeTrain, runtimeTest):
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

        print(self.bColors.FAIL + 'ACC: %.2f' %(100*metrics['val_accuracy'][0]) + ' AUC: %.2f' %(100*metrics['val_auc'][0]) + bcolors.ENDC)

        refined = self.paths.get_refined_path()
        rName = os.path.join(refined, methodName + "_refined" + ".csv" )
        metrics.to_csv(rName, sep=',', index=False)  


    def get_metrics(self):
        return  self.METRICS