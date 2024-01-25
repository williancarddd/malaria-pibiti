import os
import random
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import tensorflow as tf
from keras import backend as K

class Metrics:
  
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
    
    







# Caminhos dos arquivos e diretórios
path_project = Path().absolute().parent
weight_file = path_project / "6_resultados" / "Dataset01_100" / "MobileNetV2" / "metrics" / "_weigths" / "MobileNetV2.hdf5"
data_dir = path_project / "1_entrada" / "Dataset01_95.0" / "images"

# Função para classificar imagem
def classificar_imagem(fname):
    # Abre a imagem
    imagem_teste = Image.open(data_dir / fname)
    plt.imshow(imagem_teste)
    plt.show()
    # Redimensiona e prepara a imagem
    imagem_teste = imagem_teste.resize((128, 128))
    imagem_teste = np.array(imagem_teste) / 255.0
    imagem_teste = np.expand_dims(imagem_teste, axis=0)  # Adiciona uma dimensão para o batch

    # Carrega o modelo
    dependencies = {
                    'precision': Metrics.precision,
                    'recall': Metrics.recall,
                    'f1_score': Metrics.f1_score,
                    'specificity': Metrics.specificity,
                    'npv': Metrics.npv,
                    'mcc': Metrics.mcc
                }
    classificador = tf.keras.models.load_model(weight_file, custom_objects=dependencies)

    # Realiza a classificação
    output = classificador.predict(imagem_teste)
    predicao = np.argmax(output, axis=1)
    print(predicao, output)
    # Interpreta e exibe o resultado
    if predicao == 0:
        print('Não está infectado')
    else: 
        print('Está parasitado')
    print('___________________________')
    

# Lista de imagens disponíveis
imagens = os.listdir(data_dir)

# Seleciona uma imagem aleatoriamente
img_selecionada = random.choice(imagens)
classificar_imagem(img_selecionada)
