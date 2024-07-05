import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage.feature import hog
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model

def extract_hog_features(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    features, hog_image = hog(img_gray, orientations=9, pixels_per_cell=(8, 8),
                              cells_per_block=(2, 2), block_norm='L2-Hys',
                              visualize=True)
    return features, hog_image

def extract_cnn_features_and_visualize(img, model):
    img_resized = cv2.resize(img, (224, 224))
    x = image.img_to_array(img_resized)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    
    # Extrair características
    features = model.predict(x)
    features = features.flatten()
    
    # Obter os mapas de ativação da última camada convolucional
    layer_outputs = [layer.output for layer in model.layers if 'conv' in layer.name]
    activation_model = Model(inputs=model.input, outputs=layer_outputs)
    activations = activation_model.predict(x)
    
    return features, activations

def extract_luminosity_features(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mean_luminosity = np.mean(img_gray)
    return [mean_luminosity]

def visualize_hog(hog_image):
    plt.figure(figsize=(8, 8))
    plt.imshow(hog_image, cmap='gray')
    plt.title('HOG Features')
    plt.show()

def visualize_cnn_activations(activations):
    for i, activation in enumerate(activations):
        # Exibir apenas a primeira característica da camada
        plt.figure(figsize=(8, 8))
        plt.imshow(activation[0, :, :, 0], cmap='viridis')
        plt.title(f'Activation of Conv Layer {i+1}')
        plt.show()

def main(image_path, output_csv):
    # Carregar imagem
    img = cv2.imread(image_path)
    
    # Extrair e visualizar características HOG
    hog_features, hog_image = extract_hog_features(img)
    print(f"HOG Features: {hog_features.shape}")
    visualize_hog(hog_image)
    
    # Carregar modelo VGG16 pré-treinado
    model = VGG16(weights='imagenet', include_top=False)
    
    # Extrair e visualizar características CNN
    cnn_features, activations = extract_cnn_features_and_visualize(img, model)
    print(f"CNN Features: {cnn_features.shape}")
    visualize_cnn_activations(activations)

    # Extrair características de luminosidade
    luminosity_features = extract_luminosity_features(img)
    print(f"Luminosity Features: {luminosity_features}")
    
    # Combinar todas as características
    combined_features = np.concatenate((hog_features, cnn_features, luminosity_features))
    print(f"Combined Features: {combined_features.shape}")
    
    # Salvar características no CSV
    feature_names = [f'feature_{i}' for i in range(combined_features.shape[0])]
    df = pd.DataFrame([combined_features], columns=feature_names)
    df.to_csv(output_csv, index=False)
    print(f"Features saved to {output_csv}")
    
    return combined_features

if __name__ == "__main__":
    image_path = '/media/william/NVME/projects/malaria-pibiti/1_entrada/1-1-4-0.jpg'
    output_csv = 'features.csv'
    features = main(image_path, output_csv)
    print(f"Extracted Features: {features}")
