import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Função para carregar um modelo e obter as camadas convolucionais
def load_model_and_conv_layers(model_name):
    models = {
        "DenseNet201": tf.keras.applications.DenseNet201,
        "MobileNetV2": tf.keras.applications.MobileNetV2,
        "InceptionV3": tf.keras.applications.InceptionV3,
        "ResNet50": tf.keras.applications.ResNet50,
    }
    model = models[model_name](weights='imagenet', include_top=True)
    conv_layers = [layer.name for layer in model.layers if isinstance(layer, tf.keras.layers.Conv2D)]
    return model, conv_layers

# Função para pré-processar a imagem
def get_img_array(img_path, size):
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=size)
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Função para gerar o Grad-CAM para uma camada específica
def make_gradcam_heatmap(img_array, model, conv_layer_name):
    grad_model = tf.keras.models.Model(
        [model.input], 
        [model.get_layer(conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_output, preds = grad_model(img_array)
        pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, conv_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_output = conv_output[0]
    heatmap = conv_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

# Função para sobrepor o heatmap na imagem original
def overlay_heatmap_on_image(img_path, heatmap, alpha=0.4):
    img = tf.keras.preprocessing.image.load_img(img_path)
    img = tf.keras.preprocessing.image.img_to_array(img)

    heatmap = np.uint8(255 * heatmap)
    heatmap = np.expand_dims(heatmap, axis=-1)
    heatmap = tf.image.resize(heatmap, (img.shape[0], img.shape[1])).numpy()

    heatmap = plt.cm.jet(heatmap[..., 0])[:, :, :3]
    heatmap = np.uint8(255 * heatmap)
    heatmap = np.clip(heatmap * 2, 0, 255)

    superimposed_img = heatmap * alpha + img
    return np.uint8(np.clip(superimposed_img, 0, 255))

# Função para criar a animação do Grad-CAM para cada camada convolucional
def animate_gradcam(img_path, model_name):
    model, conv_layers = load_model_and_conv_layers(model_name)

    # Pré-processa a imagem para o tamanho correto
    img_size = model.input_shape[1:3]
    img_array = get_img_array(img_path, size=img_size)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.axis('off')

    # Função de atualização para a animação
    def update(frame):
        layer_name = conv_layers[frame]
        heatmap = make_gradcam_heatmap(img_array, model, layer_name)
        superimposed_img = overlay_heatmap_on_image(img_path, heatmap)

        ax.clear()
        ax.imshow(superimposed_img)
        ax.set_title(f"Layer: {layer_name}")
        ax.axis('off')

    # Criação da animação
    anim = FuncAnimation(fig, update, frames=len(conv_layers), interval=500)

    plt.show()

# Exemplo de uso
img_path = '/media/williancaddd/CODES/projects/malaria-pibiti/1_entrada/Dataset01_100/images/1-55-0-1.bmp'
animate_gradcam(img_path, "ResNet50")
