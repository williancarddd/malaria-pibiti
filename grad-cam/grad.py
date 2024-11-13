import os
import warnings
import numpy as np
import cv2
from torchvision import models
from pytorch_grad_cam import GradCAM, ScoreCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
from PIL import Image

warnings.filterwarnings('ignore')

# Função para processar uma imagem e gerar CAMs
def process_image(image_path, model, target_layers):
    img = np.array(Image.open(image_path))
    img = cv2.resize(img, (224, 224))
    img = np.float32(img) / 255
    input_tensor = preprocess_image(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    targets = [ClassifierOutputTarget(295)]

    # GradCAM
    with GradCAM(model=model, target_layers=target_layers) as cam:
        grayscale_cam_grad = cam(input_tensor=input_tensor, targets=targets)
        cam_image_grad = show_cam_on_image(img, grayscale_cam_grad[0, :], use_rgb=True)

    # ScoreCAM
    with ScoreCAM(model=model, target_layers=target_layers) as cam:
        grayscale_cam_score = cam(input_tensor=input_tensor, targets=targets)
        cam_image_score = show_cam_on_image(img, grayscale_cam_score[0, :], use_rgb=True)

    return img, cam_image_grad, cam_image_score

# Função para salvar as imagens combinadas em uma única imagem
def save_combined_image(image_path, img, cam_grad, cam_score):
    # Extrair o nome do arquivo original (sem extensão)
    image_name = os.path.splitext(os.path.basename(image_path))[0]

    # Combinar as imagens em uma única linha
    combined_image = np.hstack((
        np.uint8(255 * img),  # Imagem original
        np.uint8(255 * cam_grad),  # GradCAM
        np.uint8(255 * cam_score)  # ScoreCAM
    ))

    # Salvar a imagem combinada
    Image.fromarray(combined_image).save(f"combined_{image_name}.bmp")

# Caminho para a imagem específica
images_dir = "/media/williancaddd/CODES/projects/malaria-pibiti/grad-cam/images-grad-test"

# Carregar o modelo
model = models.densenet201(pretrained=True)
model.eval()
target_layers = [model.features[-1]]

# Processar a imagem e gerar as CAMs
for image_name in os.listdir(images_dir):
    image_path = os.path.join(images_dir, image_name)
    img, cam_grad, cam_score = process_image(image_path, model, target_layers)
    save_combined_image(image_path, img, cam_grad, cam_score)

print("Done!")
