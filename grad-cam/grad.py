import os
import warnings
import numpy as np
import cv2
from torchvision import models
from pytorch_grad_cam import GradCAM, ScoreCAM, AblationCAM, XGradCAM, GradCAMElementWise, HiResCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
import torch
from PIL import Image

warnings.filterwarnings('ignore')

# use gpu

print("CUDA Available: ", torch.cuda.is_available())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()



os.environ["CUDA_VISIBLE_DEVICES"] = "0" 



# Função para processar uma imagem e gerar CAMs
def process_image(image_path, model, target_layers):
    img = np.array(Image.open(image_path))
    img = cv2.resize(img, (128, 128))
    img = np.float32(img) / 255
    input_tensor = preprocess_image(img)

    targets = [ClassifierOutputTarget(295)]

    # # GradCAM
    with GradCAM(model=model, target_layers=target_layers) as cam:
        grayscale_cam_grad = cam(input_tensor=input_tensor, targets=targets)
        cam_image_grad = show_cam_on_image(img, grayscale_cam_grad[0, :], use_rgb=True)

    # ScoreCAM
    with ScoreCAM(model=model, target_layers=target_layers) as cam:
        grayscale_cam_score = cam(input_tensor=input_tensor, targets=targets)
        cam_image_score = show_cam_on_image(img, grayscale_cam_score[0, :], use_rgb=True)

    with HiResCAM(model=model, target_layers=target_layers) as cam:
        grayscale_cam_ablation = cam(input_tensor=input_tensor,  targets=targets)
        cam_image_ablation = show_cam_on_image(img, grayscale_cam_ablation[0, :], use_rgb=True)

    return img, cam_image_grad, cam_image_score, cam_image_ablation

# Função para salvar as imagens combinadas em uma única imagem
def save_combined_image(image_path, img, one, two, tree):
    # Extrair o nome do arquivo original (sem extensão)
    image_name = os.path.splitext(os.path.basename(image_path))[0]

    # Combinar as imagens em uma única linha
    combined_image = np.hstack((
        np.uint8(255 * img),  # Imagem original
        np.uint8(255 * one),  # GradCAM
        np.uint8(255 * two),  # ScoreCAM
        np.uint8(255 * tree)  # AblationCAM
    ))

    # Salvar a imagem combinada
    Image.fromarray(combined_image).save(f"combined_{image_name}.bmp")

# Caminho para a imagem específica
images_dir = "/media/williancaddd/CODES/projects/malaria-pibiti/grad-cam/images-grad-test"

# Carregar o modelo



model = models.densenet201(pretrained=True)
model.eval()
target_layers = [model.features[-1]]

'''
model = models.resnet50(pretrained=True)
model.eval()
target_layers = [model.layer4[-1]]


model = models.inception_v3(pretrained=True)
model.eval()
target_layers = [model.Mixed_7c.branch_pool]

model = models.densenet201(pretrained=True)
model.eval()
target_layers = [model.features[-1]]


model = models.mobilenet_v2(pretrained=True)
model.eval()
target_layers = [model.features[-1]]


'''

# Processar a imagem e gerar as CAMs
for image_name in os.listdir(images_dir):
    image_path = os.path.join(images_dir, image_name)
    img, grad, score, ablation = process_image(image_path, model, target_layers)
    save_combined_image(image_path, img,  grad, score, ablation)

print("Done!")
