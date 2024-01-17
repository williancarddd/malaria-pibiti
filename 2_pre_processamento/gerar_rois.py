import pandas as pd
from PIL import Image, ImageDraw
import numpy as np
import os
from pathlib import Path

def mask_images_with_percentage(csv_file_path, output_path, percentage):
    casos = pd.read_csv(csv_file_path)

    for i in range(len(casos)):
        pathName = casos['ImagePathName'][i]
        nomeArquivo = output_path / 'DataSetOriginal' / pathName[8:]

        with Image.open(nomeArquivo) as img:
            # Calcula as coordenadas para a área a ser cortada com base na porcentagem
            minC, minR, maxC, maxR = casos.loc[i, ['ObjectsBoundingBoxMinimumC', 'ObjectsBoundingBoxMinimumR', 'ObjectsBoundingBoxMaximumC', 'ObjectsBoundingBoxMaximumR']]
            rect_width = maxC - minC
            rect_height = maxR - minR
            new_width = rect_width * percentage
            new_height = rect_height * percentage

            left = minC + (rect_width - new_width) / 2
            top = minR + (rect_height - new_height) / 2
            right = left + new_width
            bottom = top + new_height

            # Corta a imagem original
            cropped = img.crop((left, top, right, bottom))

            # Cria uma máscara preta de 128x128
            mask = Image.new('RGB', (128, 128), (0, 0, 0))

            # Calcula o posicionamento do corte na máscara
            x_offset = int((128 - cropped.size[0]) / 2)
            y_offset = int((128 - cropped.size[1]) / 2)

            # Cola a imagem cortada na máscara
            mask.paste(cropped, (x_offset, y_offset))


            # Salvar a imagem processada
            category = casos['ObjectsCategory'][i]
            pathToFile = str(output_path) + "/" + 'Dataset01_' + str(percentage * 100) + "/" + "images" + "/"
            if not os.path.exists(pathToFile):
                os.makedirs(pathToFile)
            filenameSave = pathToFile + f"{casos['Exame'][i]}-{casos['InstanciaExame'][i]}-{i}-{category}.bmp"
            mask.save(filenameSave) # Salva a imagem redimensionada


path_project = Path() / '1_entrada'
csv_file =  "balanceado.csv"
output_path = Path().absolute().parent / "1_entrada"

# Porcentagens para gerar ROIs
percentages = [1, 0.95,  0.9 , 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5, 0.45,  0.4 , 0.35,  0.3 ,0.25, 0.2, 0.15, 0.1, 0.05]

# Gerar ROIs para cada porcentagem
for percentage in percentages:
    print("Imagens feitas para " + str(percentage*100))
    mask_images_with_percentage(csv_file, output_path, percentage)