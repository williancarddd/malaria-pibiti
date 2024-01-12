import pandas as pd
from PIL import Image
import numpy as np
import os
from pathlib import Path

def crop_images_with_percentage(csv_file_path, output_path, percentage):
    casos = pd.read_csv(csv_file_path)

    # Processamento de imagens para cada caso
    for i in range(len(casos)):
        pathName = casos['ImagePathName'][i]
        nomeArquivo = output_path / 'DataSetOriginal' / pathName[8:]

        with Image.open(nomeArquivo) as img:
            # Calculando as coordenadas para o recorte com a porcentagem especificada
            minC, minR, maxC, maxR = casos.loc[i, ['ObjectsBoundingBoxMinimumC', 'ObjectsBoundingBoxMinimumR', 'ObjectsBoundingBoxMaximumC', 'ObjectsBoundingBoxMaximumR']]
            rect_width = maxC - minC
            rect_height = maxR - minR
            # dminiui em retangulos
            new_width = rect_width * percentage
            new_height = rect_height * percentage

            left = minC + (rect_width - new_width) / 2
            top = minR + (rect_height - new_height) / 2
            right = maxC - (rect_width - new_width) / 2
            bottom = maxR - (rect_height - new_height) / 2

            rect = (left, top, right, bottom)
            rgbCropped = img.crop(rect)

            # Salvar a imagem processada
            category = casos['ObjectsCategory'][i]
            pathToFile = str(output_path) +"/"+ 'Dataset01_'+str(percentage*100) +"/"+ "images" +"/"
            if (not os.path.exists(pathToFile)):
                os.makedirs(pathToFile)
            filenameSave = pathToFile + f"{casos['Exame'][i]}-{casos['InstanciaExame'][i]}-{i}-{category}.bmp"  
            rgbCropped.save(filenameSave)


path_project = Path() / '1_entrada'
csv_file =  "balanceado.csv"
output_path = Path().absolute().parent / "1_entrada"

# Porcentagens para gerar ROIs
percentages = [1, 0.95,  0.9 , 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5, 0.45,  0.4 , 0.35,  0.3 ,0.25, 0.2, 0.15, 0.1, 0.05]

# Gerar ROIs para cada porcentagem
for percentage in percentages:
    print("Imagens feitas para " + str(percentage*100))
    crop_images_with_percentage(csv_file, output_path, percentage)