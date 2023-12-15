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
        nomeArquivo = output_path / 'DatasetOriginal' / pathName[8:]

        with Image.open(nomeArquivo) as img:
            # Calculando as coordenadas para o recorte com a porcentagem especificada
            minC, minR, maxC, maxR = casos.loc[i, ['ObjectsBoundingBoxMinimumC', 'ObjectsBoundingBoxMinimumR', 'ObjectsBoundingBoxMaximumC', 'ObjectsBoundingBoxMaximumR']]
            rect_width = maxC - minC
            rect_height = maxR - minR

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
            if category == "red blood cell":
                filenameSave = output_path / 'Dataset01_'+percentage*100+'/images/Healthy' / f"{casos['Exame'][i]}-{casos['InstanciaExame'][i]}-{i}.bmp"
            else:  # assumindo Plasmodium ou outro
                filenameSave = output_path / 'Dataset01_'+percentage*100+'/images/Plasmodium' / f"{casos['Exame'][i]}-{casos['InstanciaExame'][i]}-{i}.bmp"

            rgbCropped.save(filenameSave)


path_project = Path().absolute().parent / '1_entrada'
csv_file = path_project / "balanceado.csv"
output_path = Path().absolute()
crop_images_with_percentage(csv_file, output_path, 0.8)
