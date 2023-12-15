import pandas as pd
from PIL import Image, ImageDraw
import numpy as np
import os
from pathlib import Path

# Definindo o caminho base
path_project = Path().absolute().parent / '1_entrada'
pathActt = Path().absolute() 
csv_file = pathActt /  "balanceado.csv"

# Lendo o arquivo CSV
casos = pd.read_csv(csv_file)

# Inicializando a matriz de tamanhos
tamanhos = np.zeros((len(casos), 4))

# Processamento de imagens para cada caso
for i in range(len(casos)):
    # Extraindo informações da linha atual
    pathName = casos['ImagePathName'][i]
    nomeArquivo = path_project / 'DatasetOriginal' / pathName[8:]

    # Lendo a imagem
    with Image.open(nomeArquivo) as img:
        # Calculando as coordenadas para o recorte
        minC, minR, maxC, maxR = casos.loc[i, ['ObjectsBoundingBoxMinimumC', 'ObjectsBoundingBoxMinimumR', 'ObjectsBoundingBoxMaximumC', 'ObjectsBoundingBoxMaximumR']]
        rect = (minC, minR, maxC, maxR)

        # Recortando a imagem
        rgbCropped = img.crop(rect)

        # Armazenando tamanhos
        tamanhos[i, 0:2] = rgbCropped.size

        # Convertendo para escala de cinza
        Icropped = rgbCropped.convert('L')

        # Criando a máscara com borda e círculo central
        """ Labels = Image.new('L', rgbCropped.size, 0)
        draw = ImageDraw.Draw(Labels)
        borderInc = 2
        draw.rectangle([0, 0, rgbCropped.size[0], borderInc], fill=1)
        draw.rectangle([0, 0, borderInc, rgbCropped.size[1]], fill=1)
        draw.rectangle([0, rgbCropped.size[1] - borderInc, rgbCropped.size[0], rgbCropped.size[1]], fill=1)
        draw.rectangle([rgbCropped.size[0] - borderInc, 0, rgbCropped.size[0], rgbCropped.size[1]], fill=1)

        # Definindo o círculo central
        x1, x2 = rgbCropped.size[0] // 2, rgbCropped.size[1] // 2
        percentage = 0.8
        inc = round(percentage * min(x1, x2))
        draw.ellipse((x1 - inc, x2 - inc, x1 + inc, x2 + inc), fill=2) """

        # Mais processamento pode ser adicionado aqui conforme necessário

        # Salvar a imagem processada
        category = casos['ObjectsCategory'][i]
        if category == "red blood cell":
            filenameSave = path_project / 'Dataset01/Gray/Healthy' / f"{casos['Exame'][i]}-{casos['InstanciaExame'][i]}-{i}.bmp"
        else:  # assumindo Plasmodium ou outro
            filenameSave = path_project / 'Dataset01/Gray/Plasmodium' / f"{casos['Exame'][i]}-{casos['InstanciaExame'][i]}-{i}.bmp"

        Icropped.save(filenameSave)
