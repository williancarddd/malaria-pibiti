import pandas as pd
import numpy as np
import os
from pathlib import Path
from pattern import pattern_class

# Caminho para o arquivo
path_project = Path().absolute()

#linux
#csv_file = path_project / '2_pre_processamento' / "casos.csv"

csv_file = path_project /  "casos.csv"

# Lendo o arquivo CSV
casos = pd.read_csv(csv_file)

# Selecionando casos que não são células sanguíneas vermelhas
infectados = casos[casos['ObjectsCategory'] == 1 ] # está infectada

# Obtendo exames únicos dos casos infectados
unicos = infectados['Exame'].unique()
infectados_balanceados = pd.DataFrame()

print(len(unicos))
# Balanceando os dados
for exame in unicos:
    # Contando o número de casos infectados para este exame
    num_infectados = len(infectados[infectados['Exame'] == exame])

    #número igual de células sanguíneas vermelhas
    celulas_saudaveis = casos[(casos['ObjectsCategory'] == 0) & (casos['Exame'] == exame)].head(num_infectados)

    # Adicionando ao conjunto balanceado
    infectados_balanceados = pd.concat([infectados_balanceados, infectados[infectados['Exame'] == exame], celulas_saudaveis])

# Salvando o conjunto de dados balanceado em um novo arquivo CSV
infectados_balanceados.to_csv(os.path.join(path_project /   'balanceado.csv'), index=False)
print(infectados_balanceados['ObjectsCategory'].value_counts())
#linux
#infectados_balanceados.to_csv(os.path.join(path_project / '2_pre_processamento' /  'balanceado.csv'), index=False)
