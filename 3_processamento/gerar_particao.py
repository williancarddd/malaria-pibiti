import pandas as pd
from sklearn.model_selection import train_test_split
import os
from pathlib import Path

path_project = Path().absolute().parent
path_act  = Path().absolute()
# Caminho para os arquivos

csv_file = path_project / '2_pre_processamento' / 'casos.csv'

# Lendo o arquivo CSV
casos = pd.read_csv(csv_file)

# Extraindo informações relevantes
Exam = casos['Exame']
Class = casos['ObjectsCategory']
Image = casos['ImagePathName']

# Únicos exames para particionar
unicos = Exam.unique()

# Preparando para salvar as partições
DatasetName = 'Dataset01/RGB/'
partitions_path = path_project / '1_entrada' / DatasetName / 'partitions100'

# Criando 100 partições de validação cruzada
for k in range(1, 101):
    print(f"Processing partition {k}")

    # Dividindo os exames em treino e teste
    train_exams, test_exams = train_test_split(unicos, test_size=0.2)

    # Criando máscaras de treino e teste
    Train = Exam.isin(train_exams)
    Test = Exam.isin(test_exams)

    # Criando a tabela para esta partição
    tb = pd.DataFrame({'Image': Image, 'Class': Class, 'Train': Train, 'Test': Test})

    # Salvando a tabela em um arquivo CSV
    filename = os.path.join(partitions_path, f'{k:02d}b.csv')
    tb.to_csv(filename, index=False)
