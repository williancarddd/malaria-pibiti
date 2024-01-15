import json
import pandas as pd
from pathlib import Path
from pattern import pattern_class


path_project = Path().absolute().parent
#linux
#path_project = Path().absolute()
# Caminho para os arquivos
test_file = path_project / "1_entrada" / "test.json"
train_file = path_project / "1_entrada" / "training.json"

# Função para processar os dados JSON
def process_json(file_name, is_test=True):
    with open(file_name, 'r') as file:
        data = json.load(file)

    processed_data = []
    for i, item in enumerate(data):
        image_info = item['image']
        for j, obj in enumerate(item['objects']):
            isInfected = 1
            if(pattern_class[obj['category']] != 6 ):
                isInfected = 0 # não tá infectada
            exam_info = {
                'Exame': i + 1,
                'Teste': is_test,
                'Treino': not is_test,
                'InstanciaExame': j + 1,
                'ImageCheckSum': image_info['checksum'],
                'ImagePathName': image_info['pathname'],
                'ImageShapeR': image_info['shape']['r'],
                'ImageShapeC': image_info['shape']['c'],
                'ImageShapeChannels': image_info['shape']['channels'],
                'ObjectsCategory': isInfected,
                'ObjectsBoundingBoxMinimumR': obj['bounding_box']['minimum']['r'],
                'ObjectsBoundingBoxMinimumC': obj['bounding_box']['minimum']['c'],
                'ObjectsBoundingBoxMaximumR': obj['bounding_box']['maximum']['r'],
                'ObjectsBoundingBoxMaximumC': obj['bounding_box']['maximum']['c']
            }
            processed_data.append(exam_info)

    return processed_data

# Processar os arquivos JSON
test_data = process_json(test_file, is_test=True)
train_data = process_json(train_file, is_test=False)

# Combinar os dados de teste e treino
all_data = test_data + train_data

# Criar DataFrame do pandas e ordenar
cases_df = pd.DataFrame(all_data)
cases_df = cases_df.sort_values(by='Exame')

# Salvar como CSV
cases_df.to_csv(path_project / "2_pre_processamento" / 'casos.csv', index=False)
