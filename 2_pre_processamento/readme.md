# Pré-processamento
Nessa pasta aconcetece o pré-processamento dos dados, geramos um arquivo csv com todos dados identificados entre infectados
e não infectados, fazemos o balanceamento do dataset a partir do arquivo anterior, garantimos que tenha iguais quantidade de
infectados e não infectados, geramos as rois das imagens, elas são salvas em pastas Dataset01_100 até Dataset01_5,onde o número
final identifica a porcentagem de rois que serão utilizadas para treinamento e teste.

- balancear.py: Script que faz o balanceamento do dataset.
- gerar_rois.py: Script que gera as rois das imagens, 100 até 5, de 5 em 5.
- gerar_csv.py: Script que gera o arquivo csv com os dados identificados entre infectados e não infectados.
- v1: Essa pasta não é utilizada, foi um teste para gerar as rois das imagens.


# Preprocessing

In this folder, data preprocessing takes place. We generate a CSV file with all data identified between infected and non-infected, balance the dataset from the previous file, ensuring equal quantities of infected and non-infected data, generate regions of interest (ROIs) from the images, and save them in folders Dataset01_100 to Dataset01_5, where the final number identifies the percentage of ROIs that will be used for training and testing.

- balancear.py: Script for balancing the dataset.
- gerar_rois.py: Script for generating ROIs of the images, from 100 to 5, in increments of 5.
- gerar_csv.py: Script for generating the CSV file with data identified between infected and non-infected.
- v1: This folder is not used; it was a test to generate ROIs of the images.

