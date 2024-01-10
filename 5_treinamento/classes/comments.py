# , 'VGG16', 'VGG19', 'ResNet50', 'ResNet50V2', 'ResNet101', 'ResNet101V2', 'ResNet152', 'ResNet152V2', \
#     , 'InceptionResNetV2', , 'MobileNetV2', 'DenseNet121','DenseNet169', , 'EfficientNetB0', 'EfficientNetB1', 

#    ,  \
# 'EfficientNetV2B0', 'EfficientNetV2B1', 'EfficientNetV2B2', 'EfficientNetV2B3', 'EfficientNetV2S', 'EfficientNetV2M', 'EfficientNetV2L', \
# 'EfficientNetV2B0', 'EfficientNetB2', 'EfficientNetB3', , 'EfficientNetB5', 'EfficientNetB6', 'EfficientNetB7'



# Refinar:  
# 'DenseNet201':    Dense 356 Dropout 0.4  Freeze 0.2
# 'InceptionV3':    Dense 256 Dropout 0.2  Freeze 0.3 
# 'MobileNetV2':    Dense 512 Dropout 0.3  Freeze 0.35
# 'MobileNetV3':    Dense 512 Dropout 0.3  Freeze 0.35
# 'Xception':       Dense 512 Dropout 0.4  Freeze 0.2
# 'ResNet101V2':    Dense 128 Dropout 0.3  Freeze 0.2 
# 'EfficientNetB4': Dense 128 Dropout 0.2  Freeze 0.4


# 'ResNet50V2':     Dense     Dropout   Freeze 
# 'MobileNet':      Dense    Dropout   Freeze 
# 'ResNet152V2':    Dense     Dropout   Freeze 

# 'EfficientNetB4','InceptionV3', 'MobileNetV2', 'MobileNetV3', 'Xception'] 
# 'Resnet101V2', 'ResNet50V2', 'ResNet152V2', 'MobileNet'
# 'VGG16', 'VGG19', 'ResNet50', 'ResNet50V2', 'ResNet101', 'ResNet101V2', 'ResNet152', 'ResNet152V2', 'DenseNet201', 'Xception', 'EfficientNetB4'










# def load_dataset(base_path):
#     # base_path.replace("D:/", "C:/")
#     imagens, labels = list(), list()
#     classes = os.listdir(base_path)
#     for c in classes:
#         for p in glob.glob(os.path.join(base_path, c, '*.bmp')):
#             imagens.append(p)
#             labels.append(c)
    
#     return np.asarray(imagens), labels

# def load_dataset_part(folder_name):
    
#     train_X, train_Y = load_dataset(os.path.join(folder_name, "train"))
#     test_x, test_y = load_dataset(os.path.join(folder_name, "test"))

#     #images = np.array(images)/255.0
#     train_Y, test_y = np.array(train_Y), np.array(test_y) 
    
#     #(train_X, test_x, train_Y, test_y) = train_test_split(images, labels, test_size=0.20, stratify=labels)
    
#     return train_X, test_x, train_Y, test_y

