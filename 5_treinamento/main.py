import tensorflow as tf
import os, logging, datetime, time
from config.config import Config
from classes.paths import Paths
from classes.colors import bcolors
from classes.model.model_building import ModelBuilding
from classes.metrics import Metrics
from classes.data.data_processor import DataSetProcessor
from classes.model.test_model import TestModel 
from tensorflow.keras.callbacks import  ReduceLROnPlateau
from tensorflow.keras.models import load_model

if __name__ == '__main__':

  logging.basicConfig(filename='run_registers_.log', encoding='utf-8', level=logging.DEBUG)

  configMain = Config()
  configMethodsRun = configMain.get_methods_run()
  configCnn = configMain.get_cnn_config()
  configMain.start_gpu_config()

  lr_reduce = ReduceLROnPlateau(monitor='val_accuracy', factor=0.1, min_delta=configCnn['alpha'], patience=3, verbose=0)
  early = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=5, mode='max')


  makeNet = ModelBuilding()
  dataSets = configMain.get_data_sets()



  for indiceDataSet, dataSetName in enumerate(dataSets): # para cada porcentagem
    for indiceMethod, methodName  in enumerate(configMethodsRun): # para cada rede neural
      for indicePartition, partitionName in enumerate(range(1, 101)):

        paths = Paths(methodName=methodName, dataSetName=dataSetName)
        if(os.path.exists(paths.get_project_paths()['project'])):
          continue # porcamente uma validação para não refazer tudo novamente , apenas o que precisar
        metrics = Metrics(paths=paths, bgColors=bcolors)
        dataProcessor = DataSetProcessor(paths=paths, config=configMain)
        necessaryPath = paths.get_project_paths()

        testModel = TestModel(paths=paths)
        for indiceDenseNum, denseNum in enumerate(configMain.get_dense_num()):
            for indiceDropOutNum, dropOutNum in enumerate(configMain.get_drop_out()):
              for indiceFreezePercentege, freezePercentege in enumerate(configMain.get_freeze_percentage()):

                dnv = bcolors.OKGREEN + f"DataSetName: {dataSetName}\
                      {methodName}: Partition {partitionName}\
                      DenseNum {denseNum}, dropout {dropOutNum},\
                      freezePercentage {freezePercentege} " + bcolors.ENDC
                print(dnv)
                logging.debug(dnv)

                
                try:
                  initModel, checkpoint = makeNet.create_model(nClasses=configMain.get_n_classes(), methodName=methodName, 
                                                            denseNum=denseNum, dropOut=dropOutNum,
                                                            partition=partitionName, paths=paths,
                                                            metrics=metrics, config=configMain,
                                                            freezePercentage=freezePercentege,
                                                            indiceDataSet=indiceDataSet)
                except Exception as e:
                  logging.error(F"Erro na criação do modelo para {methodName} {dataSetName}")
                
                try:
                  # train_under_X, train_under_Y, test_under_X, test_under_Y = laod_balance_class_parts(os.path.join(base_path_parts, partition))
                  train_under_X, train_under_Y, test_under_X, test_under_Y = dataProcessor.load_dataset(
                  partition=partitionName, data_set_name=dataSetName, n_classes=configMain.get_n_classes())
                except Exception as e:
                  logging.error("Erro na leitura do dataset {dataSetName} {methodName}")
                print(train_under_X)
                dnvv = bcolors.OKCYAN + "Trainning " + methodName + bcolors.ENDC
                print(dnvv)
                logging.debug(dnvv)

                log_dir = necessaryPath['log'] / datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
                tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
                csv_logger = tf.keras.callbacks.CSVLogger(necessaryPath['csvs'] / f"{partitionName}-epoch-results.csv", separator=',', append=True)
                start_train = time.time()
                
                try:
                  with tf.device('/device:GPU:0'):
                    history_net = initModel.fit(train_under_X,
                                            train_under_Y,
                                            steps_per_epoch=(len(train_under_X) // configCnn['batch_size']),
                                            validation_steps = (len(test_under_X) // configCnn['batch_size']),
                                            batch_size = configCnn['batch_size'],
                                            epochs=configCnn['epoch'], 
                                            validation_data=(test_under_X, test_under_Y), 
                                            callbacks=[early, checkpoint, lr_reduce, csv_logger])
                except Exception as e:
                  logging.error(f"Erro no treinamento {methodName} para {dataSetName}")

                runtimeTrain = time.time() - start_train
                dnvvv = bcolors.OKCYAN + dataSetName + " " + methodName + " Trained in %2.2f seconds"%(runtimeTrain) + bcolors.ENDC
                print(dnvvv)
                logging.debug(dnvvv)

                dependencies = {
                            'precision': metrics.precision,
                            'recall': metrics.recall,
                            'f1_score': metrics.f1_score,
                            'specificity': metrics.specificity,
                            'npv': metrics.npv,
                            'mcc': metrics.mcc
                        }
                dnvvvv  = bcolors.OKCYAN + "Testing " +  dataSetName + " "+ methodName + bcolors.ENDC
                print(dnvvvv)
                logging.debug(dnvvvv)

                try:
                  filepath = paths.get_nets_path(partitionName)
                  model = load_model(filepath, custom_objects=dependencies)
                  start_test = time.time()
                  pred = model.predict(test_under_X)
                  runtimeTest = time.time() - start_test
                  dnvvvvv = bcolors.OKCYAN + methodName + " Tested in %2.2f seconds"%(runtimeTest) + bcolors.ENDC
                  print(dnvvvvv)
                  logging.debug(dnvvvvv)
                  print("Completed: %s Percent"%(partitionName) + bcolors.ENDC)
                  testModel.predict_test(pred, test_under_Y, methodName)
                  metrics.calculateMeasures(history_net, 
                                            necessaryPath['metrics'], 
                                            methodName, 
                                            denseNum, 
                                            dropOutNum, 
                                            freezePercentege, 
                                            configCnn['batch_size'], 
                                            runtimeTest=runtimeTest, 
                                            runtimeTrain=runtimeTrain)
                except Exception as e:
                  logging.error(f"Error na parte de testes {methodName} para {dataSetName} {e}")
                  
  logging.info("Treinamento e testes concluído para todas as RNL")
                      
                        