from pathlib import Path
import os

class Paths:
  def __init__(self, dataSetName: str, methodName: str) -> None:
    self.path_project = Path().absolute().parent
    self.save_path =  self.path_project / '6_resultados' / dataSetName / methodName 
    self.save_metrics_path = self.save_path / 'metrics'
    self.log_path  = self.save_path / '_log'
    self.save_csvs_path =  self.save_metrics_path / "csvs"
    self.refined_path = self.save_csvs_path / '_refined'
    self.save_nets_path_weigths =  self.save_metrics_path / '_weigths'
    self.save_test = self.save_metrics_path / 'test'
    self.base_path_parts =  self.path_project / "1_entrada" / "partitions"

    
    if not os.path.exists(self.save_path):
      os.makedirs(self.save_metrics_path)
      os.mkdir(self.log_path ) 
      os.mkdir(self.save_test )

    if not os.path.exists(self.save_nets_path_weigths):
      os.makedirs(self.save_nets_path_weigths)  ## cria todos os caminhos necessários 

    if not os.path.exists(self.save_csvs_path):
      os.makedirs(self.save_csvs_path)  
      os.mkdir(self.refined_path)
  
  def get_project_paths(self):
     
      return {
          "project":  self.path_project,
          "metrics":  self.save_metrics_path,
          "csvs":  self.save_csvs_path,
          "weigths":  self.save_nets_path_weigths,
          "partitions":  self.base_path_parts,
          "save_path": self.save_path,
          "log": self.log_path ,
          "test": self.save_test 
      }

  def get_nets_path(self, partition: str):
     
     return str(self.save_nets_path_weigths / f'{partition}.hdf5')
  
  def get_csv_path(self):
     return str(self.save_csvs_path)

  def get_refined_path(self):
     return str(self.refined_path)