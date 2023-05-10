
import time
import torch
import sys

sys.path.append('.') 
from codes.machine import MachineConfig
from independent_job.utils.csv_reader import CSVReader
from independent_job.utils.feature_function import features_extract_func, features_normalize_func
from independent_job.env import Cloudsim
from independent_job.model.random import RandomAlgorithm
from independent_job.config import base_config

def trainer(cfg):
    algorithm = RandomAlgorithm()
    sim = Cloudsim(cfg)
    sim.setup()
    sim.env.process(sim.simulation(algorithm))
    sim.env.run() 

if __name__ == '__main__':
    cfg = base_config()
    cfg.features_extract_func = features_extract_func
    cfg.features_normalize_func = features_normalize_func
    cfg.machine_configs = [MachineConfig(64, 1, 1) for i in range(cfg.machines_number)]
    
    csv_reader = CSVReader(cfg.jobs_csv)
    cfg.task_configs = csv_reader.generate(0, cfg.jobs_len)
    cfg.device = torch.device('cuda') if torch.cuda.is_available() else "cpu"
    trainer(cfg)