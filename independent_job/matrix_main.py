import time
import torch
import sys
from tqdm import tqdm
sys.path.append('.') 
from codes.machine import MachineConfig
from independent_job.utils.csv_reader import CSVReader
from independent_job.utils.feature_function import features_extract_func, features_normalize_func
from independent_job.env import Cloudsim
from independent_job.model.matrix import MatrixAlgorithm
from independent_job.config import matrix_config
from independent_job.utils.feature_function import features_extract_func, features_normalize_func

def trainer(cfg):
    algorithm = MatrixAlgorithm(cfg)
    logpa_sum_list = torch.zeros(size=(1, 0)).to(cfg.device)
    rewards = torch.zeros(size=(1,0)).to(cfg.device)

    algorithm.model.train()
    for i in tqdm(range(1000)):
        sim = Cloudsim(cfg)
        sim.setup()
        sim.env.process(sim.simulation(algorithm))
        sim.env.run()

        logpa_list = algorithm.logpa_list
        logpa_sum_list = torch.cat((logpa_sum_list, logpa_list.sum(dim=2)), dim=1)
        rewards = torch.cat((rewards, torch.tensor([algorithm.reward])[..., None].to(cfg.device)), dim=1)

        if (i+1) % 5 == 0:
            print(i)
            loss = algorithm.update_loss(logpa_sum_list, rewards)
            algorithm.model_save()
            # reset
            algorithm.logpa_list = torch.zeros(size=(1, 1, 0))
            algorithm.reward = 0
            logpa_sum_list = torch.zeros(size=(1, 0)).to(cfg.device)
            rewards = torch.zeros(size=(1,0)).to(cfg.device)
            print(f"loss : {loss}")
            break 

if __name__ == '__main__':
    cfg = matrix_config()
    cfg.features_extract_func = features_extract_func
    cfg.features_normalize_func = features_normalize_func
    cfg.machine_configs = [MachineConfig(64, 1, 1) for i in range(cfg.machines_number)]
    cfg.jobs_len = 5
    csv_reader = CSVReader(cfg.jobs_csv)
    cfg.task_configs = csv_reader.generate(0, cfg.jobs_len)
    cfg.device = torch.device('cuda') if torch.cuda.is_available() else "cpu"
    st = time.time()
    trainer(cfg)
    print(time.time() - st)