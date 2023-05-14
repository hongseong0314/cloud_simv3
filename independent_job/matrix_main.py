import time
import torch
import sys
import numpy as np
sys.path.append('.') 

from tqdm import tqdm
from codes.machine import MachineConfig
from independent_job.utils.csv_reader import CSVReader
from independent_job.utils.feature_function import features_extract_func, features_normalize_func
from independent_job.env import Cloudsim
from independent_job.model.matrix import MatrixAlgorithm
from independent_job.config import matrix_config
from torch.utils.tensorboard import SummaryWriter

def trainer(cfg):
    pomo_machine_configs = []
    for _ in range(3):
        np.random.shuffle(cfg.machine_configs)
        pomo_machine_configs.append(cfg.machine_configs)
    cfg.pomo_machine_configs = pomo_machine_configs

    writer = SummaryWriter()
    algorithm = MatrixAlgorithm(cfg)
    algorithm.model.train()
    with tqdm(range(50), unit="Run") as runing_bar:
        for i in runing_bar:
            loss, logpa, r, a, clock, skip = one_update(algorithm, cfg)
            runing_bar.set_postfix(loss=loss,
                                   logpa=logpa,
                                   rewards=r,
                                   advantage=a,
                                   clock=clock,)
            writer.add_scalar("Loss/train", loss, i)
            writer.add_scalar("rewards/train", r, i)
            writer.add_scalar("clock/train", clock, i)
            writer.add_scalar("skip/train", skip, i)
    writer.flush()

def one_update(algorithm, cfg):
    logpa_sum_list = torch.zeros(size=(1, 0)).to(cfg.device)
    rewards = torch.zeros(size=(1,0)).to(cfg.device)
    skip_cnt = []
    clock_list = []

    for machine_configs in cfg.pomo_machine_configs:
        cfg.machine_configs = machine_configs
        sim = Cloudsim(cfg)
        sim.setup()
        sim.env.process(sim.simulation(algorithm))
        sim.env.run()
        clock_list.append(sim.env.now)
        skip_cnt.append(sim.skip_cnt_f)
        logpa_list = algorithm.logpa_list
        logpa_sum_list = torch.cat((logpa_sum_list, logpa_list.sum(dim=2)), dim=1)
        rewards = torch.cat((rewards, torch.tensor([algorithm.reward])[..., None].to(cfg.device)), dim=1)
    
    loss, advantage = algorithm.update_loss(logpa_sum_list, rewards)    
    algorithm.model_save()
    
    algorithm.logpa_list = torch.zeros(size=(1, 1, 0)).to(cfg.device)
    algorithm.reward = 0.0

    return loss, \
            logpa_sum_list.detach().cpu().numpy().mean(), \
            rewards.detach().cpu().numpy().mean(), \
            advantage, \
            np.mean(clock_list), \
            np.mean(skip_cnt)

if __name__ == '__main__':
    cfg = matrix_config()
    cfg.features_extract_func = features_extract_func
    cfg.features_normalize_func = features_normalize_func
    # cfg.machine_configs = [MachineConfig(64, 1, 1) for i in range(cfg.machines_number)]
    cfg.machine_configs = [MachineConfig(cpu, mem_disk, mem_disk) for cpu, mem_disk in zip([128, 64, 32, 16, 16],\
                    [2, 1, 1, 0.5, 0.5])]
    
    cfg.jobs_len = 2
    csv_reader = CSVReader(cfg.jobs_csv)
    cfg.task_configs = csv_reader.generate(0, cfg.jobs_len)
    cfg.device = torch.device('cuda') if torch.cuda.is_available() else "cpu"
    st = time.time()
    trainer(cfg)
    print(time.time() - st)