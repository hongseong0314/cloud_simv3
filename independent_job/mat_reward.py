import time
import torch
import sys
import random
import numpy as np
sys.path.append('.') 

from tqdm import tqdm
from codes.machine import MachineConfig
from independent_job.utils.csv_reader import CSVReader
from independent_job.utils.feature_function import features_extract_func, features_normalize_func
from independent_job.env import CloudsimOnline
from independent_job.model.matrix import MMatrixAlgorithm3
from independent_job.config import matrix_config
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from independent_job.model.matrix_net.model import BGC

def trainer(cfg):
    writer = SummaryWriter()
    cfg.agent = BGC(cfg)
    # algorithm = MMatrixAlgorithm3(cfg)
    cfg.agent.model.train()
    clock_lists, clock_lists_smooth = [], []
    with tqdm(range(500), unit="Run") as runing_bar:
        for i in runing_bar:
            cfg.agent.scheduler.step()

            cpus, mems = cfg.m_resource_config[int(i % 2 == 0)]
            cfg.cpus_max = max(cpus)
            cfg.mems_max = max(mems)
            cfg.machine_configs = [MachineConfig(cpu, mem_disk, mem_disk) for cpu, mem_disk in zip(cpus,\
                            mems)]

            loss, clock = job_len_rollout(cfg)
            runing_bar.set_postfix(loss=loss,
                                #    logpa=logpa,
                                #    rewards=r,
                                #    advantage=a,
                                   clock=clock,)
            clock_lists.append(clock)
            clock_lists_smooth.append(np.mean(clock_lists))
            writer.add_scalar("Loss/train", loss, i)
            # writer.add_scalar("rewards/train", r, i)
            writer.add_scalar("clock/train", clock, i)
    writer.flush()
    # plt.plot(clock_lists_smooth)
    # plt.show()

def job_len_rollout(cfg):
    losses, clocks = [], []

    for one_sc_job_len in range(1, 5):
        tmp_idx = sorted(np.random.choice(np.arange(cfg.jobs_len), one_sc_job_len, replace=False))
        cfg.task_configs = np.array(cfg.task_configs_init)[tmp_idx].tolist()

        loss, clock = one_update(cfg)
        
        losses.append(loss)
        clocks.append(clock)
    return np.mean(losses), \
            np.mean(clocks), \
        

def one_update(cfg):
    # trajectories = []
    clock_list = []

    for _ in range(3):
        algorithm = MMatrixAlgorithm3(cfg)
        sim = CloudsimOnline(cfg)
        sim.setup(algorithm)
        sim.env.process(sim.simulation())
        sim.env.run() 
        algorithm.agent.save_trajectorys()
        # trajectories.append(algorithm.current_trajectory)
        clock_list.append(sim.env.now)

    loss, adv = cfg.agent.update_parameters()
    cfg.agent.model_save()
    return loss, np.mean(clock_list)

if __name__ == '__main__':
    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED) 
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    cfg = matrix_config()
    cfg.features_extract_func = features_extract_func
    cfg.features_normalize_func = features_normalize_func

    cfg.m_resource_config = [
                            [[128, 64, 32, 16, 16], [3, 1, 1, 0.5, 0.5]],
                            [[64, 64, 64, 64, 64], [1, 1, 1, 1, 1]]
                            ]

    cfg.model_params['TMHA'] = 'depth'
    cfg.model_params['MMHA'] = 'depth'
    
    cfg.jobs_len = 10
    csv_reader = CSVReader(cfg.jobs_csv)
    cfg.task_configs_init = csv_reader.generate(0, cfg.jobs_len)
    cfg.device = torch.device('cuda') if torch.cuda.is_available() else "cpu"
    # cfg.reward_sacle = 100
    cfg.model_params['save_path'] = 'd_1000_10_nll_{}_{}_500_{}.pth'.format(
                                                                        cfg.model_params['TMHA'],
                                                                         cfg.model_params['MMHA'],
                                                                         SEED)
    cfg.model_params['device'] = cfg.device
    st = time.time()
    trainer(cfg)
    print(time.time() - st)