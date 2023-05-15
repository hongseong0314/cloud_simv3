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
from independent_job.model.fit import RLAlgorithm 
from independent_job.config import fit_config
from independent_job.model.fit_model.model import Qnet, Agent
# from torch.utils.tensorboard import SummaryWriter

def trainer(cfg):
    model = Qnet(6) 
    agent = Agent(model, 0.999, reward_to_go=True, nn_baseline=True, normalize_advantages=True,
                model_save_path='rein.pth')
    
    trajectories = []
    # makespans = []
    # average_completions = []
    # average_slowdowns = []
    clock_list = []

    # writer = SummaryWriter()
    for i in tqdm(range(cfg.n_iter)):
        for _ in range(cfg.n_episode):
            algorithm = RLAlgorithm(agent, -1, features_extract_func=features_extract_func,
            features_normalize_func=features_normalize_func)
            sim = Cloudsim(cfg)
            sim.setup()
            sim.env.process(sim.simulation(algorithm))
            sim.env.run() 
            trajectories.append(algorithm.current_trajectory)
            clock_list.append(sim.env.now)
        all_observations = []
        all_actions = []
        all_rewards = []
        for trajectory in trajectories:
            observations = []
            actions = []
            rewards = []
            for node in trajectory:
                observations.append(node.observation)
                actions.append(node.action)
                rewards.append(node.reward)

            all_observations.append(observations)
            all_actions.append(actions)
            all_rewards.append(rewards)
        loss, adv = agent.update_parameters(all_observations, all_actions, all_rewards)
        agent.save_parm()

    #     writer.add_scalar("Loss/train", loss, i)
    #     writer.add_scalar("Advatage/train", adv, i)
    # writer.flush()
    np.save('fit_clock_list', np.array(clock_list))



if __name__ == '__main__':
    cfg = fit_config()
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