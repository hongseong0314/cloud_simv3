import numpy as np
import torch
import ctypes

from codes.base import Algorithm
from torch.optim import Adam as Optimizer
from independent_job.model.matrix_net.model import OneStageModel

class MatrixAlgorithm(Algorithm):
    def __init__(self, cfg):
        self.device = cfg.device
        self.model = OneStageModel(**cfg.model_params).to(self.device)
        self.optimizer = Optimizer(self.model.parameters(), **cfg.optimizer_params['optimizer'])
        self.logpa_list = torch.zeros(size=(1, 1, 0)).to(self.device)
        self.reward = 0.0
    
    def __call__(self, cluster, clock, full_tasks_map, state):
        machines = cluster.machines
        # tasks = cluster.tasks_which_has_waiting_instance

        machine_feature = state.machine_feature.to(self.device)
        task_feature = state.task_feature.to(self.device)
        D_TM = state.D_TM.to(self.device)
        ninf_mask = state.ninf_mask
        machine_pointer = state.machine_pointer
        machine_ninf_mask = ninf_mask[:, [machine_pointer], :]
        machine_ninf_mask_plus_1 = torch.cat((torch.tensor([[[0]]]), \
                                              machine_ninf_mask), dim=-1).to(self.device)

        task_selected, logpa = \
                self.model(machine_feature, task_feature, D_TM, machine_ninf_mask_plus_1, machine_pointer)
        # print(logpa)
        # print(f"pointer : {machine_pointer} task : {int(task_selected[0][0])}")
        
        self.logpa_list = torch.cat((self.logpa_list, logpa[:, :, None]), dim=2)
        self.reward = -clock
        # for i, machine in enumerate(machines):
        #     print(f"machine num {i} : {len(machine.running_task_instances)}, time : {clock}")

        if int(task_selected[0][0]) == 0:
            # skip
            return None, 0
        
        else:
            task = ctypes.cast(full_tasks_map[int(task_selected[0][0])-1], ctypes.py_object).value #start_task_instance(machine)
            return machines[machine_pointer], task

    def update_loss(self, logpas, rewards):
        advantage = rewards - rewards.float().mean(dim=1, keepdims=True)
        loss = -advantage * logpas 
        loss_mean = loss.mean()

        self.model.zero_grad()
        loss_mean.backward()
        self.optimizer.step()
        return loss_mean.item()
    
    def model_save(self):
        torch.save(self.model.state_dict(), "p.pth")