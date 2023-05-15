import simpy
import ctypes
import simpy
import torch

from dataclasses import dataclass
from codes.job import Job
from codes.cluster import Cluster

@dataclass
class State:
    machine_pointer : int = 0
    #--------------------------------------
    machine_feature: torch.Tensor = None
    task_feature: torch.Tensor = None
    D_TM: torch.Tensor = None
    ninf_mask: torch.Tensor = None

    
class Cloudsim(object):
    job_cls = Job
    def __init__(self, cfg):
        self.machine_configs = cfg.machine_configs
        self.task_configs = cfg.task_configs
        self.features_extract_func = cfg.features_extract_func
        self.features_normalize_func = cfg.features_normalize_func

        self.env = simpy.Environment()
        self.cluster = Cluster()   
        self.task_num = [0]
        self.task2idx = {}
        self.idx2task = {}
        self.full_tasks_map = {}
        self.machine_num = len(self.machine_configs)
        self.nT = 4
        self.nM = 2
        self.cpus_max = cfg.cpus_max
        self.mems_max = cfg.mems_max

        self.skip_cnt_f = 0
        
    def setup(self):
        self.cluster.add_machines(self.machine_configs)
        self.done = False
        self.env.process(self.arrived_job_check())

        self.trajectory = []
        self.arrived_job = []
        
        # state
        self.skip_cnt = 0
        self.step_state = State()
        self.machine_feature = torch.tensor([[m.cpu, m.memory] \
                                             for m in self.machine_configs], \
                                            dtype=torch.float32)[None, ...].expand(1, self.machine_num, self.nM)
        self.task_feature = torch.zeros(size=(1, 0, self.nT), dtype=torch.float32)
        self.D_TM = torch.zeros(size=(1, self.task_num[0], self.machine_num))
        self.ninf_mask = torch.full(size=(1, self.machine_num, self.task_num[0]),fill_value=float('-inf'))

    def state_update(self):
        TASK_IDX = torch.arange(self.machine_num)[:, None].expand(self.machine_num, self.task_num[0])
        MACHINE_IDX = torch.arange(self.task_num[0])[None, :].expand(self.machine_num, self.task_num[0])
        
        # machine feature update [B, M, Feature:2]
        self.machine_feature = torch.tensor([[m.cpu, m.memory] \
                                                for m in self.cluster.machines], \
                                            dtype=torch.float32)[None, ...].expand(1, self.machine_num, self.nM)
        # D_MT update [M, T]
        self.D_TM = self.task_feature[None, ..., 2].expand(1, self.machine_num, self.task_num[0]).transpose(2,1)

        ### mask_update
        ## available_task [B, T]
        available_task = ~(self.task_feature[..., -1] == 0)
        ## available_machine [B, M, T]
        available_machine = (self.machine_feature[:, TASK_IDX, :] >= \
                             self.task_feature[:, MACHINE_IDX, :2]).all(dim=3)

        task_enable = (available_task[..., MACHINE_IDX, ...] & available_machine)
        self.ninf_mask = torch.full(size=(1, self.machine_num, self.task_num[0]),fill_value=float('-inf'))
        self.ninf_mask[task_enable] = 0 

        # step_update
        # self.step_state.machine_feature = self.machine_feature.clone()
        # self.step_state.task_feature = self.task_feature[:, :self.task_num[0], :].clone()
        # self.step_state.D_TM = self.D_TM[:, :self.task_num[0], :].clone()
        # self.step_state.ninf_mask = self.ninf_mask[:, :, :self.task_num[0]].clone()

        self.step_state.machine_feature = (self.machine_feature.clone() - \
                                           torch.tensor([0,0],dtype=torch.float32)) / \
                                            torch.tensor([self.cpus_max, self.mems_max],dtype=torch.float32)

        self.step_state.task_feature = (self.task_feature.clone() - 
                                        torch.tensor([0.65, 0.009, 74.0, 80.3],dtype=torch.float32)) / \
                                        torch.tensor([0.23, 0.005, 108.0, 643.5],dtype=torch.float32)
        self.step_state.D_TM = (self.D_TM.clone() - 
                                torch.tensor([74.0],dtype=torch.float32)) / \
                                torch.tensor([108.0],dtype=torch.float32)
        self.step_state.ninf_mask = self.ninf_mask.clone()


    def step(self, decision_maker):
        while True:
            if self.env.now > 600: #terminal Done
                break
            self.state_update()
            machine, task = decision_maker(self.cluster, 
                                           self.env.now,
                                           self.full_tasks_map,
                                           self.step_state,
                                           )
            self.step_state.machine_pointer = (self.step_state.machine_pointer + 1) % self.machine_num
            parallel_machine_time_done = self.parallel_machine_time(machine, task)
            if parallel_machine_time_done:
                break
            
            # step decision
            # if task == 0:
            #     self.skip_cnt += 1
            #     continue
            # else:
            task.start_task_instance(machine)
            # print(f"task {task.id} -> machine {machine.id - 5} assign, left : {task.waiting_task_instances_number}")
            # task instance update
            self.task_feature[0, self.task2idx[task.task_index], -1] -= 1


    def simulation(self, decision_maker):
        while not (self.done \
               and len(self.cluster.unfinished_jobs) == 0):
            if self.env.now > 600:
                break
            self.step(decision_maker)
            yield self.env.timeout(1)


    def arrived_job_check(self):
        for job_config in self.task_configs:
            assert job_config.submit_time >= self.env.now
            yield self.env.timeout(job_config.submit_time - self.env.now)
            job = Cloudsim.job_cls(self.env, job_config, self)
            # print('a task arrived at time %f' % self.env.now)
            self.cluster.add_job(job)
            # self.arrived_job.append([self.env.now, self.cluster.tasks_which_has_waiting_instance])
        self.done = True

    def parallel_machine_time(self, machine, task):
        done = False
        if machine is None or task is None:
            done = True
        if self.skip_cnt + 1 == self.machine_num:
            self.skip_cnt = 0
            self.skip_cnt_f += 1
            done = True
        if self.step_state.machine_pointer == 0:
            # print("skip reset")
            self.skip_cnt = 0
        return done