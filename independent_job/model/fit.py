import torch
import numpy as np

class Node(object):
    def __init__(self, observation, action, reward, clock):
        self.observation = observation
        self.action = action
        self.reward = reward
        self.clock = clock


class RLAlgorithm(object):
    def __init__(self, agent, reward_giver, features_normalize_func, features_extract_func):
        self.agent = agent
        self.reward_giver = reward_giver
        self.features_normalize_func = features_normalize_func
        self.features_extract_func = features_extract_func
        self.current_trajectory = []

    def extract_features(self, valid_pairs):
        features = []
        for machine, task in valid_pairs:
            features.append([machine.cpu, machine.memory] + self.features_extract_func(task))
        features = self.features_normalize_func(features)
        return features

    def __call__(self, cluster, clock, full, state):
        machines = cluster.machines
        tasks = cluster.tasks_which_has_waiting_instance
        all_candidates = []
        # print(tasks)
        for machine in machines:
            for task in tasks:
                if machine.accommodate(task):
                    all_candidates.append((machine, task))

        if len(all_candidates) == 0:
            self.current_trajectory.append(Node(None, None, self.reward_giver, clock))
            return None, None
        else:
            features = self.extract_features(all_candidates)
            features = torch.from_numpy(features).to(torch.float32)
            pair_index = self.agent.Qnet.full_pass(features)
            node = Node(features, pair_index, 0, clock)
            self.current_trajectory.append(node)

        return all_candidates[pair_index]

class RRLAlgorithm(object):
    def __init__(self, agent, reward_giver):
        self.agent = agent
        self.reward_giver = reward_giver
        self.current_trajectory = []

    def __call__(self, cluster, clock, full_tasks_map, state):
        machines = cluster.machines
        machine_feature = state.machine_feature
        task_feature = state.task_feature
        ninf_mask = state.ninf_mask

        if (~torch.isinf(ninf_mask)).sum() == 0:
            self.current_trajectory.append(Node(None, None, self.reward_giver, clock))
            return None, None

        machine_num = machine_feature.size(1)
        task_num = task_feature.size(1)
        MACHINE_IDX = torch.arange(task_num)[None, :].expand(machine_num, task_num)
        TASK_IDX = torch.arange(machine_num)[:, None].expand(machine_num, task_num)

        features = torch.cat([machine_feature[:, TASK_IDX, :], task_feature[:, MACHINE_IDX, :]],
          dim=-1).reshape(task_num*machine_num, -1)[(~torch.isinf(ninf_mask)).reshape(-1,)]

        m_idx, t_idx = torch.nonzero((~torch.isinf(ninf_mask)).squeeze(0), as_tuple=True)

        pair_index = self.agent.Qnet.full_pass(features)
        node = Node(features, pair_index, 0, clock)
        self.current_trajectory.append(node)
        task = ctypes.cast(full_tasks_map[int(t_idx[pair_index])], ctypes.py_object).value 
        return machines[m_idx[pair_index]], task