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
