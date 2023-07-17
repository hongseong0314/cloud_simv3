import copy
import numpy as np
import torch
import torch.nn.functional as F

class Qnet(torch.nn.Module):
    def __init__(self, nS):
        super(Qnet, self).__init__()

        self.feature_extract = torch.nn.Sequential(
            torch.nn.Linear(nS, 3),
            torch.nn.Tanh(),
            torch.nn.Linear(3, 9),
            torch.nn.Tanh(),
            torch.nn.Linear(9, 18),
            torch.nn.Tanh(),
            torch.nn.Linear(18, 9),
            torch.nn.Tanh(),
        )
        self.FC = torch.nn.Linear(9, 1)
        self.initialize_weights()
    
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_normal_(m.weight)

    def forward(self, x):
        x = self.feature_extract(x)
        x = self.FC(x)
        return x.squeeze(-1).unsqueeze(0)

    def full_pass(self, x):
        logits = self.forward(x)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        return action.item()

class Agent(object):
    def __init__(self, device, Qnet, gamma, reward_to_go, 
                 nn_baseline, normalize_advantages, 
                 model_save_path=None,
                 ):
        super().__init__()
        self.device = device
        self.gamma = gamma
        self.reward_to_go = reward_to_go
        self.baseline = nn_baseline
        self.normalize_advantages = normalize_advantages
        self.Qnet = Qnet.to(device)
        self.optimizer = torch.optim.Adam(self.Qnet.parameters() ,lr=0.001)
        self.model_save_path = model_save_path

    def save_parm(self):
        torch.save(self.Qnet.state_dict(), self.model_save_path)
        pass

    def _G_t(self, all_rewards):
        # G_t 
        all_returns = []
        for rewards in all_rewards:
            T = len(rewards)
            discounts = np.logspace(0, T, num=T, base=self.gamma, endpoint=False)
            returns = np.array([np.sum(discounts[:T-t] * rewards[t:]) for t in range(T)])
            all_returns.append(returns)
        return all_returns

    def _compute_advantage(self, all_returns):
        adv_n = copy.deepcopy(all_returns)
        max_length = max([len(adv) for adv in adv_n])

        # pad
        for i in range(len(adv_n)):
            adv_n[i] = np.append(adv_n[i], np.zeros(max_length - len(adv_n[i])))

        adv_n = np.array(adv_n)
        adv_n = adv_n - adv_n.mean(axis=0)

        # origin 
        advs = [adv_n[i][:all_returns[i].shape[0]] for i in range(len(adv_n))]
        return advs


    def _loss(self, state, action, G_t):
        logits = self.Qnet(state.to(self.device))

        # categorical
        # dist = torch.distributions.Categorical(logits=logits)
        # logp = dist.log_prob(torch.tensor(action))

        logp = -F.nll_loss(F.softmax(logits), torch.tensor([action]).to(self.device))
        self.logps.append(-logp.detach().cpu().numpy())
        self.avg.append(G_t)
        
        return -logp * torch.tensor(G_t).to(self.device)
    
    def update_parameters(self, all_states, all_actions, all_rewards):
        # G_t
        all_returns = self._G_t(all_rewards)
        # advantage
        adv_n = self._compute_advantage(all_returns)

        if self.normalize_advantages:  
            adv_s = []
            for advantages in adv_n:
                for advantage in advantages:
                    adv_s.append(advantage)
            adv_s = np.array(adv_s)
            mean = adv_s.mean()
            std = adv_s.std()
        adv_n__ = [(advantages - mean) / (std + np.finfo(np.float32).eps) for advantages in adv_n]
        adv_n = adv_n__

        # trajectorys   
        loss_values = []
        advantages__ = []
        for states, actions, adv in zip(all_states, all_actions, adv_n):
            self.logps, self.avg = [], []
            loss_by_trajectory = []
            cnt = 1
            # trajectory
            for s, a, r in zip(states, actions, adv):
                if s is None or a is None:
                    continue
                loss = self._loss(s, a, r)
                loss_by_trajectory.append(loss)
                loss_values.append(loss.detach().cpu().numpy())
                advantages__.append(r)
                
                if cnt % 1000 == 0:
                    self.optimizer.zero_grad()
                    policy_gradient = torch.stack(loss_by_trajectory).mean()
                    policy_gradient.backward()
                    self.optimizer.step()
                    loss_by_trajectory = []
                cnt += 1
            if len(loss_by_trajectory) > 0:
                self.optimizer.zero_grad()
                policy_gradient = torch.stack(loss_by_trajectory).mean()
                policy_gradient.backward()
                self.optimizer.step()
        return np.mean(loss_values), np.mean(advantages__)