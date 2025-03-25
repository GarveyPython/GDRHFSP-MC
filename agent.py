import copy
import math
import os
import random

random.seed(1)
from math import exp
# import matplotlib.pyplot as plt
import numpy as np
import torch
import time
from torch.distributions import Categorical
# from .model import Actor, Critic
from scipy.stats import wasserstein_distance

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# device
device = 'cuda' if torch.cuda.is_available() else 'cpu'


class Buffer:
    def __init__(self, batch_size):
        self.state = []
        self.action = []
        self.reward = []
        self.state_ = []
        self.done = []
        self.cnt = 0
        self.batch_size = batch_size

    def store(self, state, action, reward, state_, done):
        self.state.append(state)
        self.action.append(action)
        self.reward.append(reward)
        self.state_.append(state_)
        self.done.append(done)
        self.cnt += 1

    def clean(self):
        self.cnt = 0
        self.state = []
        self.action = []
        self.reward = []
        self.state_ = []
        self.done = []


class Agent_Config:
    def __init__(self, args):
        if args.network == "DNN":
            from .model import Actor, Critic
            print('MODEL---DNN')
        else:
            from utils.attention_model import Actor, Critic
            print('MODEL---Attention')
            print(f'DEVICE-{device}')
        n_hidden = [32, 32, 32]
        self.w = None
        self.a_update_step = args.a_update_step
        self.c_update_step = args.c_update_step
        self.lr = args.lr
        self.gamma = args.gamma
        self.gae_lambda = args.gae_lambda
        self.epsilon = args.epsilon
        self.batch_size = args.batch_size
        self.action_dim = args.ra_action_space
        self.network_mode = args.network
        self.save_path = args.ra_ckpt_path
        self.state_dim = args.ra_state_dim + args.objective
        self.actor = Actor(self.state_dim, self.action_dim, n_hidden).to(device)
        self.old_actor = copy.deepcopy(self.actor).to(device)
        self.critic = Critic(self.state_dim, 1, n_hidden).to(device)
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=self.lr)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=self.lr)
        self.buffer = Buffer(self.batch_size)
        self.critic_loss = []
        self.actor_loss = []
        self.learn_step = 0  # iteration
        self.train_loss_path = './log/train'
        # For Wasserstein-Clip
        self.threshold_base = args.threshold_base  # 初始阈值
        self.attenuation_factor = args.attenuation_factor  # 衰减因子
        self.rollback_factor = args.rollback_factor  # 回滚因子

        self.clip_method = args.clip
        self.last_a_step_loss = None

    def cweight(self, weight):
        self.w = weight

    def choose_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        # print("state_buff", self.buffer.state[-1]) # 第一次选动作时buffer是空的
        prob = self.actor(state).squeeze(0)
        # 根据概率分布采样
        # print(prob)
        dist = Categorical(prob)
        action = dist.sample().item()
        # 按照最大概率选择
        # action = torch.argmax(prob)
        return action

    def update_threshold(self):
        threshold = self.threshold_base * exp(-self.attenuation_factor * (self.learn_step))
        # print(f"ra_threshold{threshold}")
        return threshold

    def wasserstein_clip(self, old_prob, prob, ratio):
        threshold = self.update_threshold()
        # print(threshold)
        # old_prob_lst = old_prob.detach().numpy()
        old_prob_lst = old_prob.cpu().detach().numpy()
        # prob_lst = prob.detach().numpy()
        prob_lst = prob.cpu().detach().numpy()
        clip_res_lst = []
        for i in range(len(prob_lst)):
            wasserstein_d = wasserstein_distance(old_prob_lst[i], prob_lst[i])
            # print("EMD", wasserstein_d, threshold)
            if wasserstein_d >= threshold:
                clip_res_lst.append([-self.rollback_factor * ratio[i][0].detach().item()])
            else:
                clip_res_lst.append([ratio[i][0].detach().item()])
        return torch.FloatTensor(clip_res_lst).to(device)

    def actor_update(self, adv):
        state = torch.FloatTensor(self.buffer.state).to(device)
        action = torch.LongTensor(self.buffer.action).view(-1, 1).to(device)
        prob = self.actor(state)  # 传入策略网络
        # print(prob)
        old_prob = self.old_actor(state)
        prob1 = prob.gather(1, action)
        old_prob1 = old_prob.gather(1, action)

        ratio = torch.exp(torch.log(prob1) - torch.log(old_prob1))
        surr = ratio * adv

        if self.clip_method == "PPO":
            # PPO-Clip torch.clamp将张量的值限制在指定的范围内
            loss = - torch.mean(torch.min(surr, torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * adv))
            # print(loss)
        else:
            # Wasserstein-Clip
            # print(adv)
            loss = - torch.mean(torch.min(surr, self.wasserstein_clip(old_prob, prob, ratio) * adv))
            # print(loss)

        self.actor_optim.zero_grad()
        loss.backward()
        self.actor_optim.step()

        return loss.item()

    def critic_update(self, target):
        state = torch.FloatTensor(self.buffer.state).to(device)
        v = self.critic(state)
        mse_loss = torch.nn.MSELoss()
        # print(len(v))
        loss = mse_loss(v, target)  # 均方差 mse_loss(target, v) 功能上等价
        self.critic_optim.zero_grad()
        loss.backward()
        self.critic_optim.step()
        return loss.item()

    def cal_target(self, state_, done):
        state_ = torch.FloatTensor(state_).unsqueeze(0).to(device)
        reward = np.array(self.buffer.reward)
        # 对reward进行scale操作
        # reward = reward / reward.std()
        # gamma防止过高估计
        td_target = reward + self.gamma * self.critic(state_).detach().cpu().numpy() * (1 - done)  # 当done为真时已完成一个调度Episode
        td_target = torch.FloatTensor(td_target).to(device).view(-1, 1)
        return td_target

    def cal_advantage(self, td_target):
        state = torch.tensor(self.buffer.state, dtype=torch.float).to(device)
        v = self.critic(state).detach().cpu().numpy()
        td_delta = td_target.cpu().numpy() - v  # 时序差分值
        advantage = 0  # 优势函数初始化
        advantage_list = []

        # 计算优势函数
        for delta in td_delta[::-1]:  # 逆序时序差分值 axis=1轴上倒着取 [], [], []
            # 优势函数GAE的公式:计算优势函数估计，使用时间差分误差和上一个时间步的优势函数估计
            advantage = self.gamma * self.gae_lambda * advantage + delta
            advantage_list.append(advantage)
        # 正序
        advantage_list.reverse()
        advantage_list = np.array(advantage_list)
        # numpy --> tensor [b,1]
        adv = torch.FloatTensor(advantage_list).to(device)
        return adv

    def learn(self, state_, done):
        td_target = self.cal_target(state_, done)
        advantage = self.cal_advantage(td_target)
        # adv norm
        # advantage = (advantage - advantage.mean()) / advantage.std()
        # print(advantage)
        self.old_actor.load_state_dict(self.actor.state_dict())
        actor_loss = 0
        critic_loss = 0
        for _ in range(self.a_update_step):
            actor_loss += self.actor_update(advantage)
        for _ in range(self.c_update_step):
            critic_loss += self.critic_update(td_target)
        self.learn_step += 1
        self.buffer.clean()
        actor_loss = actor_loss / self.a_update_step
        critic_loss = critic_loss / self.c_update_step
        self.actor_loss.append(actor_loss)
        self.critic_loss.append(critic_loss)
        return actor_loss, critic_loss

    def save(self, prefix=None, weight=None):
        if weight is None:
            dir_name = time.strftime('%m-%d-%H-%M')
        else:
            str_w = [str(round(100 * w)) for w in weight.reshape(-1, ).tolist()]
            dir_name = 'w' + '_'.join(str_w)

        path = '/'.join([self.save_path, dir_name])
        if not os.path.exists(path):
            os.makedirs(path)
        if prefix is None:
            actor_file = 'actor.pkl'
            critic_file = 'critic.pkl'
        else:
            actor_file = prefix + 'actor.pkl'
            critic_file = prefix + 'critic.pkl'
        actor_file = '/'.join([path, actor_file])
        critic_file = '/'.join([path, critic_file])
        torch.save(self.actor.net.state_dict(), actor_file)
        torch.save(self.critic.net.state_dict(), critic_file)

    def load(self, pkl_list):
        self.actor.net.load_state_dict(torch.load(pkl_list[0], map_location=torch.device(device)))
        self.critic.net.load_state_dict(torch.load(pkl_list[1], map_location=torch.device(device)))


class Route_Agent(Agent_Config):
    def __init__(self, args):
        super().__init__(args)
        # self.save_path = args.ra_ckpt_path
        # self.state_dim = args.ra_state_dim + args.objective
        self.record = None

    def store(self, state, action, reward, done):
        # print(state)
        if self.record is None:  # 第一次进来,先不存
            self.record = [state, action, reward, done]
        else:
            s = self.record[0]
            a = self.record[1]
            r = self.record[2]
            d = self.record[3]
            self.buffer.store(s, a, r, state, d)
            if done is True:  # 回合结束后,next_state置0
                state_ = np.zeros(self.state_dim).tolist()
                self.buffer.store(state, action, reward, state_, done)
            self.record = [state, action, reward, done]


class Sequence_Agent(Agent_Config):
    def __init__(self, args):
        super().__init__(args)
        self.save_path = args.sa_ckpt_path
        self.state_dim = args.sa_state_dim + args.objective
