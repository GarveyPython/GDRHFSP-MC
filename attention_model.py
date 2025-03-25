import torch
import torch.nn as nn
import torch.nn.functional as F


class Actor(nn.Module):
    def __init__(self, n_state, n_action, n_hidden):
        super(Actor, self).__init__()
        self.net = nn.ModuleList()
        self.net.append(nn.Sequential(nn.Linear(n_state, n_hidden[0]),
                                      nn.ReLU()))
        self.net.append(nn.Linear(n_hidden[0], n_hidden[0]))  # Q
        self.net.append(nn.Linear(n_hidden[0], n_hidden[0]))  # K
        self.net.append(nn.Linear(n_hidden[0], n_hidden[0]))  # V
        self.net.append(nn.MultiheadAttention(embed_dim=n_hidden[0], num_heads=1))
        self.net.append(nn.LSTM(input_size=n_hidden[0], hidden_size=n_hidden[1], num_layers=1))
        self.net.append(nn.Sequential(
            nn.Linear(n_hidden[1], n_action),
        ))

    def forward(self, x):
        residual = None
        q, k, v = None, None, None
        for i, m in enumerate(self.net):
            if i == 0:
                x = m(x)
                residual = x
            elif i == 1:
                q = m(x)
            elif i == 2:
                k = m(x)
            elif i == 3:
                v = m(x)
            elif i == 4:
                x, _ = m(q, k, v)
                x = x + residual # 残差连接增加网络稳定性
            elif i == 5:
                x, _ = m(x)
            else:
                x = m(x)

        x_ = (torch.ones_like(x) * 1e-3).to(x.device)
        if torch.isnan(x).any():
            x = torch.where(torch.isnan(x), x_, x)
        out = F.softmax(x, dim=1)
        return out


class Critic(nn.Module):
    def __init__(self, n_state, out, n_hidden):
        super(Critic, self).__init__()
        self.net = nn.ModuleList()
        self.net.append(nn.Sequential(nn.Linear(n_state, n_hidden[0]),
                                      nn.ReLU()))
        self.net.append(nn.Linear(n_hidden[0], n_hidden[0]))  # Q
        self.net.append(nn.Linear(n_hidden[0], n_hidden[0]))  # K
        self.net.append(nn.Linear(n_hidden[0], n_hidden[0]))  # V
        self.net.append(nn.MultiheadAttention(embed_dim=n_hidden[0], num_heads=1))
        self.net.append(nn.LSTM(input_size=n_hidden[0], hidden_size=n_hidden[1], num_layers=1))
        self.net.append(nn.Sequential(
            nn.Linear(n_hidden[1], out),
        ))

    def forward(self, x):
        residual = None
        q, k, v = None, None, None
        for i, m in enumerate(self.net):
            if i == 0:
                x = m(x)
                residual = x
            elif i == 1:
                q = m(x)
            elif i == 2:
                k = m(x)
            elif i == 3:
                v = m(x)
            elif i == 4:
                x, _ = m(q, k, v)
                x = x + residual
            elif i == 5:
                x, _ = m(x)
            else:
                x = m(x)
        return x