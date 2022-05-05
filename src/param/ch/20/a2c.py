import common_base_funs
from main_configure_approach import A2cConfigure, MainProcessConf
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os


class A2CNet(nn.Module):
    def __init__(self, s_dim, a_dim, aid):
        super(A2CNet, self).__init__()
        self.s_dim = s_dim
        self.a_dim = a_dim
        self.aid = aid
        self.p1 = nn.Linear(s_dim, 32)
        self.p2 = nn.Linear(32, 32)
        self.pfc = nn.Linear(32, a_dim)
        self.v1 = nn.Linear(s_dim, 32)
        self.v2 = nn.Linear(32, 32)
        self.vfc = nn.Linear(32, 1)
        self.distribution = torch.distributions.Categorical

    def forward(self, s):
        out = F.relu(self.p1(s))
        out = F.relu(self.p2(out))
        policies = self.pfc(out)
        out = F.relu(self.v1(s))
        out = F.relu(self.v2(out))
        values = self.vfc(out)
        return policies, values

    def choose_action(self, iter, s):
        self.eval()
        policy, value = self.forward(s)

        policy_content = str(iter)+'-policy:'+str(policy)
        common_base_funs.log(MainProcessConf.log_policy, policy_content+'\n')

        action_prob = F.softmax(policy, dim=-1)

        action_prob_content = str(iter) + '-action_prob:' + str(action_prob)
        common_base_funs.log(MainProcessConf.log_policy, action_prob_content + '\n')

        cat = torch.distributions.Categorical(action_prob)
        action = cat.sample()
        return action

    def loss_func(self, s, a, v_t, n, ent_coef=0.01, vf_coef=0.5):
        self.train()
        policies, values = self.forward(s)
        td = v_t - values
        c_loss = td.pow(2)  # grad_c = td^2

        probs = F.softmax(policies, dim=1)
        m = self.distribution(probs)
        exp_v = m.log_prob(a) * td.detach().squeeze()
        a_loss = -exp_v  # grad_a = -td^log(a)

        entropy = m.entropy()

        total_loss = (a_loss - entropy * ent_coef + c_loss * vf_coef).mean()

        # if n == 8:
        #     losslog0 = open('log' + os.sep + 'loss_log8.txt', 'a')
        #     losslog0.flush()
        #     losslog0.write('state is:' + str(s) + '\n')
        #     losslog0.write('v_t is:' + str(v_t) + '\n')
        #     losslog0.write('policy is:' + str(policies) + '\n')
        #     losslog0.write('value is:' + str(values) + '\n')
        #     losslog0.write('td is:' + str(td) + '\n')
        #     losslog0.write('c_loss is:' + str(c_loss) + '->' + str(c_loss * vf_coef) + '\n')
        #     losslog0.write('prob is:' + str(probs) + '\n')
        #     losslog0.write('m is:' + str(m) + '\n')
        #     losslog0.write('m.log_prob(a) is:' + str(m.log_prob(a)) + '\n')
        #     losslog0.write('exp_v is:' + str(exp_v) + '\n')
        #     losslog0.write("a_loss: " + str(a_loss) + '\n')
        #     losslog0.write("entropy" + str(entropy) + '->' + str(entropy * ent_coef) + '\n')
        #     losslog0.write("total_loss: " + str(total_loss) + '\n')
        #     losslog0.close()

        return total_loss


class A2CClu:
    def __init__(self):
        self.n_single_agent = A2cConfigure.n_single_agent
        self.n_group_agent = A2cConfigure.n_group_agent
        self.n_agent = self.n_single_agent + self.n_group_agent
        self.s_dim = A2cConfigure.s_dim
        self.single_action = A2cConfigure.single_action
        self.group_action = A2cConfigure.group_action
        self.nets = [A2CNet(self.s_dim, len(self.single_action), _)
                     if _ < self.n_single_agent else A2CNet(self.s_dim, len(self.group_action), _ - self.n_single_agent)
                     for _ in range(self.n_agent)]
        self.optims = [torch.optim.Adam(self.nets[_].parameters(), lr=A2cConfigure.lr) for _ in range(len(self.nets))]

    def choose_actions(self, iter, s):
        return [net.choose_action(iter, s) for net in self.nets]

    def load_nets(self):
        nets = []
        for i in range(len(self.nets)):
            if i < self.n_single_agent:
                path = A2cConfigure.model_dir + A2cConfigure.single_net_prefix + str(i+1)\
                       + A2cConfigure.single_net_suffix
            else:
                path = A2cConfigure.model_dir + A2cConfigure.group_net_prefix + str(i+1)\
                       + A2cConfigure.group_net_suffix
            net = torch.load(path)
            nets.append(net)
        self.nets = nets
        return nets

    def save_nets(self, model_path=A2cConfigure.model_dir):
        model_path = common_base_funs.add_sep(model_path)
        common_base_funs.mkdir_if_not_exists(model_path)
        for i in range(len(self.nets)):
            model_obj = self.nets[i]
            if i < self.n_single_agent:
                path = model_path + A2cConfigure.single_net_prefix + str(i + 1) + A2cConfigure.single_net_suffix
            else:
                path = model_path + A2cConfigure.group_net_prefix + str(i + 1) + A2cConfigure.group_net_suffix
            torch.save(model_obj, path)

    def update(self, buffer_a, buffer_s, buffer_r, s, done, GAMMA=0.9):
        '''
        different reward for each agent
        '''
        for n in range(len(self.nets)):
            # calculate the loss function and backpropagate
            if done:
                v_s_ = 0
            else:
                v_s_ = self.nets[n](s)[-1].data.numpy()[0]  # V(s_t+n)

            buffer_v_target = []
            buff_r = buffer_r[:, n]  # reward buffer for agent n
            for r in buff_r[::-1]:  # reverse buffer r
                v_s_ = r + GAMMA * v_s_
                buffer_v_target.append(v_s_)

            buffer_v_target.reverse()

            ss = torch.stack(buffer_s)
            a = torch.Tensor(np.array(buffer_a)[:, n])
            v_t = torch.Tensor(np.array(buffer_v_target)[:, None])  # .unsqueeze(1)

            loss = self.nets[n].loss_func(ss, a, v_t, n)  # gradient = grad[r + gamma * V(s_) - V(s)]
            self.optims[n].zero_grad()
            loss.backward()
            self.optims[n].step()
