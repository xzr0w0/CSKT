import torch.optim as optim
from model import *
import util
import numpy as np
from torch.optim import lr_scheduler
import os
class trainer():
    def __init__(self,device,lr,num_timesteps_input,num_timesteps_output, q_num, c_num, q_matrix, in_dim, seq_length, hidden_size, dropout, load_dir, lr_mul = 1., n_warmup_steps = 2000, quantile = 0.7, is_quantile = False, warmup_epoch = 10):
        self.model = CSKT(device,num_timesteps_input,num_timesteps_output, q_num, c_num, q_matrix, dropout, in_dim=32, out_dim=12,hidden_size=32, layers=3, prob_mul = False).to(device=device)
        self.model.to(device)
        # The learning rate setting below will not affct initial learning rate
        self.optimizer = optim.Adam(self.model.parameters(), lr = lr, betas = (0.9, 0.98), eps = 1e-9)
        self.scheduler = lr_scheduler.StepLR(self.optimizer, step_size=200, gamma=0.8)
        self.loss = util.KTLoss
        self.evaluate = util.KTValue
        self.step_input = num_timesteps_input 
        self.step_output = num_timesteps_output
        self.q_matrix = q_matrix
        self.c_num = c_num
        self.clip = 5
        self.n_warmup_steps = n_warmup_steps
        self.flag = is_quantile
        self.quantile = quantile
        self.cur_epoch = 0
        self.warmup_epoch = warmup_epoch
        self.awl = AutomaticWeightedLoss(2) 
        self.load_dir = load_dir

    def train(self, Q, Y, cur_epoch):
        if self.load_dir:
            load_dir = self.load_dir
            model_file_name = 'ST'
            model_file = os.path.join(load_dir, model_file_name + '.pt')
            optimizer_file = os.path.join(load_dir, model_file_name + '-Optimizer.pt')
            scheduler_file = os.path.join(load_dir, model_file_name + '-Scheduler.pt')
            self.model.load_state_dict(torch.load(model_file))
            self.optimizer.load_state_dict(torch.load(optimizer_file))
            self.scheduler.load_state_dict(torch.load(scheduler_file))
            self.load_dir = False


        expert_num = 3
        self.model.train()
        # self.scheduler.zero_grad()
        output, gate, res, lpkt_out = self.model(Q,Y)
        #output([bz,out_len])  gate([bz,out_len,3]) res([bz,out_len,3])
        q_y = Q[:,-self.step_output:]
        real = Y[:,-self.step_output:]
        
        # gate = torch.sum(gate.permute(3,0,2,1) * q_skill, dim = -1).permute(1,2,0)
        # gate = torch.softmax(gate,dim=-1) #[bz,seq_len,3]

        #print(res.shape)  #torch.Size([160, 102, 1, 3])    
        ind_y = torch.unsqueeze(real,dim=2).repeat(1, 1, expert_num)

        ind_loss = self.loss(res, ind_y)  #[bz,len,3]

        # worst_avoidance = -.5 * l_worst_avoidance * torch.log(gate)
        # best_choice = -.5 * l_best_choice * torch.log(gate)
        loss_real = self.loss(output,real)
        # loss = loss1
        if cur_epoch < self.warmup_epoch:
            loss = ind_loss
        else:
            loss = loss_real
        self.optimizer.zero_grad() 
        loss.backward()  # 2
        self.optimizer.step()  # 3

        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)

        # self.scheduler.step_and_update_lr()
        # mape = util.masked_mape(output, real, 0.0).item()
        # rmse = util.masked_rmse(output, real, 0.0).item()
        auc, acc, mse, mae, rmse,r2 = self.evaluate(output,real)
        auc1, acc1, mse1, mae1, rmse1,r2_1 = self.evaluate(lpkt_out[:,-self.step_output:],real)
        # print('ind_out',ind_out)
        # print('output',output)
        return loss, auc, acc, mse, mae, rmse, r2, auc1, acc1,

    def save_state_dict(self, model_file, optimizer_file,scheduler_file):
        torch.save(self.model.state_dict(), model_file)
        torch.save(self.optimizer.state_dict(), optimizer_file)
        torch.save(self.scheduler.state_dict(), scheduler_file)
        
    def load_state_dict(self, model):
        self.model.load_state_dict(model)

    def eval(self, Q, Y):
        self.model.eval()
        output,lpkt_out = self.model(Q,Y)
        #output = [batch_size,12,c_num,1]
        real = Y[:,-self.step_output:]
        q_y = Q[:,-self.step_output:]
        loss = self.loss(output, real)
        auc, acc, mse, mae, rmse,r2 = self.evaluate(output,real)
        auc1, acc1, mse1, mae1, rmse1,r2_1 = self.evaluate(lpkt_out[:,-self.step_output:],real)
        return loss, auc, acc, mse, mae, rmse,r2,auc1,acc1



import torch
import torch.nn as nn

# -*- coding: utf-8 -*-

import torch
import torch.nn as nn

class AutomaticWeightedLoss(nn.Module):
    """automatically weighted multi-task loss

    Params：
        num: int，the number of loss
        x: multi-task loss
    Examples：
        loss1=1
        loss2=2
        awl = AutomaticWeightedLoss(2)
        loss_sum = awl(loss1, loss2)
    """
    def __init__(self, num=2):
        super(AutomaticWeightedLoss, self).__init__()
        params = torch.ones(num, requires_grad=True)
        self.params = torch.nn.Parameter(params)

    def forward(self, *x):
        loss_sum = 0
        for i, loss in enumerate(x):
            loss_sum += 0.5 / (self.params[i] ** 2) * loss + torch.log(1 + self.params[i] ** 2)
        return loss_sum
