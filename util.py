import os
import random
import torch
import zipfile
import numpy as np
# from sklearn.base import r2_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, auc, roc_curve, accuracy_score
import matplotlib.pyplot as plt
from torch import nn
random.seed(7)
np.random.seed(7)
torch.manual_seed(7)
detach = lambda o: o.cpu().detach().numpy().tolist() 

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=7, verbose=True, delta=0, model_file='   .pt', trace_func=print, monitor='loss'):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.model_file = model_file
        self.trace_func = trace_func
        self.monitor = monitor

    # def __call__(self, val_loss, model):
    def __call__(self, val_loss, model, monitor):

        if monitor == 'loss':
            self.monitor = monitor
            score = -val_loss
        if monitor == 'auc':
            self.monitor = monitor
            score = val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            if self.monitor == 'loss':
                self.trace_func(
                    f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
            if self.monitor == 'valid':
                self.trace_func(
                    f'Validation auc increased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')

        torch.save(model.state_dict(), self.model_file)
        self.val_loss_min = val_loss

def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels.reshape(-1)).double()
    correct = correct.sum()
    return correct / len(labels)

def find_yt(a):
    batch_size = a.shape[0]
    non_padding_mask = torch.ne(a, -1)
    sum_a = torch.sum(non_padding_mask, dim=-1)
    last_non_padding_index = torch.argmax(sum_a, dim=1, Keepdim = True)
    last_non_padding_value = a[torch.arange(batch_size), last_non_padding_index]




class ScheduledOptim():
    def __init__(self, optimizer, d_model, n_warmup_steps, lr_mul = 1.0):
        self._optimizer = optimizer
        self.lr_mul = lr_mul
        self.d_model = d_model
        self.n_warmup_steps = n_warmup_steps
        self.n_steps = 0

    def step_and_update_lr(self):
        self._update_lr()
        self._optimizer.step()

    def zero_grad(self):
        self._optimizer.zero_grad()

    def _get_lr_scale(self):
        d_model = self.d_model
        n_steps, n_warmup_steps = self.n_steps, self.n_warmup_steps
        return (d_model ** -0.5) * min(n_steps ** (-0.5), n_steps * n_warmup_steps ** (-1.5))

    def _update_lr(self):
        self.n_steps += 1
        lr = self.lr_mul * self._get_lr_scale()
        #if self.n_steps > 2000:
        #    lr = 3e-3
        for param_group in self._optimizer.param_groups:
            param_group['lr'] = lr


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
            loss_sum += 0.5 * torch.exp(-log_vars[i]) * loss + self.params[i]
        return loss_sum

class CosineWarmupScheduler():
    def __init__(self, optimizer, d_model, n_warmup_steps, lr_mul = 1.0):
        self._optimizer = optimizer
        self.lr_mul = lr_mul
        self.d_model = d_model
        self.n_warmup_steps = n_warmup_steps
        self.n_periodic_steps = n_warmup_steps
        self.n_steps = 0

    def step_and_update_lr(self):
        self._update_lr()
        self._optimizer.step()

    def zero_grad(self):
        self._optimizer.zero_grad()

    def _get_lr_scale(self):
        d_model = self.d_model
        n_steps, n_warmup_steps = self.n_steps, self.n_warmup_steps
        if n_steps <= self.n_warmup_steps:
            return (d_model ** -0.5) * min(n_steps ** (-0.5), n_steps * n_warmup_steps ** (-1.5))
        else:
            base = (d_model ** -0.5) * n_warmup_steps ** (-0.5) * (1 + np.cos(np.pi * ((n_steps - self.n_warmup_steps) % self.n_periodic_steps) / self.n_periodic_steps))
            return base
    
    def _update_lr(self):
        self.n_steps += 1
        lr = self.lr_mul * self._get_lr_scale()
        for param_group in self._optimizer.param_groups:
            param_group['lr'] = lr




def KTLoss(pred_answers, real_answers):
        """
        Parameters:
            pred_answers: the correct probability of questions answered at the next timestamp
            real_answers: the real results(0 or 1) of questions answered at the next timestamp
        Shape:
            pred_answers: [batch_size,seq_len,3]
            real_answers: [batch_size,seq_len,3]
        Return:
        """
        criterion = nn.BCELoss() 
        # calculate auc and accuracy metrics

        y_true = real_answers.float()
        y_pred = pred_answers.float()
        loss = criterion(y_pred, y_true) 
        return loss

def KTValue(pred_answers, real_answers):
        """
        Parameters:
            pred_answers: the correct probability of questions answered at the next timestamp
            real_answers: the real results(0 or 1) of questions answered at the next timestamp
        Shape:
            pred_answers: [batch_size,output_step]
            real_answers: [batch_size,seq_len]
        Return:
        """
        # calculate auc and accuracy metrics
        y_true = real_answers.float().cpu().detach().numpy()
        y_pred = pred_answers.float().cpu().detach().numpy()

        try:
            mse_value = mean_squared_error(y_true, y_pred)
            mae_value = mean_absolute_error(y_true, y_pred)
            rmse_value=np.sqrt(mean_squared_error(y_true,y_pred))
            r2_value=r2_score(y_true,y_pred)
            bi_y_pred = [1 if i >= 0.5 else 0 for i in y_pred]
            acc_value = accuracy_score(y_true, bi_y_pred)
            auc_value = auc(fpr, tpr)
            
            # print("auc_value",auc_value)
            # print("acc_value",acc_value)
        except ValueError as e:
            auc_value, acc_value, auc_value1 = -1, -1, -1
            # print(e)

        return auc_value, acc_value, mse_value, mae_value, rmse_value,r2_value
