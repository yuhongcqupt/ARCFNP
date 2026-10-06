import math
import os
import torch
import torchnet
from .data import *
from .engine import PASEModelEngine
from .metrics import *
from .networks import PASENet

class PASEModel:

    def __init__(self, configs={}):
        self.configs = configs
        self.net = PASENet(configs)
        self.net.type(self.configs['dtype'])
        self.engine = PASEModelEngine(configs)

    def train(self, train_dataloader, val_dataloader=None, quiet_mode=False):
        self.net.training_start(train_dataloader, val_dataloader)
        main_optimizer, main_scheduler, meta_optimizer, meta_scheduler = self.net.configure_optimizers()
        self.engine.learn(self, train_dataloader, val_dataloader, [main_optimizer, meta_optimizer], [main_scheduler, meta_scheduler], self.configs['start_epoch'], self.configs['max_epoch'], quiet_mode)

    def training_step(self, X, y, main_net_temp=None):
        output = self.net(X, main_net_temp)
        loss_dict = self.net.loss_function_train(output, y)
        return loss_dict

    def training_step_meta_net(self, X, y, main_net_temp):
        output = self.net.forward_meta(X, main_net_temp)
        loss_dict = self.net.loss_function_meta(output, y)
        return loss_dict

    def validation_step(self, X, y):
        output = self.net(X)
        loss_dict = self.net.loss_function_eval(output, y)
        pred_probs = output[-1].sigmoid_()
        pred_labels = (pred_probs > 0.5).type_as(pred_probs)
        loss_dict['AveragePrecision'] = AveragePrecision(pred_probs, y)
        loss_dict['HammingLoss'] = HammingLoss(pred_labels, y)
        return loss_dict

    def compare_metric(self, m1, m2):
        if m1['AveragePrecision'] > m2['AveragePrecision']:
            return True
        if math.fabs(m1['AveragePrecision'] - m2['AveragePrecision']) < self.configs['eps']:
            if m1['HammingLoss'] < m2['HammingLoss']:
                return True
            if math.fabs(m1['HammingLoss'] - m2['HammingLoss']) < self.configs['eps']:
                return m1['Loss'] < m2['Loss']
        return False

    def test(self, dataloader):
        self.net.eval()
        with torch.no_grad():
            targets = []
            pred_labels = []
            pred_probs = []
            for X, y in dataloader:
                X = X.to(self.configs['device'])
                y = y.to(self.configs['device'])
                pred_label, pred_prob = self.predict(X)
                targets.append(y)
                pred_labels.append(pred_label)
                pred_probs.append(pred_prob)
            targets = torch.cat(targets, dim=0)
            pred_labels = torch.cat(pred_labels, dim=0)
            pred_probs = torch.cat(pred_probs, dim=0)
        return self.evaluate(pred_labels, pred_probs, targets)

    def predict(self, X):
        return self.net.predict(X)

    def evaluate(self, pred_labels, pred_probs, y):
        metrics = {}
        for metric_name in self.configs['label_metrics']:
            metrics[metric_name] = eval(metric_name)(pred_labels, y)
        for metric_name in self.configs['score_metrics']:
            metrics[metric_name] = eval(metric_name)(pred_probs, y)
        return metrics

    def load_checkpoint(self, checkpoint):
        if os.path.isfile(checkpoint):
            checkpoint = torch.load(checkpoint)
            self.configs['start_epoch'] = checkpoint['epoch']
            self.net.load_state_dict(checkpoint['state_dict'])
            self.net.to(self.configs['device'])
            self.configs['dtype'] = list(self.net.parameters())[0].dtype

    def reset_parameters(self):
        self.net.reset_parameters()

def cross_validation(model, dataset, nfold=10, shuffle=True, random_state=None, eval_on_trainset=False, quiet_mode=True, save_model=False):
    test_metrics = {}
    for metric in model.configs['label_metrics']:
        test_metrics[metric] = torchnet.meter.AverageValueMeter()
    for metric in model.configs['score_metrics']:
        test_metrics[metric] = torchnet.meter.AverageValueMeter()
    if eval_on_trainset:
        train_metrics = {}
        for metric in model.configs['label_metrics']:
            train_metrics[metric] = torchnet.meter.AverageValueMeter()
        for metric in model.configs['score_metrics']:
            train_metrics[metric] = torchnet.meter.AverageValueMeter()
    else:
        train_metrics = None
    for count in range(1, nfold + 1):
        dataset.data_cv_splitter(count, nfold, shuffle, random_state)
        train_dataloader = dataset.train_dataloader
        val_dataloader = dataset.val_dataloader
        test_dataloader = dataset.test_dataloader
        model.reset_parameters()
        model.configs['best_epoch'] = 0
        model.train(train_dataloader, val_dataloader, quiet_mode=quiet_mode)
        model.load_checkpoint(model.configs['best_checkpoint_path'])
        model.configs['start_epoch'] = 0
        metrics = model.test(test_dataloader)
        for key in metrics:
            test_metrics[key].add(metrics[key])
        if eval_on_trainset:
            metrics = model.Test(train_dataloader)
            for key in metrics:
                train_metrics[key].add(metrics[key])
        if save_model:
            fileName = 'checkpoint_{:d}_{:d}_{:d}_cv'.format(shuffle, random_state, nfold)
            path = os.path.join(model.configs['save_checkpoint_path'], fileName + '{:d}.pth'.format(count))
            _save_checkpoint({'epoch': model.configs['best_epoch'], 'state_dict': model.net.state_dict()}, path)
    return (test_metrics, train_metrics)

def _save_checkpoint(checkpoint, path):
    torch.save(checkpoint, path)

def _fmt(x):
    return f'{x:.6f}' if x is not None else 'None'
