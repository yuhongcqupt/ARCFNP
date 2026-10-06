import os
import itertools
import torch
import torchnet
from .optim import MetaAdam

class Engine:

    def __init__(self, configs={}):
        self.device = configs['device']
        self.save_checkpoint_path = configs['save_checkpoint_path']
        if not os.path.exists(self.save_checkpoint_path):
            os.makedirs(self.save_checkpoint_path)
        self.dataset_name = configs['dataset_name']
        self.display_freq = configs['display_freq']

    def learn(self, model, train_dataloader, val_dataloader=None, optimizer=None, scheduler=None, start_epoch=0, max_epoch=30, quiet_mode=False):
        if optimizer is None:
            optimizer = torch.optim.Adam(model.net.parameters())
        model.net.to(self.device)
        self.state = {}
        self.quiet_mode = quiet_mode
        if not self.quiet_mode:
            pass
        self.max_epoch = max_epoch
        for epoch in range(start_epoch, max_epoch):
            self.state['epoch'] = epoch
            self.state['lr'] = optimizer.param_groups[0]['lr']
            self._train(model, train_dataloader, optimizer, scheduler)
            if val_dataloader is not None:
                is_best, _ = self._validate(model, val_dataloader)
            else:
                is_best = False
            self._save_checkpoint(model, is_best)
            if scheduler is not None:
                scheduler.step_epoch(epoch + 1)
        if not self.quiet_mode:
            pass

    def _train(self, model, train_dataloader, optimizer, scheduler=None):
        model.net.train()
        self._training_epoch_start()
        self.state['max_iters'] = len(train_dataloader)
        for i, (X, y) in enumerate(train_dataloader):
            self.state['iteration'] = i
            X = X.to(self.device)
            y = y.to(self.device)
            loss_dict = model.training_step(X, y)
            loss = loss_dict['Loss']
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            self._training_save_infors(loss_dict)
            self._training_step_end()
            if scheduler is not None:
                scheduler.step_iter(self.state['epoch'] * self.state['max_iters'] + i + 1)
                self.state['lr'] = optimizer.param_groups[0]['lr']
        self._training_epoch_end()

    def _training_epoch_start(self):
        for key in self.state:
            if isinstance(self.state[key], torchnet.meter.AverageValueMeter):
                self.state[key].reset()

    def _training_epoch_end(self):
        pass

    def _training_step_end(self):
        pass

    def _training_save_infors(self, infors):
        for key in infors:
            if key not in self.state:
                self.state[key] = torchnet.meter.AverageValueMeter()
            self.state[key].add(infors[key].detach().item())

    def _validate(self, model, val_dataloader):
        model.net.eval()
        with torch.no_grad():
            self._validation_epoch_start()
            self.state['val_max_iters'] = len(val_dataloader)
            for i, (X, y) in enumerate(val_dataloader):
                self.state['iteration'] = i
                X = X.to(self.device)
                y = y.to(self.device)
                val_metric = model.validation_step(X, y)
                self._validation_save_infors(val_metric)
            val_metric = self._validation_epoch_end()
            is_best = self._is_best_model(model, val_metric)
        return (is_best, val_metric)

    def _validation_epoch_start(self):
        for key in self.state:
            if isinstance(self.state[key], torchnet.meter.AverageValueMeter):
                self.state[key].reset()

    def _validation_epoch_end(self):
        return {'Loss': self.state['Loss'].value()[0]}

    def _validation_step_end(self):
        pass

    def _validation_save_infors(self, infors):
        for key in infors:
            if key not in self.state:
                self.state[key] = torchnet.meter.AverageValueMeter()
            self.state[key].add(infors[key])

    def _is_best_model(self, model, metric):
        if 'best_val_metric' in self.state and (not model.compare_metric(metric, self.state['best_val_metric'])):
            return False
        self.state['best_val_metric'] = metric
        return True

    def _save_checkpoint(self, model, is_best=False):
        if is_best:
            filename = self.dataset_name + '_best_checkpoint.pth'
            filename = os.path.join(self.save_checkpoint_path, filename)
            model.configs['best_checkpoint_path'] = filename
            model.configs['best_epoch'] = self.state['epoch'] + 1
            torch.save({'epoch': self.state['epoch'] + 1, 'state_dict': model.net.state_dict()}, filename)

class PASEModelEngine(Engine):

    def learn(self, model, train_dataloader, val_dataloader=None, optimizers=None, schedulers=None, start_epoch=0, max_epoch=30, quiet_mode=False):
        model.net.to(self.device)
        self.state = {}
        self.quiet_mode = quiet_mode
        if not self.quiet_mode:
            pass
        self.max_epoch = max_epoch
        for epoch in range(start_epoch, max_epoch):
            self.state['epoch'] = epoch
            self.state['lr'] = optimizers[0].param_groups[0]['lr']
            self._train(model, train_dataloader, val_dataloader, optimizers, schedulers)
            if val_dataloader is not None:
                is_best, _ = self._validate(model, val_dataloader)
            else:
                is_best = False
            self._save_checkpoint(model, is_best)
        if not self.quiet_mode:
            pass

    def _train(self, model, train_dataloader, val_dataloader, optimizers, schedulers=None):
        model.net.train()
        main_net_optimizer = optimizers[0]
        meta_net_optimizer = optimizers[1]
        self._training_epoch_start()
        self.state['max_iters'] = len(train_dataloader)
        val_dataloader_iter = itertools.cycle(val_dataloader)
        for i, (X, y) in enumerate(train_dataloader):
            self.state['iteration'] = i
            X = X.to(self.device)
            y = y.to(self.device)
            main_net_temp = model.net.copy_main_net()
            X_val, y_val = next(val_dataloader_iter)
            X_val = X_val.to(self.device)
            y_val = y_val.to(self.device)
            loss_dict = model.training_step(X, y, main_net_temp)
            loss = loss_dict['Loss']
            grads = torch.autograd.grad(loss, main_net_temp.parameters(), create_graph=True)
            pseudo_optimizer = MetaAdam(main_net_temp, main_net_temp.parameters(), self.state['lr'])
            pseudo_optimizer.load_state_dict(main_net_optimizer.state_dict())
            pseudo_optimizer.meta_step(grads)
            del grads
            loss_dict = model.training_step_meta_net(X_val, y_val, main_net_temp)
            meta_loss = loss_dict['Loss']
            meta_net_optimizer.zero_grad()
            meta_loss.backward()
            meta_net_optimizer.step()
            loss_dict = model.training_step(X, y)
            loss = loss_dict['Loss']
            main_net_optimizer.zero_grad()
            loss.backward()
            main_net_optimizer.step()
            self._training_save_infors(loss_dict)
            self._training_step_end()
        self._training_epoch_end()

    def _training_epoch_end(self):
        if not self.quiet_mode:
            pass

    def _validation_epoch_end(self):
        if not self.quiet_mode:
            pass
        return {'Loss': self.state['Loss'].value()[0], 'AveragePrecision': self.state['AveragePrecision'].value()[0], 'HammingLoss': self.state['HammingLoss'].value()[0]}
