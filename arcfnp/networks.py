import random
from typing import Tuple, TypeVar
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
Tensor = TypeVar('torch.tensor')

def init_random_seed(seed=0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

class TSKMultiLabel(nn.Module):

    def __init__(self, in_features: int, num_classes: int, n_rules: int):
        super().__init__()
        self.in_features = in_features
        self.num_classes = num_classes
        self.n_rules = n_rules
        self.centers = nn.Parameter(torch.zeros(num_classes, n_rules, in_features))
        self.log_sigma = nn.Parameter(torch.zeros(num_classes, n_rules, in_features))
        self.consequents = nn.Parameter(torch.zeros(num_classes, n_rules, in_features + 1))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.centers, mean=0.0, std=0.1)
        nn.init.constant_(self.log_sigma, 0.0)
        nn.init.normal_(self.consequents[..., :self.in_features], mean=0.0, std=0.05)
        nn.init.constant_(self.consequents[..., -1], 0.0)

    def forward(self, x: torch.Tensor, scale: torch.Tensor=None, bias: torch.Tensor=None) -> torch.Tensor:
        B, N = x.shape
        Q, R = (self.num_classes, self.n_rules)
        QR = Q * R
        C = self.centers.reshape(QR, N)
        L = self.log_sigma.reshape(QR, N)
        invs2 = torch.exp(-2.0 * L)
        x2 = x * x
        w_x2 = invs2.t()
        w_dot = (2.0 * C * invs2).t()
        c2_over_s2 = (C * C * invs2).sum(dim=1)
        A = x2 @ w_x2
        B_term = x @ w_dot
        dist = A - B_term + c2_over_s2.unsqueeze(0)
        log_gauss = (-0.5 * dist).reshape(B, Q, R)
        m = log_gauss.max(dim=2, keepdim=True).values
        z = torch.exp(log_gauss - m)
        w = z / (z.sum(dim=2, keepdim=True) + 1e-12)
        a = self.consequents[..., :N].contiguous().reshape(QR, N)
        b0 = self.consequents[..., -1].contiguous().reshape(QR)
        y_r = x @ a.t() + b0.unsqueeze(0)
        y_r = y_r.reshape(B, Q, R)
        if self.training and scale is not None:
            if bias is None:
                bias = 0.0
            y_r = bias + scale * y_r
        logits = (w * y_r).sum(dim=2)
        return logits

class MainNet(nn.Module):

    def __init__(self, configs):
        super(MainNet, self).__init__()
        self.configs = configs
        in_features = configs['in_features']
        num_classes = configs['num_classes']
        n_rules = configs['n_rules']
        self.classifier = TSKMultiLabel(in_features, num_classes, n_rules)
        self.to(configs['device'])

    def reset_parameters(self):
        self.classifier.reset_parameters()

    def forward(self, input: Tensor, scale: Tensor=None, bias: Tensor=None) -> Tuple[Tensor, ...]:
        preds = self.classifier(input, scale, bias)
        return preds

    def forward_enc(self, input: Tensor) -> Tensor:
        return input

class MetaNet(nn.Module):

    def __init__(self, configs):
        super(MetaNet, self).__init__()
        self.configs = configs
        self.scale = nn.Parameter(torch.zeros(configs['train_batch_size'], configs['num_classes'], 1))
        self.bias = nn.Parameter(torch.zeros(configs['train_batch_size'], configs['num_classes'], 1))
        self.to(configs['device'])

    def reset_parameters(self):
        nn.init.zeros_(self.scale)
        nn.init.zeros_(self.bias)

    def reset_coef(self):
        nn.init.zeros_(self.scale)
        nn.init.zeros_(self.bias)

    def get_config_optim(self):
        return [{'params': self.scale, 'weight_decay': 0.0}, {'params': self.bias, 'weight_decay': 0.0}]

    def forward(self, input: Tensor, reuse_eps=False) -> Tuple[Tensor, ...]:
        if self.training:
            scale = F.softplus(self.scale[:input.size(0)])
            bias = self.bias[:input.size(0)]
        else:
            scale = None
            bias = None
        return (scale, bias)

class PASENet(nn.Module):

    def __init__(self, configs):
        super(PASENet, self).__init__()
        self.configs = configs
        init_random_seed(self.configs['rand_seed'])
        self.main_net = MainNet(self.configs)
        self.meta_net = MetaNet(self.configs)
        self.to(configs['device'])
        self.reset_parameters()

    def reset_parameters(self):
        init_random_seed(self.configs['rand_seed'])
        self.meta_net.reset_parameters()
        self.main_net.reset_parameters()

    def copy_main_net(self):
        main_net_temp = MainNet(self.configs)
        main_net_temp.load_state_dict(self.main_net.state_dict())
        main_net_temp.train()
        return main_net_temp

    def forward(self, input: Tensor, main_net_temp=None) -> Tuple[Tensor, ...]:
        if main_net_temp is not None:
            scale, bias = self.meta_net(input)
            preds = main_net_temp(input, scale, bias)
        elif self.training:
            scale, bias = self.meta_net(input, reuse_eps=True)
            preds = self.main_net(input, scale, bias)
        else:
            preds = self.main_net(input)
        return (preds,)

    def forward_meta(self, input: Tensor, main_net_temp) -> Tuple[Tensor, ...]:
        preds = main_net_temp(input)
        return (preds,)

    def training_start(self, train_dataloader, val_dataloader=None):
        self.iters_per_epoch = len(train_dataloader)

    def loss_function_train(self, preds: Tuple[Tensor, ...], targets: Tensor) -> dict:
        Loss, Cls_loss = self._compute_loss(*preds, targets)
        return {'Loss': Loss, 'Cls_loss': Cls_loss}

    def loss_function_meta(self, preds: Tuple[Tensor, ...], targets: Tensor) -> dict:
        Loss = self._compute_loss_meta(*preds, targets)
        return {'Loss': Loss}

    def loss_function_eval(self, preds: Tuple[Tensor, ...], targets: Tensor) -> dict:
        Loss, Cls_loss = self._compute_loss(*preds, targets)
        return {'Loss': Loss.detach().item(), 'Cls_loss': Cls_loss.detach().item()}

    def predict(self, input: Tensor) -> Tuple[Tensor, Tensor]:
        self.eval()
        with torch.no_grad():
            pred_probs = self.main_net(input).sigmoid_()
            pred_labels = (pred_probs > self.configs['threshold']).type_as(pred_probs)
        return (pred_labels, pred_probs)

    def configure_optimizers(self) -> Tuple[Any, Any]:
        main_optimizer = torch.optim.Adam(self.main_net.parameters(), lr=self.configs['lr'], weight_decay=self.configs['weight_decay'])
        meta_optimizer = torch.optim.Adam(self.meta_net.get_config_optim(), lr=self.configs['lr_meta'], weight_decay=self.configs['weight_decay'])
        return (main_optimizer, None, meta_optimizer, None)

    def _compute_loss(self, preds: Tensor, targets: Tensor) -> Tuple[Tensor, ...]:
        Cls_loss = F.multilabel_soft_margin_loss(preds, targets) * targets.size(1)
        Loss = Cls_loss
        return (Loss, Cls_loss)

    def _compute_loss_meta(self, preds: Tensor, targets: Tensor) -> Tuple[Tensor, ...]:
        Loss = F.multilabel_soft_margin_loss(preds, targets) * targets.size(1)
        return Loss
