import math
import torch
from torch.optim.adam import Adam

class MetaAdam(Adam):

    def __init__(self, net, *args, **kwargs):
        super(MetaAdam, self).__init__(*args, **kwargs)
        self.net = net

    def set_parameter(self, current_module, name, parameters):
        if '.' in name:
            name_split = name.split('.')
            module_name = name_split[0]
            rest_name = '.'.join(name_split[1:])
            for children_name, children in current_module.named_children():
                if module_name == children_name:
                    self.set_parameter(children, rest_name, parameters)
                    break
        else:
            current_module._parameters[name] = parameters

    def meta_step(self, grads):
        group = self.param_groups[0]
        for (name, parameter), grad in zip(self.net.named_parameters(), grads):
            parameter.detach_()
            amsgrad = group['amsgrad']
            state = self.state[parameter]
            if len(state) == 0:
                state['step'] = 0
                state['exp_avg'] = torch.zeros_like(parameter, memory_format=torch.preserve_format)
                state['exp_avg_sq'] = torch.zeros_like(parameter, memory_format=torch.preserve_format)
                if amsgrad:
                    state['max_exp_avg_sq'] = torch.zeros_like(parameter, memory_format=torch.preserve_format)
            exp_avg, exp_avg_sq = (state['exp_avg'], state['exp_avg_sq'])
            if amsgrad:
                max_exp_avg_sq = state['max_exp_avg_sq']
            beta1, beta2 = group['betas']
            state['step'] += 1
            bias_correction1 = 1 - beta1 ** state['step']
            bias_correction2 = 1 - beta2 ** state['step']
            if group['weight_decay'] != 0:
                grad = grad.add(parameter, alpha=group['weight_decay'])
            exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
            exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
            if amsgrad:
                torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                denom = (max_exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
            else:
                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
            step_size = group['lr'] / bias_correction1
            self.set_parameter(self.net, name, parameter.addcdiv(exp_avg, denom, value=-step_size))
