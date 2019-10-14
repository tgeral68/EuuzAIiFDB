import torch
import math
from torch.optim.optimizer import Optimizer, required
from function_tools import poincare_function as pf


class PoincareOptimizer(Optimizer):
    def __init__(self, params, lr=required):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        defaults = dict(lr=lr)
        super(PoincareOptimizer, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(PoincareOptimizer, self).__setstate__(state)

    def _optimization_method(self, p, d_p, lr):
        return p.new(p.size()).zero_()

    def step(self, closure=None):
        for group in self.param_groups:
            for p in group['params']:  
                if p.grad is None:
                    continue
                d_p = p.grad.data
                self._optimization_method(p.data, d_p.data, lr=group['lr'])

class PoincareBallSGD(PoincareOptimizer):

    def __init__(self, params, lr=required):
        super(PoincareBallSGD, self).__init__(params, lr=lr)

    def _optimization_method(self, p, d_p, lr):
        retractation_factor = ((1 - torch.sum(p** 2, dim=-1))**2)/4
        p.add_(-lr, d_p * retractation_factor.unsqueeze(-1).expand_as(d_p))

class PoincareBallSGDAdd(PoincareOptimizer):
    def __init__(self, params, lr=required):
        super(PoincareBallSGDAdd, self).__init__(params, lr=lr)

    def _optimization_method(self, p, d_p, lr):
        p.copy_(pf.add(pf.renorm_projection(p.data), -lr*pf.exp(d_p.new(d_p.size()).zero_(),d_p)))

class PoincareBallSGDExp(PoincareOptimizer):
    def __init__(self, params, lr=required):
        super(PoincareBallSGDExp, self).__init__(params, lr=lr)

    def _optimization_method(self, p, d_p, lr):
        p.copy_(pf.exp(p, -lr*d_p))

class PoincareBallSGAExp(PoincareOptimizer):
    def __init__(self, params, lr=required):
        super(PoincareBallSGAExp, self).__init__(params, lr=lr)

    def _optimization_method(self, p, d_p, lr):
        p.copy_(pf.exp(p, lr*d_p))