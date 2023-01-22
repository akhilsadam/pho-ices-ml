
class FFAOptimizer():
    def __init__(self,base_lr):
        self.base = base_lr
        self.lr = base_lr
    def update_lr(self,factor):
        self.lr = self.base * factor
    def zero_grad(self):
        pass
    
class FFAScheduler():
    def __init__(self, optimizer, function):
        self.opt = optimizer
        self.f = function
        self.epoch = 0
        
        self.update_opt()
        
    def update_opt(self):
        self.opt.update_lr(self.f(self.epoch))
        
    def step(self):
        self.epoch += 1
        self.update_opt()
        
    def get_lr(self):
        return self.opt.lr

import torch
from torch import Tensor
from torch.nn import functional as F

class FFALinear():

    def __init__(self, in_features: int, out_features: int, bias: bool = False,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.empty((out_features, in_features), **factory_kwargs) # removed parameter on weight

    def forward(self, input: Tensor) -> Tensor:
        return F.linear(input, self.weight)

