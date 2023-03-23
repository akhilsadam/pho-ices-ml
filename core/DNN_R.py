import torch
import numpy as np
from tqdm import tqdm

class DNN(torch.nn.Module):
    def __init__(self, device, sizes=None, opttype=None, act=torch.nn.ReLU(), crit = torch.nn.MSELoss(), bias=False, **kwargs,):
        if sizes is None:
            raise ValueError("sizes must be specified as lists/iterables")
        if opttype is None:
            opttype = lambda x : torch.optim.ASGD(x, lr=0.001)
            
        super().__init__()

        self.z = [*sizes]

        self.n_components = 1 # 1 loss.

        self.device = device
        self.act = act if isinstance(act, list) else [act,]*(len(self.z)-1)
        self.crit = crit
        
        self.layers = torch.nn.ModuleList([
            torch.nn.Linear(
                self.z[i], self.z[i+1],
                bias=bias,
                device = self.device,
                )
            for i in range(len(self.z)-1)])
        
        self.optimizer = opttype(self.parameters())

        self.lossnames = ["regression"]

    def train(self, x, y, minibatch_ratio=1.0, repeat_epochs = 1,**kwargs):
        # get batch size, assumes x is of shape (nbatch, nfeatures)
        repeat_epochs = int(max(repeat_epochs,1))
        nbatch = x.shape[0] if len(x.shape) > 1 else 1
        
        loss = []

        # minibatching
        # a ratio of 1.0 means no minibatching 
        # minibatching makes later layers become more independent of earlier layers
        minibatch_size = max(int(nbatch*minibatch_ratio),1)
        nminibatch = nbatch // minibatch_size
        for i in range(nminibatch):
            qloss = 0.0
            for _ in (range(repeat_epochs)):
                closs = self.crit(self.forward(x[i*minibatch_size:(i+1)*minibatch_size]), \
                    y[i*minibatch_size:(i+1)*minibatch_size])
                self.optimizer.zero_grad()
                qloss += closs.item()
                closs.backward()
                self.optimizer.step()
                
            loss.append(qloss / (nminibatch*repeat_epochs))
        return np.array(loss)
    
    def make_sch(self,scheduler):
        self.scheduler = scheduler(self.optimizer)

    def sch_step(self):
        self.scheduler.step()
    
    def sch_rate(self):
        return self.scheduler.get_lr()

    def forward(self, x):
        for i,layer in enumerate(self.layers):
            y = layer(x)
            if i < len(self.layers)-1:
                x = self.act[i](y)
        return y
        
