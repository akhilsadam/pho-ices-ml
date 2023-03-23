import torch
import numpy as np
from tqdm import tqdm
from FLLayer import FLLayer

class FL(torch.nn.Module):
    def __init__(self, device, sizes=None, delta=None, k=None, output_size=1, opttype=None, act=torch.nn.ReLU(), crit = torch.nn.MSELoss(), rl=torch.nn.ReLU(), bias=False,):
        if sizes is None:
            raise ValueError("sizes, delta, k must be specified as lists/iterables")
        if opttype is None:
            opttype = lambda x : torch.optim.ASGD(x, lr=0.001)
            
        super().__init__()

        self.z = [*sizes]
        self.z_out = output_size

        self.n_components = 2 # 2 losses.

        self.rl = rl
        self.device = device
        
        self.layers = torch.nn.ModuleList([
            FLLayer(
                self.z[i], self.z[i+1], self.z_out,
                bias=bias,
                act=act[i] if isinstance(act, list) else act,
                criterion=crit,
                opt=opttype,
                delta=delta[i],
                k=k[i],
                rl = self.rl,
                device = self.device,
                )
            for i in range(len(self.z)-1)])

        self.lossnames = ["localizer", "regression"]

    def train(self, x, y, minibatch_ratio=1.0, repeat_epochs = 1,**kwargs):
        # get batch size, assumes x is of shape (nbatch, nfeatures)
        repeat_epochs = int(max(repeat_epochs,1))
        nbatch = x.shape[0] if len(x.shape) > 1 else 1
        
        loss = np.array([0.0]*2)

        # minibatching
        # a ratio of 1.0 means no minibatching 
        # minibatching makes later layers become more independent of earlier layers
        minibatch_size = max(int(nbatch*minibatch_ratio),1)
        nminibatch = nbatch // minibatch_size
        m_pos = [x[i*minibatch_size:(i+1)*minibatch_size] for i in range(nminibatch)]
        m_neg = [None]*nminibatch
        m_y = [y[i*minibatch_size:(i+1)*minibatch_size] for i in range(nminibatch)]
        for layer in self.layers:
            for i in range(nminibatch):
                if repeat_epochs > 1:
                    for _ in (range(repeat_epochs-1)):
                        layer.train(m_pos[i], m_neg[i], m_y[i], **kwargs)
                m_pos[i], m_neg[i], m_y[i], local_loss, reg_loss = layer.train(m_pos[i], m_neg[i], m_y[i], **kwargs)
                loss[0] += local_loss / nminibatch
                loss[1] += reg_loss / nminibatch

        return loss
    
    def make_sch(self,scheduler):
        for layer in self.layers:
            layer.make_sch(scheduler)

    def sch_step(self):
        for layer in self.layers:
            layer.sch_step()
    
    def sch_rate(self):
        return self.layers[0].sch_rate()

    def forward(self, x):
        yhat = torch.zeros((x.shape[0],self.z_out),device=self.device)
        xc = x
        for layer in self.layers:
            xc = layer.localizer(xc)
            yhat += layer.linear(xc)
        return yhat
        
