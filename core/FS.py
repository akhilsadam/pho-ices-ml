import torch
import numpy as np
from tqdm import tqdm
from FSLayer import FSLayer

class FS(torch.nn.Module):
    def __init__(self, device, sizes=None, delta=None, k=None,  n_pool = 2, opttype=None, act=torch.nn.ReLU(), crit = torch.nn.MSELoss(), rl=torch.nn.ReLU(), bias=True,):
        if sizes is None or delta is None or k is None:
            raise ValueError("sizes, delta, k must be specified as lists/iterables")
        if opttype is None:
            opttype = lambda x : torch.optim.ASGD(x, lr=0.001)
            
        super().__init__()

        self.z = [*sizes]
        
        self.n_components = 1 # 1 loss.

        self.rl = rl
        self.device = device
        
        self.n_pool = n_pool
        self.layers = torch.nn.ModuleList([ \
            FSLayer(self.z[i], self.z[i+1], self.n_pool, first = i == 0,
                    bias=bias,
                    act=act[i] if isinstance(act, list) else act,
                    criterion=crit,
                    opt=opttype,
                    delta=delta[i],
                    k=k[i],
                    rl = self.rl,
                    device = self.device)
                for i in range(len(self.z)-1)])

        self.lossnames = ["symbolic"]

    def train(self, x, y, minibatch_ratio=1.0, repeat_epochs = 1,**kwargs):
        # get batch size, assumes x is of shape (nbatch, nfeatures)
        repeat_epochs = int(max(repeat_epochs,1))
        nbatch = x.shape[0] if len(x.shape) > 1 else 1
        
        loss = np.array([0.0])

        # minibatching
        # a ratio of 1.0 means no minibatching 
        # minibatching makes later layers become more independent of earlier layers
        minibatch_size = max(int(nbatch*minibatch_ratio),1)
        nminibatch = nbatch // minibatch_size
        b_pos = [x[i*minibatch_size:(i+1)*minibatch_size] for i in range(nminibatch)]
        m_pos = [x[i*minibatch_size:(i+1)*minibatch_size] for i in range(nminibatch)]
        m_neg = [None]*nminibatch
        m_y = [y[i*minibatch_size:(i+1)*minibatch_size] for i in range(nminibatch)]
        for layer in self.layers:
            for i in range(nminibatch):
                if repeat_epochs > 1:
                    for _ in tqdm(range(repeat_epochs-1)):
                        layer.train(b_pos[i], m_pos[i], m_neg[i], m_y[i], **kwargs)
                m_pos[i], m_neg[i], m_y[i], local_loss = layer.train(b_pos[i], m_pos[i], m_neg[i], m_y[i], **kwargs)
                loss[0] += local_loss / nminibatch

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
        xc = self.layers[0].forward(x)
        for layer in self.layers[1:]:
            xc = layer.forward(x,xc)
        return xc
        
