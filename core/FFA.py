import torch
import numpy as np
from FFALayer import FFALayer, FFALayer1

class FFA(torch.nn.Module):
    def __init__(self, device, sizes=None, threshold=0.5, bias=False, losstype=0):
        if sizes is None:
            sizes = [800,10]
        super().__init__()

        self.s = 28**2
        self.nl = 10

        self.activation = torch.nn.ReLU()
        self.norm = torch.nn.functional.normalize
        self.id = torch.nn.Identity()
        self.prob = torch.nn.Softmax()


        self.device = device

        self.pre = lambda x : self.norm(x.reshape((x.shape[0], np.prod(x.shape[1:]))).to(torch.float32))
        self.sizes = [self.s,*sizes]
        if losstype == 0:
            self.layers = torch.nn.ModuleList([FFALayer(self.sizes[i], self.sizes[i+1], bias=bias, act=self.activation, threshold=threshold, device = self.device) for i in range(len(self.sizes)-1)])
        else:
            self.layers = torch.nn.ModuleList([FFALayer1(self.sizes[i], self.sizes[i+1], bias=bias, act=self.activation, threshold=threshold, device = self.device) for i in range(len(self.sizes)-1)])

        

    ###############                                     
    def join(self, x, y):
        """Replace the first 10 pixels of data [x] with one-hot-encoded label [y]
        """
        x_ = x.clone()
        x_[:, :self.nl] *= 0.0
        x_[range(x.shape[0]), y] = 1.0 # x.max()
        return x_
            
    # training, assumes batches of size > 1 
    def train(self, x, true_label, optimizer, minibatch_ratio=1.0, **kwargs):
        x = self.pre(x)
        nbatch = x.shape[0]
        plabel = torch.argmax(true_label,axis=-1)           # positive data
        nlabel = (torch.randint(low=1,high=self.nl,size=(nbatch,),device=self.device) + plabel) % self.nl # random negative data

        loss = 0.0

        # batching
        pos = self.join(x,plabel) # label addition
        neg = self.join(x,nlabel)

        # minibatching
        # a ratio of 1.0 means no minibatching 
        # minibatching makes later layers become more independent of earlier layers
        minibatch_size = int(nbatch*minibatch_ratio)
        nminibatch = nbatch // minibatch_size
        m_pos = [pos[i*minibatch_size:(i+1)*minibatch_size] for i in range(nminibatch)]
        m_neg = [neg[i*minibatch_size:(i+1)*minibatch_size] for i in range(nminibatch)]
        for layer in self.layers:
            for i in range(nminibatch):
                m_pos[i], m_neg[i], layer_loss = layer.train(m_pos[i], m_neg[i], optimizer)
                loss += layer_loss / nminibatch

        return loss

    ##############
    def forward(self, x):
        x = self.pre(x)
        assert x.device == self.device
        goodness_per_label = []
        for label in range(10):
            h = self.join(x, label)
            goodness = []
            for layer in self.layers:
                h = layer(h)
                goodness += [h.pow(2).mean(1)]
            goodness_per_label += [sum(goodness).unsqueeze(1)]
        goodness_per_label = torch.cat(goodness_per_label, 1)
        return goodness_per_label

        
    def predict(self,x):
        x = self.forward(x)
        return torch.argmax(x,1)
