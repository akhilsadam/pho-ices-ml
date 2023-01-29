# Don't Decay the Learning Rate, Increase the Batch Size : https://arxiv.org/abs/1711.00489
import torch
import torch.optim as optim
import numpy as np
from tqdm import tqdm

class dojo:
    def __init__(self,
    optimizer_lambda = lambda x : optim.ASGD(x, lr=0.001), 
    epochs = 400,
    max_batch_size = 10000):
        self.optimizer_lambda = optimizer_lambda
        self.epochs = epochs
        self.max_batch_size = max_batch_size

    def init_rate(self, final_percent_lr = 0.75,splits = [0.2,0.75],start_batch_size=50,**kwargs):
        def batch_size(f,batch_size=self.max_batch_size, start_batch_size=start_batch_size):
            if f < splits[0]:
                f/=2
            return int((batch_size-start_batch_size)*f) + start_batch_size

        linear = lambda x : 0.7 + 0.3 * x
        decay = lambda x: 0.5*(1-final_percent_lr)*(np.cos(x*np.pi*0.6)+1) + final_percent_lr

        def learning_rate_decay(epoch):
            f = epoch/self.epochs
            q = batch_size(f)
            if f < splits[0]:
                out = q / 2
            elif f > splits[1]:
                out = q * decay((f-splits[1])/(1-splits[1]))
            else:
                out = q
            return out
            
            
        return learning_rate_decay, batch_size

    def train(self, net, train_X, train_y, **kwargs):
        learning_rate_decay, batch_size = self.init_rate(**kwargs)
        optimizer = self.optimizer_lambda(net.parameters())
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=[learning_rate_decay,])
        
        losses = []
        sizes = []
        miloss = []
        mix = []
        rl = []
        n = train_X.shape[0]
        for epoch in tqdm(range(self.epochs)): 
            # shuffle
            idx = torch.randperm(train_X.shape[0])
            train_X = train_X[idx]
            train_y = train_y[idx]


            running_loss = 0.0
            cbatch_size = batch_size(epoch/self.epochs)
            mini = n//cbatch_size
            for i in range(mini):

                # zero the parameter gradients
                optimizer.zero_grad()
                
                # forward + backward + optimize
                ls = net.train(train_X[i:i+cbatch_size], train_y[i:i+cbatch_size], optimizer, **kwargs)

                # print statistics
                running_loss += ls/mini
                miloss.append(ls)
                mix.append(epoch + i/mini)
                
            # collect statistics
            rl.append(scheduler.get_lr())
            sizes.append(cbatch_size)
            losses.append(running_loss)
            
            # update learning rate
            scheduler.step()
            
        return [rl, sizes, mix, miloss, losses]