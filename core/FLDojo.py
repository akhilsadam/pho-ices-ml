# Don't Decay the Learning Rate, Increase the Batch Size : https://arxiv.org/abs/1711.00489
import torch
import torch.optim as optim
import numpy as np
import math
from tqdm import tqdm

class dojo:
    def __init__(self,
    epochs = 400,
    max_batch_size = 10000):
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

    def train(self, net, train_X, train_y,test_X=None,test_y=None, test_crit=torch.nn.MSELoss(), **kwargs):
        learning_rate_decay, batch_size = self.init_rate(**kwargs)
        scheduler = lambda x : torch.optim.lr_scheduler.LambdaLR(x, lr_lambda=[learning_rate_decay,])
        net.make_sch(scheduler)

        lossnames = net.lossnames if hasattr(net, 'lossnames') else None
        test = test_X is not None and test_y is not None
        test_freq = kwargs.get('test_freq', 0.1)

        losses = []
        sizes = []
        miloss = []
        mtloss = []
        mix = []
        mtx = []
        rl = []
        n = train_X.shape[0]
        for epoch in tqdm(range(self.epochs)): 
            # shuffle
            idx = torch.randperm(train_X.shape[0])
            train_X = train_X[idx]
            train_y = train_y[idx]


            running_loss = np.zeros((net.n_components,))
            cbatch_size = batch_size(epoch/self.epochs)
            mini = n//cbatch_size
            for i in range(mini):

                # forward + backward + optimize
                inp = train_X[i:i+cbatch_size]

                ls = net.train(inp, train_y[i:i+cbatch_size], **kwargs)

                if any(math.isnan(l) for l in ls):
                    print("NAN")
                    return [rl, sizes, mix, miloss, losses]

                # print statistics
                running_loss += ls/mini
                miloss.append(ls)
                mix.append(epoch + i/mini)

            # collect statistics
            rl.append(net.sch_rate())
            sizes.append(cbatch_size)
            losses.append(running_loss)

            # update learning rate
            net.sch_step()

            # test
            if test:
                it = (epoch*test_freq)
                it -= int(it)
                if abs(it) < 1e-19:
                    mtx.append(epoch)
                    mtloss.append(test_crit(net.forward(test_X), test_y).detach().cpu().numpy())
        if test:
            return [np.squeeze(np.array(rl)), sizes, mix, np.array(miloss), np.array(losses), lossnames, mtx, np.array(mtloss)]
        return [np.squeeze(np.array(rl)), sizes, mix, np.array(miloss), np.array(losses), lossnames]