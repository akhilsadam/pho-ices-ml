# Don't Decay the Learning Rate, Increase the Batch Size : https://arxiv.org/abs/1711.00489
import torch
import numpy as np
from tqdm import tqdm

def train(net, epochs, max_batch_size, base_lr, train_X, train_y, final_percent_lr = 0.75,splits = [0.2,0.75],start_batch_size=50):

    def batch_factor(f,batch_size=max_batch_size, start_batch_size=start_batch_size):
        if f < splits[0]:
            f/=2
        return int((batch_size-start_batch_size)*f) + start_batch_size
    
    linear = lambda x : 0.7 + 0.3 * x
    decay = lambda x: 0.5*(1-final_percent_lr)*(np.cos(x*np.pi*0.6)+1) + final_percent_lr

    def lr_decay(epoch):
        f = epoch/epochs
        q = batch_factor(f)
        if f < splits[0]:
            return q / 2
        if f > splits[1]:
            return q * decay((f-splits[1])/(1-splits[1]))
        return q 

    
    optimizer, scheduler, criterion = net.init_train(base_lr, lr_decay)
    
    losses = []
    sizes = []
    miloss = []
    mix = []
    rl = []
    n = train_X.shape[0]
    for epoch in tqdm(range(epochs)): 
        # shuffle
        idx = torch.randperm(train_X.shape[0])
        train_X = train_X[idx]
        train_y = train_y[idx]


        running_loss = 0.0
        cbatch_size = batch_factor(epoch/epochs)
        mini = n//cbatch_size
        for i in range(mini):

            # zero the parameter gradients
            optimizer.zero_grad()
            
            # forward + backward + optimize
            ls = net.train(train_X[i:i+cbatch_size], train_y[i:i+cbatch_size], optimizer, criterion)

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