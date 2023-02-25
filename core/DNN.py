import torch
import numpy as np

class DNN(torch.nn.Module):
    def __init__(self, sizes=None, d2_input=True, classification=True, activation=torch.nn.ReLU(), criterion=torch.nn.MSELoss(), bias=False, init=None):
        if sizes is None:
            sizes = [784,800,10]
        super().__init__()

        self.sizes = sizes

        self.activation = activation
        self.prob = torch.nn.Softmax() if classification else torch.nn.Identity()

        self.pre = lambda x : x.reshape((x.shape[0], np.prod(x.shape[1:]))).to(torch.float32) if d2_input else x.to(torch.float32)
        self.layers = torch.nn.ModuleList([torch.nn.Linear(self.sizes[i], self.sizes[i+1], bias=bias) for i in range(len(self.sizes)-1)])
        self.post = [*([self.activation,]*(len(sizes)-1)),self.prob]

        self.criterion = criterion     

        if init is not None:
            for layer in self.layers:
                torch.nn.init.xavier_uniform_(layer.weight, gain=init)
        
    ###############    
    def train(self, x, y, optimizer, clip=None, return_outputs=False, **kwargs):
        outputs = self.forward(x)
        loss = self.criterion(outputs, y)
        loss.backward()
        if clip is not None:
            torch.nn.utils.clip_grad_norm_(self.parameters(), clip)
        optimizer.step()
        if return_outputs :
            with torch.no_grad():
                kout = self.forward(x)
            return (loss.item(), kout) 
        return loss.item()
    ###############   
    
    def forward(self, x):
        x = self.pre(x)
        for i in range(len(self.layers)):
            x = self.layers[i](x)
            x = self.post[i](x)
        return x

    def predict(self,x):
        x = self.forward(x)
        return torch.argmax(x,axis=-1)