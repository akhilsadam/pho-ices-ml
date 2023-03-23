import torch
import torch.nn as nn
import torch.optim as optim

def FSPool(*tensors):
    # elementwise product of tensors
    return torch.stack(tensors, dim=0).prod(dim=0)

class FSLayer(nn.Module):
    def __init__(self, in_features, out_features, n_pool, first=False,
                 bias=True, act=nn.ReLU(), criterion=nn.MSELoss(), 
                 opt=lambda x : optim.ASGD(x, lr=0.001), 
                 delta=1.0, k=1.0, eps= 1e-6, rl = nn.ReLU(), device=None, dtype=None):
        super().__init__()
        
        self.act = act
        self.delta = delta
        self.k = k # Lipschitz constant     
        
        self.first = first
        self.functions = [
            torch.sin,
            torch.exp,
            lambda x : torch.log(torch.abs(x)),
            lambda x: 1.0/x,
            lambda x: x,            
        ]
        
        nin = in_features if first else in_features*2
        nin *= len(self.functions)       
        self.n_pool = n_pool 
        self.lin = nn.ModuleList([nn.Linear(nin, out_features, bias=bias, device=device, dtype=dtype) for _ in range(n_pool)])
        self.optFS = opt(self.lin.parameters())
        
        for q in self.lin:
            q.weight.data = torch.zeros_like(q.weight)+0.01*torch.randn_like(q.weight)
            q.weight.data[-1,:] = 1.0/q.weight.size(1)
            if bias:
                q.bias.data.fill_(0.0)
           
        self.criterion = criterion
        self.rl = rl
        self.eps = eps
        
        
    def make_sch(self,scheduler):
        self.sFS = scheduler(self.optFS)
    
    def sch_step(self):
        self.sFS.step()
    
    def sch_rate(self):
        return self.sFS.get_lr()
        
    def activation(self, *x):
        return torch.concat([torch.stack([f(xi) for f in self.functions], dim=0) for xi in x], dim=0)
    
    def forward(self, x, xc=None):
        x_ = self.activation(x) if self.first else self.activation(x,xc)
        y = [torch.sum(q(x_.T),dim=-1) for q in self.lin] # TODO: check if dim is correct
        return FSPool(*y).T
    
    def train(self, x0, x_p, x_n, y, augment_rate:int=2, **kwargs):
        augment_rate = int(max(augment_rate,2))

        # # augment x_p data with passive data
        # xpa = self.passive_augment(x_p, n=augment_rate-1)
        # x_p_aug = torch.concat([x_p,xpa], dim=0)
        # y_rpt = y.repeat(augment_rate,1)

        # xpa2 = self.passive_augment(x0, n=augment_rate-1)
        # x_p_aug2 = torch.concat([x0,xpa2], dim=0)

        # # augment x_n data with active data
        # # print(x_n.size() if x_n is not None else None)
        # xna = self.active_negative_augment(x_p, n=augment_rate-1)
        # x_n_aug = xna if x_n is None else torch.concat([x_n,xna], dim=0)
        
        # xna2 = self.active_negative_augment(x0, n=augment_rate-1)
        # x_n_aug2 = xna2 if x_n is None else torch.concat([x_n,xna2], dim=0)
        
        # yna = y.repeat(augment_rate-1,1)
        # y_rptn = yna if x_n is None else torch.concat([y[:x_n.size(0)],yna],dim=0)
        

        # forward pass
        xin_p = x_p
        xin_n = None
        y_hat_p = self.forward(xin_p) if self.first else self.forward(x0, x_p)
        # print('x_p_aug', x_p_aug.size())
        # print('y_rpt', y_rpt.size())
        # print('x_n_aug', x_n_aug.size())
        # print('y_rptn', y_rptn.size())
        # print('xin_p', xin_p.size())
        # print('xin_n', xin_n.size())
        # print('y_hat_p', y_hat_p.size())
        # print('y_hat_n', y_hat_n.size())


        # lift function to relational space     
        lifted_y_hat_p = torch.concat([y_hat_p,xin_p], dim=1)
        lifted_y_p = torch.concat([y,xin_p], dim=1)

        # train Forward Localizer (FS) layer
        vp = self.criterion(lifted_y_hat_p,lifted_y_p) - self.k*self.delta

        loss1 = self.rl(vp)# + self.rl(vn)
        lossFS = self._update(self.optFS, loss1)

        return xin_p, \
                       xin_n, \
                       y, \
                       lossFS

    def _update(self, arg0, loss, retain_graph=False):
        arg0.zero_grad()
        loss.backward(retain_graph=retain_graph)
        arg0.step()
        return loss.item()
    
    def passive_augment(self, x, n):
        # generate n noisy copies of x to create passive data, 
        # noise is sampled from a uniform distribution of size delta
        
        if n==0:
            return x
        
        x_aug = x.repeat(n,1)
        x_aug += torch.randn_like(x_aug)*self.delta
        return x_aug
    
    def active_negative_augment(self, x, n):
        # generate n noisy copies of x to create active data,
        # given that the noisy copies are just outside the delta-ball of x
        
        if n==0:
            return x
        
        x_aug = x.repeat(n,1)
        noise =  torch.randn_like(x_aug)*self.delta
        direction = noise / (noise.norm(2, 1, keepdim=True) + self.eps)
        x_aug += direction * (self.delta + self.eps)
        return x_aug
        
    
