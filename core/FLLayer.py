import torch
import torch.nn as nn
import torch.optim as optim

class FLLayer(nn.Linear):
    def __init__(self, in_features, out_features,fin_features, 
                 bias=True, act=nn.ReLU(), criterion=nn.MSELoss(), 
                 opt=lambda x : optim.ASGD(x, lr=0.001), 
                 delta=1.0, k=1.0, eps= 1e-6, rl = nn.ReLU(), device=None, dtype=None):
        super().__init__(in_features, out_features, bias, device, dtype)
        self.act = act
        self.delta = delta
        self.k = k # Lipschitz constant     
        self.optFL = opt(self.parameters())
        
        self.lin = nn.Linear(out_features, fin_features, bias=False, device=device, dtype=dtype)
        self.optLR = opt(self.lin.parameters())
        
        self.criterion = criterion
        self.rl = rl
        self.eps = eps
        
    def make_sch(self,scheduler):
        self.sLR = scheduler(self.optLR)
        self.sFL = scheduler(self.optFL)
    
    def sch_step(self):
        self.sLR.step()
        self.sFL.step()
    
    def sch_rate(self):
        return self.sLR.get_lr() # assuming both schedulers have the same rate
    
    def localizer(self, x):
        # x_direction = x / (x.norm(2, 1, keepdim=True) + 1e-4)
        if self.bias is None:
            return self.act(
                torch.mm(x, self.weight.T))
        elif len(x.size()) > 1 and x.size(0) > 1:
            return self.act(
                torch.mm(x, self.weight.T) +
                self.bias.unsqueeze(0))
        return self.act(
            torch.mm(x, self.weight.T) + self.bias)
        
    def linear(self, x):
        return self.lin(x)
        
    def forward(self, x):
        return self.lin(self.localizer(x))
    
    def train(self, x_p, x_n, y, augment_rate:int=2, **kwargs):
        augment_rate = int(max(augment_rate,2))
        
        # augment x_p data with passive data
        xpa = self.passive_augment(x_p, n=augment_rate-1)
        x_p_aug = torch.concat([x_p,xpa], dim=0)
        y_rpt = y.repeat(augment_rate,1)
        
        # augment x_n data with active data
        # print(x_n.size() if x_n is not None else None)
        xna = self.active_negative_augment(x_p, n=augment_rate-1)
        x_n_aug = xna if x_n is None else torch.concat([x_n,xna], dim=0)
        yna = y.repeat(augment_rate-1,1)
        y_rptn = yna if x_n is None else torch.concat([y[:x_n.size(0)],yna],dim=0)
        
        # forward pass
        xin_p = self.localizer(x_p_aug)
        xin_n = self.localizer(x_n_aug)
        y_hat_p = self.lin(xin_p)
        y_hat_n = self.lin(xin_n)    
                
        # # print all sizes
        # print('x_p_aug', x_p_aug.size())
        # print('y_rpt', y_rpt.size())
        # print('x_n_aug', x_n_aug.size())
        # print('y_rptn', y_rptn.size())
        # print('xin_p', xin_p.size())
        # print('xin_n', xin_n.size())
        # print('y_hat_p', y_hat_p.size())
        # print('y_hat_n', y_hat_n.size())
                
        # train Linear Regression (LR) layer        
        loss0 = self.criterion(y_hat_p,y_rpt) # - self.criterion(y_hat_n,y)
        lossLR = self._update(self.optLR, loss0)
        
        # detach all for second forward pass? for now just recomputing. TODO
        xin_p_2 = xin_p.detach().requires_grad_(True)#self.localizer(x_p_aug)
        xin_n_2 = xin_n.detach().requires_grad_(True)#self.localizer(x_n_aug)
        y_hat_p_2 = y_hat_p.detach().requires_grad_(True)#self.lin(xin_p_2)
        y_hat_n_2 = y_hat_n.detach().requires_grad_(True)#self.lin(xin_n_2)                 
    
        
        # lift function to relational space     
        
        lifted_y_hat_p = torch.concat([y_hat_p_2,xin_p_2], dim=1)
        lifted_y_p = torch.concat([y_rpt,xin_p_2], dim=1)
        lifted_y_hat_n = torch.concat([y_hat_n_2,xin_n_2], dim=1)
        lifted_y_n = torch.concat([y_rptn,xin_n_2], dim=1)  

        # train Forward Localizer (FL) layer
        vp = self.criterion(lifted_y_hat_p,lifted_y_p) - self.k*self.delta
        vn = self.k*self.delta - self.criterion(lifted_y_hat_n,lifted_y_n)

        loss1 = self.rl(vp) + self.rl(vn)
        lossFL = self._update(self.optFL, loss1)
        
        
        # recompute again TODO
        xin_p_3 = self.localizer(x_p_aug).detach()
        xin_n_3 = self.localizer(x_n_aug).detach()
        y_hat_p_3 = self.lin(xin_p_3).detach()

        return xin_p_3, \
               xin_n_3, \
               y_rpt-y_hat_p_3, \
               lossFL, lossLR

    def _update(self, arg0, loss, retain_graph=False):
        arg0.zero_grad()
        loss.backward(retain_graph=retain_graph)
        arg0.step()
        return loss.item()
    
    def passive_augment(self, x, n):
        # generate n noisy copies of x to create passive data, 
        # noise is sampled from a uniform distribution of size delta
        x_aug = x.repeat(n,1)
        x_aug += torch.randn_like(x_aug)*self.delta
        return x_aug
    
    def active_negative_augment(self, x, n):
        # generate n noisy copies of x to create active data,
        # given that the noisy copies are just outside the delta-ball of x
        x_aug = x.repeat(n,1)
        noise =  torch.randn_like(x_aug)*self.delta
        direction = noise / (noise.norm(2, 1, keepdim=True) + self.eps)
        x_aug += direction * (self.delta + self.eps)
        return x_aug
        
    
