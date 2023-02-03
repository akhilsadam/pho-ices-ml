import torch
import torch.nn as nn
# credit to @mohammadpz on GitHub for the more efficient implementation
class FFALayer(nn.Linear):
    def __init__(self, in_features, out_features,
                 bias=True, act=torch.nn.ReLU(), threshold=0.5, device=None, dtype=None):
        super().__init__(in_features, out_features, bias, device, dtype)
        self.act = act
        self.threshold = threshold

    def forward(self, x):
        x_direction = x / (x.norm(2, 1, keepdim=True) + 1e-4)
        if self.bias is None:
            return self.act(
                torch.mm(x_direction, self.weight.T))
        return self.act(
            torch.mm(x_direction, self.weight.T) +
            self.bias.unsqueeze(0))

    def train(self, x_pos, x_neg, opt, reg=lambda x,y: 0):
        # compute goodness for batches
        g_pos = self.forward(x_pos).pow(2).mean(1)
        g_neg = self.forward(x_neg).pow(2).mean(1)
        # The following loss pushes pos (neg) samples to
        # values larger (smaller) than the self.threshold.
        loss = torch.log(1 + torch.exp(torch.cat([
            -g_pos + self.threshold,
            g_neg - self.threshold])) + reg(g_pos,g_neg)).mean()
        opt.zero_grad()
        # this backward just computes the derivative and is not considered backpropagation.
        loss.backward()
        opt.step()
        return self.forward(x_pos).detach(), self.forward(x_neg).detach(), loss.item()