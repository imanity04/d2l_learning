import torch
from torch import nn
from d2l import torch as d2l

net = nn.Sequential(nn.LazyLinear(256), nn.ReLU(), nn.LazyLinear(10))

print(net[0].weight)

X = torch.rand(2, 20)
net(X)

print(net[0].weight.shape)

def apply_init(net, inputs, init=None):
    net.forward(*inputs)
    if init is not None:
        net.apply(init)

def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

net = nn.Sequential(nn.LazyLinear(256), nn.ReLU(), nn.LazyLinear(10))
apply_init(net,[X], init_weights)

print(net[0].weight[0, :5]) 
print(net[0].bias[0]) 