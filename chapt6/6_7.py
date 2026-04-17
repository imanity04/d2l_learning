import torch
from torch import nn
from d2l import torch as d2l

# print(d2l.num_gpus())
# print(d2l.try_gpu())
# print(d2l.try_gpu(10))
# print(d2l.try_all_gpus())

X = torch.ones(2,3,device=d2l.try_gpu())
print(X)

Y = torch.rand(2, 3, device=d2l.try_gpu(1))
print(Y)

Z = Y.cuda(0)
print(X)
print(Z)

print(X+Z)
print(X.cuda(0) is X)
print(X.to(device=d2l.try_gpu()) is X)

net = nn.Sequential(nn.LazyLinear(1))
net = net.to(device=d2l.try_gpu())
print(net(X))
print(net[0].weight.data.device)
