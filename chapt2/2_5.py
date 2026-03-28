import torch

x = torch.arange(4.0)

x.requires_grad_(True)
print(x)

y = 2 * torch.dot(x,x)

print(y)
y.backward()
print(x.grad)

x.grad.zero_()
y = x.sum()
y.backward()
print(x.grad)

""" 非标量变量的反向传播 """
x.grad.zero_()
y = x * x
print(y)
y.backward(gradient = torch.ones(len(y)))
print(x.grad)

x.grad.zero_()
y = x * x   # 不允许在同一张图上进行两次backward
y.sum().backward()
print(x.grad)

""" 分离计算图 """
x.grad.zero_()
y = x * x
u = y.detach()
z = x * u
z.backward(gradient = torch.ones(len(z)))
print(x.grad)

x.grad.zero_()
y.sum().backward()
print(x.grad)

""" 控制流的梯度计算 """
def f(a):
    b = a * 2
    while b.norm() < 1000:
        b = b * 2
    if b.sum() > 0:
        c = b
    else:
        c = 100 * b
    return c

a = torch.randn(size = (), requires_grad = True)
print(a)
d = f(a)
d.backward()
print(a.grad)
print(a.grad == d/a)