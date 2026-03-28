import torch

""" 创建张量及基本操作 """
x = torch.arange(12, dtype=torch.float32)
# print(x)
# print(x.numel())
# print(x.shape)

X = x.reshape(-1, 4)
# print(X)

# print(torch.zeros((2,3,4)))

# print(torch.ones((2,3,4)))

# print(torch.randn(3,4))

# print(torch.tensor([[2, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]]))

""" 张量的索引与切片 """
# print(X[-1])
# print(X[1:3])

X[1,2] = 17
# print(X)

X[:2, :] = 12
# print(X)

""" 张量的运算操作 """
# print(torch.exp(x))

x = torch.tensor([1.0, 2, 4, 8])
y = torch.tensor([2, 2, 2, 2])
# print(x+y)

X = torch.arange(12, dtype=torch.float32).reshape((3,4))
Y = torch.tensor([[2.0, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
# print(torch.cat((X, Y), dim=0))
# print(torch.cat((X, Y), dim=1))

# print(X==Y)
# print(X.sum())

""" 广播机制 """
a = torch.arange(12).reshape(-1,2,3)    # shape:(2,2,3)
print(a)    
b = torch.tensor([1,2,3])   # shape:(3)
print(b)
print(a+b)  # b_shape:(3)-->(2,3)-->(2,2,3)