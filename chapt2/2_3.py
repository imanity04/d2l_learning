import torch

"""" 标量 """
a = torch.tensor(2.)
print(a)

"""" 向量 """
x = torch.arange(3)
print(x)
print(len(x))
print(x.shape)

"""" 矩阵 """
A = torch.arange(6).reshape((2,3))
print(A)
print(A.T)  # A的转置
""" 
A为对称阵
[[1,2,3],
 [2,0,4],
 [3,4,5]]
 """
A = torch.tensor([[1,2,3],[2,0,4],[3,4,5]])
print(A == A.T)

""" 张量 """
A = torch.arange(6,dtype=torch.float32).reshape(2,3)
B = A.clone()
print(A+B)
print(A*B)

""" cumsum的用法 """
a = torch.tensor([1,2,3,4])
print(a.cumsum(axis=0))

b = torch.arange(6).reshape(2,3)
print(b)
print(b.cumsum(axis = 0))
print(b.cumsum(axis = 1))

""" 点积 """
x = torch.tensor([0.,1,2])
y = torch.ones(3,dtype = torch.float32)
print(torch.dot(x,y))   # 注意dot两个向量的dtype要一致

print(torch.mv(A,x))
print(A@x)

""" 范数 """
u = torch.tensor([3.,-4.])
print(torch.norm(u))    # 欧几里得范数

print(torch.norm(torch.ones(4,9)))  # 费罗贝尼乌斯范数
