import numpy as np
import torch
from torch import nn
from d2l import torch as d2l

class LinearRegression(d2l.Module):  #@save
    """The linear regression model implemented with high-level APIs."""
    def __init__(self, lr):
        super().__init__()
        self.save_hyperparameters()
        self.net = nn.LazyLinear(1)
        self.net.weight.data.normal_(0, 0.01)
        self.net.bias.data.fill_(0)

@d2l.add_to_class(LinearRegression)  #@save
def forward(self, X):
    return self.net(X)

@d2l.add_to_class(LinearRegression)  #@save
def loss(self, y_hat, y):
    fn = nn.MSELoss()
    return fn(y_hat, y)

@d2l.add_to_class(LinearRegression)  #@save
def configure_optimizers(self):
    return torch.optim.SGD(self.parameters(), self.lr)

# 1. 实例化线性回归模型，学习率设置为0.03
model = LinearRegression(lr=0.03)
# 2. 生成合成回归数据集，真实权重w=[2, -3.4]，真实偏置b=4.2
data = d2l.SyntheticRegressionData(w=torch.tensor([2, -3.4]), b=4.2)
# 3. 实例化训练器，最大训练轮数3个epoch
trainer = d2l.Trainer(max_epochs=3)
# 4. 启动训练
trainer.fit(model, data)

d2l.plt.ioff()
d2l.plt.show()

@d2l.add_to_class(LinearRegression)  #@save
def get_w_b(self):
    return (self.net.weight.data, self.net.bias.data)

# 获取模型学到的参数
w, b = model.get_w_b()
# 计算与真实参数的误差

print(w.shape,data.w.shape)
print(f'error in estimating w: {data.w - w}')   # 直接广播就行(因为这里的w是一维向量)
# print(f'error in estimating w: {data.w - w.reshape(data.w.shape)}') # 如果w不是一维向量，那么也不能用这个，得用Transpose转置w矩阵
print(f'error in estimating b: {data.b - b}')