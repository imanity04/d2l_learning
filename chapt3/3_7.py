import torch
from torch import nn
from d2l import torch as d2l

class Data(d2l.DataModule):
    def __init__(self, num_train, num_val, num_inputs, batch_size):
        self.save_hyperparameters()
        n = num_train + num_val
        # 生成标准正态分布的特征
        self.X = torch.randn(n, num_inputs)
        # 生成高斯噪声
        noise = torch.randn(n, 1) * 0.01
        # 定义真实权重和偏置
        w, b = torch.ones((num_inputs, 1)) * 0.01, 0.05
        # 生成标签
        self.y = torch.matmul(self.X, w) + b + noise

    def get_dataloader(self, train):
        # 划分训练集和验证集
        i = slice(0, self.num_train) if train else slice(self.num_train, None)
        return self.get_tensorloader([self.X, self.y], train, i)
    
def l2_penalty(w):
    return (w ** 2).sum() / 2
    
class WeightDecayScratch(d2l.LinearRegressionScratch):
    def __init__(self, num_inputs, lambd, lr, sigma=0.01):
        # 继承基类的线性回归初始化逻辑
        super().__init__(num_inputs, lr, sigma)
        # 保存超参数，便于后续调用
        self.save_hyperparameters()

    def loss(self, y_hat, y):
        # 总损失 = 原始均方误差损失 + 正则化惩罚项
        return (super().loss(y_hat, y) +
                self.lambd * l2_penalty(self.w))
    
# 实例化数据集：20训练样本、100验证样本、200维输入、批量大小5
data = Data(num_train=20, num_val=100, num_inputs=200, batch_size=5)
# 实例化训练器，最大训练轮数10
trainer = d2l.Trainer(max_epochs=10)

def train_scratch(lambd):
    # 实例化带权重衰减的模型
    model = WeightDecayScratch(num_inputs=200, lambd=lambd, lr=0.01)
    # y轴设为对数刻度，更清晰观察损失变化
    model.board.yscale='log'
    # 模型训练
    trainer.fit(model, data)
    d2l.plt.ioff()
    d2l.plt.show()
    # 打印训练完成后权重的L2范数，直观查看权重收缩效果
    print('L2 norm of w:', float(l2_penalty(model.w)))
    
# train_scratch(3)

class WeightDecay(d2l.LinearRegression):
    def __init__(self, wd, lr):
        super().__init__(lr)
        self.save_hyperparameters()
        self.wd = wd
        

    def configure_optimizers(self):
        # 参数分组：权重启用权重衰减，偏置不启用
        return torch.optim.SGD([
            {'params': self.net.weight, 'weight_decay': self.wd},
            {'params': self.net.bias}], lr=self.lr)

model = WeightDecay(wd=3, lr=0.01)
model.board.yscale='log'
trainer.fit(model, data)

d2l.plt.ioff()
d2l.plt.show()

print('L2 norm of w:', float(l2_penalty(model.get_w_b()[0])))