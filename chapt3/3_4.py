import torch
from d2l import torch as d2l

class SGD(d2l.HyperParameters):  #@save
    """Minibatch stochastic gradient descent."""
    def __init__(self, params, lr):
        self.save_hyperparameters()

    def step(self):
        for param in self.params:
            param -= self.lr * param.grad

    def zero_grad(self):
        for param in self.params:
            if param.grad is not None:
                param.grad.zero_()

class LinearRegressionScratch(d2l.Module):  #@save
    """The linear regression model implemented from scratch."""
    def __init__(self, num_inputs, lr, sigma=0.01):
        super().__init__()
        self.save_hyperparameters()
        self.w = torch.normal(0, sigma, (num_inputs, 1), requires_grad=True)
        self.b = torch.zeros(1, requires_grad=True)

    def forward(self, X):
        return torch.matmul(X, self.w) + self.b

    def loss(self, y_hat, y):
        l = (y_hat - y) ** 2 / 2
        return l.mean() # 对批次内所有样本的损失取平均值，得到批次的平均损失
    
    def configure_optimizers(self):
        return SGD([self.w, self.b], self.lr)
    
@d2l.add_to_class(d2l.Trainer)  #@save
def prepare_batch(self, batch):
    return batch

@d2l.add_to_class(d2l.Trainer)  #@save
def fit_epoch(self):
    # 1. 设置模型为训练模式
    self.model.train()
    # 2. 遍历训练集的所有批次，执行训练步骤
    for batch in self.train_dataloader:
        # 2.1 前向传播+损失计算
        loss = self.model.training_step(self.prepare_batch(batch))
        # 2.2 梯度清零
        self.optim.zero_grad()
        # 2.3 反向传播计算梯度
        with torch.no_grad():
            loss.backward()
            # 梯度裁剪（后续章节讲解，此处为预留逻辑）
            if self.gradient_clip_val > 0:
                self.clip_gradients(self.gradient_clip_val, self.model)
            # 2.4 优化器执行参数更新
            self.optim.step()
        # 训练批次计数更新
        self.train_batch_idx += 1
    # 若无验证集，直接结束本轮epoch
    if self.val_dataloader is None:
        return
    # 3. 设置模型为评估模式，执行验证步骤
    self.model.eval()
    for batch in self.val_dataloader:
        # 关闭梯度计算，仅执行前向传播
        with torch.no_grad():
            self.model.validation_step(self.prepare_batch(batch))
        # 验证批次计数更新
        self.val_batch_idx += 1

# 1. 实例化线性回归模型：2个输入特征，学习率0.03
model = LinearRegressionScratch(2, lr=0.03)
# 2. 生成合成回归数据集：真实权重w=[2, -3.4]，真实偏置b=4.2
data = d2l.SyntheticRegressionData(w=torch.tensor([2, -3.4]), b=4.2)
# 3. 实例化训练器：最大训练轮数3个epoch
trainer = d2l.Trainer(max_epochs=3)
# 4. 执行训练：传入模型与数据集
trainer.fit(model, data)

d2l.plt.ioff()
d2l.plt.show()

with torch.no_grad():
    print(f'input_w_shape{data.w.shape},output_w_shape{model.w.shape}')
    print(f'error in estimating w: {data.w - model.w.reshape(data.w.shape)}')
    print(f'error in estimating b: {data.b - model.b}')
