import torch
from torch import nn
from d2l import torch as d2l

class AlexNet(d2l.Classifier):
    def __init__(self, lr=0.1, num_classes=10):
        super().__init__()
        self.save_hyperparameters()
        self.net = nn.Sequential(
            nn.LazyConv2d(out_channels=96, kernel_size=11, stride=4),nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.LazyConv2d(out_channels=256, kernel_size=5, padding=2),nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.LazyConv2d(out_channels=384, kernel_size=3, padding=1),nn.ReLU(),
            nn.LazyConv2d(out_channels=384, kernel_size=3, padding=1),nn.ReLU(),
            nn.LazyConv2d(out_channels=256, kernel_size=3, padding=1),nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Flatten(),
            nn.LazyLinear(out_features=4096),nn.ReLU(),nn.Dropout(p=0.5),
            nn.LazyLinear(out_features=4096),nn.ReLU(),nn.Dropout(p=0.5),
            nn.LazyLinear(out_features=num_classes)
        )

AlexNet().layer_summary((1, 1, 224, 224))

model = AlexNet(lr=0.01)
data = d2l.FashionMNIST(batch_size=128, resize=(224, 224))
trainer = d2l.Trainer(max_epochs=10, num_gpus=1)
model.apply_init([next(iter(data.get_dataloader(True)))[0]], d2l.init_cnn)

trainer.fit(model, data)
d2l.plt.ioff()
d2l.plt.show()