import torch
from torch import nn


class Lenet(nn.Module):
    def __init__(self) -> None:
        super(Lenet, self).__init__()


        self.conv_Layers = nn.Sequential(nn.Conv2d(1, 6, 5, 1), #32x32x1->28x28x6
                                         nn.ReLU(),
                                         nn.AvgPool2d(2, 2),#28x28x6 -> 14x14x6
                                         nn.Conv2d(6, 16, 5, 1), #14x14x6 -> 10x10x16
                                         nn.ReLU(),
                                         nn.AvgPool2d(2, 2), #5x5x16
                                         )
        
        self.linear_layers = nn.Sequential(nn.Linear(400, 120),
                                           nn.ReLU(),
                                           nn.Dropout(), #加的dropout
                                           nn.Linear(120, 84),
                                           nn.ReLU(),
                                           nn.Linear(84, 10)) #我这里把sigmoid都换成了relu，还加了一层dropout
        

    def forward(self, x):
        return self.linear_layers(torch.flatten(self.conv_Layers(x), 1))
    



if __name__ == '__main__':
    net = Lenet()

    x = torch.randn(1, 1, 32, 32)
    print(net(x))
