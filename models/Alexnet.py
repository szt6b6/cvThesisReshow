import torch
from torch import nn


class Alexnet(nn.Module):
    def __init__(self):
        super(Alexnet, self).__init__()

        self.conv_layers = nn.Sequential(
                nn.Conv2d(3, 96, 11, 4), #224 x 224 x 3 -> 54 x 54 x 96
                nn.ReLU(),
                nn.MaxPool2d(3, 2),# 54 x 54 x 96 -> 26 x 26 x96
                nn.Conv2d(96, 256, 5, 1, 2), # 26 x 26 x96 ->  26 x 26 x 256                         
                nn.ReLU(),
                nn.MaxPool2d(3, 2), # 26 x 26 x 256 -> 12 x 12 x 256 
                nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, stride=1, padding=1),# 12 x 12 x 256 -> 12 x 12 x 384
                nn.ReLU(),
                nn.Conv2d(384, 384, 3, 1, 1), # 12 x 12 x 384-> 12 x 12 x 384
                nn.ReLU(),
                nn.Conv2d(384, 256, 3, 1, 1),# 12 x 12 x 384 ->  12 x 12 x 256
                nn.ReLU(),
                nn.MaxPool2d(3, 2), # 12 x 12 x 256-> 5 x 5 x 256   
                )
        
        self.linear_layer = nn.Sequential(nn.Linear(6400, 4096),
                                          nn.ReLU(),
                                          nn.Dropout(0.5),
                                          nn.Linear(4096, 1000))   


    def forward(self, x):
        #这里是torch.flatten 不是nn.Flatten
        return self.linear_layer(torch.flatten(self.conv_layers(x), 1))
    


if __name__ == '__main__':
    net = Alexnet()
    fake_image = torch.rand((1, 3, 224, 224))
    result = net(fake_image)
    print(result.shape)