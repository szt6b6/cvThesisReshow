import torch
from torch import nn
from torch.nn.functional import relu, avg_pool2d


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1) -> None:
        super(ResBlock, self).__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(), #inplace=True的话，指原地进行操作，操作完成后覆盖原来的变量, 默认false
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels)
        )

        #定义残差连接
        self.short_cout = nn.Sequential()
        if(stride != 1 or out_channels != in_channels):
            self.short_cout = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride), #注意看这里(a-1)//stride == (a+2-3) //stride
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.conv_layers(x)
        out = out + self.short_cout(x)
        out = relu(out)
        return out
    


class ResNet(nn.Module):
    def __init__(self) -> None:
        super(ResNet, self).__init__()

        #resnet 网络输出时3x224x224
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3), #-> 64x112x112
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1) #-> 64x56x56
        )     

        self.conv2_x = ResBlock(64, 64, stride=1) #-> 64x56x56
        self.conv3_x = ResBlock(64, 128, stride=2) #-> 128x28x28
        self.conv4_x = ResBlock(128, 256, stride=2) #-> 256x14x14
        self.cnov5_x = ResBlock(256, 512, stride=2) #-> 512x7x7

        self.linear_layer = nn.Linear(512, 1000) #Number class is 1000. Can be modified.

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2_x(out)
        out = self.conv3_x(out)
        out = self.conv4_x(out)
        out = self.cnov5_x(out)
        out = avg_pool2d(out, 7).view(out.shape[0], -1) #out.shape[0] 表示batch，共四维
        out = self.linear_layer(out)

        return out
    

#测试用minist来训练一下resnet
class ResNetForMinist(nn.Module):
    def __init__(self) -> None:
        super(ResNetForMinist, self).__init__()

        #resnet 网络输出时1x32x32
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 3, kernel_size=3, stride=2, padding=1), #-> 3x16x16
            nn.BatchNorm2d(3),
            nn.ReLU(),
        )     

        self.conv2_x = ResBlock(3, 3, stride=1) #-> 3x16x16
        self.conv3_x = ResBlock(3, 6, stride=2) #-> 6x8x8

        self.linear_layer = nn.Sequential(nn.Linear(384, 84), #Number class is 1000. Can be modified.
                                          nn.Dropout(0.5),
                                          nn.ReLU(),
                                          nn.Linear(84, 10)
                                        )

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2_x(out)
        out = self.conv3_x(out)
        out = out.view(out.shape[0], -1)
        out = self.linear_layer(out)

        return out


if __name__ == "__main__":
    # img = torch.randn(1,3,224,224)
    # net = ResNet()

    # out = net(img)

    # print(out.shape)

    img = torch.randn(1,1,32,32)
    net = ResNetForMinist()
    out = net(img)
    print(out.shape)