'''
    Sequeeze and Excitation Network
    
    参考：
    论文:《Squeeze-and-Excitation Networks》
    
    论文链接: https://arxiv.org/abs/1709.01507 

    代码地址: https://github.com/hujie-frank/SENet 

    PyTorch代码地址: https://github.com/miraclewkf/SENet-PyTorch

    SENet的核心思想在于通过网络根据loss去学习特征权重, 使得有效的feature map权重大, 无效或效果小的feature map权重小的方式训练模型达到更好的结果。
    Sequeeze-and-Excitation(SE) block并不是一个完整的网络结构, 而是一个子结构, 可以嵌到其他分类或检测模型中
'''

from torch import nn
import torch



class SEBlock(nn.Module):
    def __init__(self, r, c) -> None:
        super(SEBlock, self).__init__()

        #输入是 CxHxW 要求得到Cx1x1的输出 再用Cx1x1的权重去scale输入得到CxHxW的最终输出
        self.SE_layer_pool = nn.AdaptiveAvgPool2d((1,1))

        self.SE_layer_fc = nn.Sequential(
            nn.Linear(c, c // r),
            nn.ReLU(),
            nn.Linear(c // r, c),
            nn.Sigmoid() 
        )
    
    def forward(self, x):
        b, c, _, _ = x.shape
        out = self.SE_layer_pool(x).view(b, c)
        out = self.SE_layer_fc(out).view(b, c, 1, 1)
        return x * out.expand_as(x)
    
class SE_LEnet(nn.Module):
    def __init__(self, r) -> None:
        super(SE_LEnet, self).__init__()

        self.conv_Layers = nn.Sequential(nn.Conv2d(1, 6, 5, 1), #32x32x1->28x28x6
                                         nn.ReLU(),
                                         nn.AvgPool2d(2, 2),#28x28x6 -> 14x14x6
                                         nn.Conv2d(6, 16, 5, 1), #14x14x6 -> 10x10x16
                                         nn.ReLU(),
                                         nn.AvgPool2d(2, 2), #5x5x16
                                         SEBlock(r, 16) #用一层SE Block 试试
                                         )
        
        self.linear_layers = nn.Sequential(nn.Linear(400, 120),
                                           nn.ReLU(),
                                           nn.Dropout(), #加的dropout
                                           nn.Linear(120, 84),
                                           nn.ReLU(),
                                           nn.Linear(84, 10)) #我这里把sigmoid都换成了relu，还加了一层dropout
        

    def forward(self, x):
        return self.linear_layers(torch.flatten(self.conv_Layers(x), 1))

if __name__ == "__main__":
    x = torch.randn((4, 1, 32, 32))
    block = SE_LEnet(4)

    out = block(x)

    print(out.shape)