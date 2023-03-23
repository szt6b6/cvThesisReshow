'''
    复现一下vision transformer
    ref: 
    https://zhuanlan.zhihu.com/p/438432155
    图结构来源https://blog.csdn.net/baidu_36913330/article/details/120198840
'''

import torch
from torch import nn
from SelfAttention_block import SelfAttention_Block_MHA


class Patch_Embeding(nn.Module):
    def __init__(self, in_channel, out_embeding_len, img_size=32, patch_size=4) -> None:
        super().__init__() # 写super(Patch_Embeding, self).__init__() 效果一样

        self.conv = nn.Conv2d(in_channels=in_channel, out_channels=out_embeding_len, kernel_size=patch_size, stride=patch_size) #(32-4)/4=7 -> out_embeding_len,8,8
        
        self.num_patches = (img_size // patch_size) ** 2 #(img_size-patch_size) // patch_size + 1 的平方
        self.out_embeding_len = out_embeding_len
        self.patch_size = patch_size

        self.cls_token = nn.Parameter(torch.zeros(1, 1, out_embeding_len))#经过卷积后在展平和这个concat
        self.pos_token = nn.Parameter(torch.zeros(1, self.num_patches+1, out_embeding_len))
    
    def forward(self, x):
        b, c, h, w = x.shape

        #h,w=32, patch_size=4,c=1,out_embeding_len=64
        out = self.conv(x) #b,1,32,32 -> b,64,8,8 
        
        out = out.flatten(2).transpose(1,2) # -> b, 64, 64
        
        cls_token = self.cls_token.expand(out.shape[0], -1, -1)
        out = torch.concatenate([cls_token, out], dim=1) # -> b,65, 64

        pos_token = self.pos_token
        out = out + pos_token #->b,65,64

        return out

class MLP(nn.Module):
    def __init__(self, in_dim, hidden_indim, out_dim) -> None:
        super().__init__()
        self.layer = nn.Sequential(nn.Linear(in_dim, hidden_indim),
                                   nn.GELU(), # 文中图使用的是这个
                                   nn.Dropout(0.2),
                                   nn.Linear(hidden_indim, out_dim),
                                   nn.Dropout()
                                   )
        
    def forward(self, x):
        return self.layer(x)


class Encoder_Block(nn.Module):
    def __init__(self, in_channel, hidden_indim, out_embeding_len) -> None:
        super().__init__()
        self.layer1 = nn.Sequential(
                                    nn.LayerNorm(out_embeding_len),
                                    SelfAttention_Block_MHA(out_embeding_len, dk=36, dv=36), #dk，dv是乱取得
                                    nn.Dropout(0.2)
        )
        
        self.layer2 = nn.Sequential(
                                    nn.LayerNorm(out_embeding_len),
                                    MLP(out_embeding_len, hidden_indim, out_embeding_len), #dk，dv是乱取得
                                    nn.Dropout(0.2)
        )
        
    def forward(self, x):
        temp = x + self.layer1(x)
        res = temp + self.layer2(temp)

        return res

class Vit_Transformer(nn.Module):
    def __init__(self, in_channel, hidden_indim, out_embeding_len, img_size=32, patch_size=4) -> None:
        super().__init__()
        
        # input: b, c, h, w  output: b, (img_size//patch_size)**2+1, out_embeding_len
        self.Emb = Patch_Embeding(in_channel, out_embeding_len, img_size=32, patch_size=4)
        
        self.dropout = nn.Dropout(0.2)

        # encoder论文里用了12个，这里用2个
        self.encoder = nn.Sequential(
            Encoder_Block(in_channel, hidden_indim, out_embeding_len),
            Encoder_Block(in_channel, hidden_indim, out_embeding_len)
        )

        self.layerNorm = nn.LayerNorm(out_embeding_len)

        # input 是: b, 1, out_embeding_len
        self.MLP_head = nn.Sequential(nn.Linear(out_embeding_len, 10),
                                nn.ReLU())
    def forward(self, x):
        out = self.Emb(x) #b,1,32,32 -> b,65,64
        out = self.dropout(out)
        out = self.encoder(out) #->b,65,65
        out = self.layerNorm(out)

        class_token = out[:, 0, :]

        res = self.MLP_head(class_token)
        return res


if __name__ == "__main__":

    # # test Patch_Embeding
    x = torch.randn(2, 1, 32, 32)

    # emb = Patch_Embeding(1, 64, x.shape[2], 4)

    # res = emb(x)

    # print(res.shape) # -> 2,65,64

    vit = Vit_Transformer(1, 32, 64, 32, 4)

    res = vit(x)

    print(res.shape)

    