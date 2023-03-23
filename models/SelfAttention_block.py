'''
    自注意力pytorch代码实现
    ref:
    知乎: https://zhuanlan.zhihu.com/p/48508221
    CSDN: https://blog.csdn.net/qq_43152622/article/details/118876385
'''

import torch
from math import sqrt

class SelfAttention_Block(torch.nn.Module):
    def __init__(self, emb_len, dk, dv) -> None:
        super(SelfAttention_Block, self).__init__()

        self.Wq = torch.nn.Linear(emb_len, dk)
        self.Wk = torch.nn.Linear(emb_len, dk)
        self.Wv = torch.nn.Linear(emb_len, dv)

        self.soft_max = torch.nn.Softmax(dim=2)

        self.W0 = torch.nn.Linear(dv, emb_len) #根据需要 映射成想要的形状 这里映射成emb_len
        self.dk = dk
        self.dv = dv

    def forward(self, x):
        Q = self.Wq(x) # -> batchs, sequence_len, dk
        K = self.Wk(x) # -> batchs, sequence_len, dk
        V = self.Wv(x) # -> batchs, sequence_len, dv

        res = torch.matmul(Q, K.permute(0, 2, 1))  # -> batchs, sequence_len, sequence_len
        # 上面这行等价于 res = torch.matmul(Q, K.transpose(1, 2))
        res = res / sqrt(self.dk)
        res = self.soft_max(res)
        res = torch.matmul(res, V) # -> batchs, sequence_len, dv

        res = self.W0(res) # -> batchs, sequence_len, emb_len
        return res


class SelfAttention_Block_MHA(torch.nn.Module):
    def __init__(self, emb_len, dk, dv) -> None:
        super(SelfAttention_Block_MHA, self).__init__()

        # multi-head-attention 就是使用多组不同的Wq,Wk,Wv得到不同的Q,K,V,
        # 再计算出各自的最后再concat起来再进行一次线性变换
        self.Wq1 = torch.nn.Linear(emb_len, dk)
        self.Wk1 = torch.nn.Linear(emb_len, dk)
        self.Wv1 = torch.nn.Linear(emb_len, dv)

        self.Wq2 = torch.nn.Linear(emb_len, dk)
        self.Wk2 = torch.nn.Linear(emb_len, dk)
        self.Wv2 = torch.nn.Linear(emb_len, dv)

        self.Wq3 = torch.nn.Linear(emb_len, dk)
        self.Wk3 = torch.nn.Linear(emb_len, dk)
        self.Wv3 = torch.nn.Linear(emb_len, dv)

        self.soft_max = torch.nn.Softmax(dim=2)

        self.W0 = torch.nn.Linear(3*dv, emb_len) #根据需要 映射成想要的形状 这里映射成emb_len
        self.dk = dk
        self.dv = dv

    def forward(self, x):
        Q1 = self.Wq1(x) # -> batchs, sequence_len, dk
        K1 = self.Wk1(x) # -> batchs, sequence_len, dk
        V1 = self.Wv1(x) # -> batchs, sequence_len, dv

        Q2 = self.Wq2(x) 
        K2 = self.Wk2(x) 
        V2 = self.Wv2(x) 

        Q3 = self.Wq3(x) 
        K3 = self.Wk3(x) 
        V3 = self.Wv3(x) 

        res1 = torch.matmul(self.soft_max(torch.matmul(Q1, K1.permute(0, 2, 1)) / sqrt(self.dk)), V1) # -> batchs, sequence_len, dv
        res2 = torch.matmul(self.soft_max(torch.matmul(Q2, K2.permute(0, 2, 1)) / sqrt(self.dk)), V2) 
        res3 = torch.matmul(self.soft_max(torch.matmul(Q3, K3.permute(0, 2, 1)) / sqrt(self.dk)), V3) 
        res = torch.concatenate([res1, res2, res3], dim=2)# -> batchs, sequence_len, 3*dv

        res = self.W0(res)
        return res
    
if __name__ == "__main__":
    # (batchs, sequence_len, emb_len)
    x = torch.randn(1, 5, 7)
    # net = SelfAttention_Block(emb_len=x.shape[2], dk=4, dv=3)
    net = SelfAttention_Block_MHA(emb_len=x.shape[2], dk=4, dv=3) #test Multi-Head-Attention at 3 heads

    res = net(x)
    print(res)
