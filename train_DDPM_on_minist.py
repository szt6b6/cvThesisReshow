from matplotlib import pyplot as plt
from models.DDPM_simple import Diffusion_simple, trainDDPM_unet
import matplotlib.pyplot as plt
import numpy as np    
import torch
from utils import *

def train(diffusion):

    # 定义epoch，学习率， 损失， 默认使用Adam方法反向传播
    from models.Unet import UNet_DDPM
    net =  UNet_DDPM(1, 1, time_steps, device).to(device) # 1个通道 支持任意图像大小
    epochs = 100
    lr = 3e-4
    loss_func = torch.nn.MSELoss()
    optimizer = torch.optim.AdamW(net.parameters(), lr=lr)
    # 训练网络

    _, test_dataloader = load_minist_data() # 返回有训练集和测试集
    trainDDPM_unet(device, test_dataloader, epochs, diffusion, net, loss_func, optimizer)


if __name__ == '__main__':
    time_steps=256
    beta_start=0.0001
    beta_end=0.02
    img_size=32
    device="cuda"

    diffusion = Diffusion_simple(1, time_steps, beta_start, beta_end, img_size, device)

    # train(diffusion)

    # 测试网络 使用minist测试集图片训练100个epochs后的结果 能够输出类似数字的图片了
    net = torch.load("pre_trained\\ddpm_unet_100epochs.pth").to(device)
    x = diffusion.sample_noise_back_to_image(net, 1)
    # 1, 1, 32, 32的输入和输出
    x = np.array((x[0][0]).cpu().numpy())
    plt.imshow(x)
    plt.show()