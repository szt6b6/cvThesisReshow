'''
    This realized denoise diffusion probabilistic model.
    The basic principle refers thesis: (https://arxiv.org/pdf/2006.11239.pdf)
    And Zhihu: https://zhuanlan.zhihu.com/p/572770333
    The codes refer: https://github.com/dome272/Diffusion-Models-pytorch/blob/main/ddpm.py
    In total, given a img then:
    forward: img +noise_1-> noise_2 -> noise_2 -> ... -> + noise_n -> img_n
    backward: img_1 <- img_2 - predicted_noise_2 ... <- img_n-1 - predicted_noise_n-2 <- img_n - predicetd_noise_n

    The training and sample processed refer to Alogrithm 1 and 2 in thesis.
'''

import torch
import cv2
import torchvision
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from matplotlib import pyplot as plt
# from Unet import UNet_DDPM

class Diffusion_simple:
    # Init all values used in diffusion process
    def __init__(self, in_channel=3, time_steps=1000, beta_start=1e-4, beta_end=0.02, img_size=512, device="cuda") -> None:
        self.in_c = in_channel
        self.time_steps = time_steps
        self.device = device
        self.img_size = img_size

        self.betas = torch.linspace(beta_start, beta_end, self.time_steps).to(device)
        self.alphas = 1 - self.betas
        self.alphas_hat = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_hat = torch.sqrt(self.alphas_hat)
        self.sqrt_one_minus_alphas_hat = torch.sqrt(1-self.alphas_hat)

        pass


    '''
        According the formular, see in refer, we can get noisedImg at time t by using x_0
    '''
    def get_noisedImg_and_noise_at_t(self, x, t):
        if(isinstance(t, int)):
            t = [t]
        # generate standard gaussian noise.
        noise = torch.randn_like(x).to(self.device)
        # according to x0 and noise to get x_t directly.
        noisedImg = self.sqrt_alphas_hat[t][:, None, None, None] * x + self.sqrt_one_minus_alphas_hat[t][:, None, None, None] * noise
        
        # return noisedImg and noise at time step t
        # q: 这里的noise应该不能固定下来，要不然网络应该会最终趋向于这个noise
        return noisedImg, noise

    # 随机获得time t
    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.time_steps, size=(n,))
    
    '''
        backward process, turn random noise into a image
        model->pre-trained unet model, n->image number

        Solving.....
    '''
    def sample_noise_back_to_image(self, model, n=1):

        x = torch.randn((n, self.in_c, self.img_size, self.img_size)).to(self.device)
        noise = None
        with torch.no_grad():
            for t in reversed(range(self.time_steps)):
                t = (torch.ones(n) * t).long().to(self.device)
                predicted_noises = model(x, t) # 用网络在t时间步的噪声图预测出该时刻的噪声
                alpha = self.alphas[t][:, None, None, None]
                alpha_hat = self.alphas_hat[t][:, None, None, None]
                beta = self.betas[t][:, None, None, None]

                if t > 1:
                    noise = torch.randn_like(predicted_noises)
                else:
                    noise = torch.zeros_like(predicted_noises)
                
                # 根据公式获得t-1时间步的噪声图
                x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noises) + torch.sqrt(beta) * noise

        x = (x.clamp(-1, 1) + 1) / 2
        return x


def get_data(img_size, dataset_path, batch_size):
    transforms = torchvision.transforms.Compose([
        # torchvision.transforms.Resize(80),  # args.image_size + 1/4 *args.image_size
        torchvision.transforms.Resize((img_size, img_size)), # args.image_size
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    dataset = torchvision.datasets.ImageFolder(dataset_path, transform=transforms)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader


# train Unet to predict noise
def trainDDPM_unet(device, dataloader, epochs, diffusion, net, loss_func, optimizer):

    for epoch in range(epochs):
        totalLoss = 0
        pbar = tqdm(dataloader)
        for i, (images, _) in enumerate(pbar): # 这里（images, _)这样写能把列表2x()变成2x()四维tensor
            
            # 随机产生时间点 t
            t = diffusion.sample_timesteps(images.shape[0]).to(device)
            images = images.to(device)
            # 获得t时刻的噪声图片和噪声
            noised_imgs, noises = diffusion.get_noisedImg_and_noise_at_t(images, t)
            # 将噪声图片和时刻t传入网络获得返回的预测噪声
            predict_noises = net(noised_imgs, t)
            loss = loss_func(noises, predict_noises)
            net.zero_grad()

            loss.backward()
            optimizer.step()

            pbar.set_postfix(MSE=loss.item())

            totalLoss += loss
        print("epoch: %d, loss: %.3f" % (epoch, totalLoss))
    torch.save(net, "pre_trained\\"+net._get_name()+".pth")

if __name__ == "__main__":

    time_steps=256
    beta_start=0.0001
    beta_end=0.02
    img_size=256
    device="cuda"

    diffusion = Diffusion_simple(3, time_steps, beta_start, beta_end, img_size, device)
    
    imgdir_path = "C:\\Users\\cls\\Pictures\\Saved Pictures\\test"
    dataloader = get_data(img_size=img_size, dataset_path=imgdir_path, batch_size=16)
    # 定义epoch，学习率， 损失， 默认使用Adam方法反向传播
    from Unet import UNet_DDPM
    net =  UNet_DDPM(3, 3, time_steps, device).to(device)
    epochs = 100
    lr = 3e-4
    loss_func = torch.nn.MSELoss()
    optimizer = torch.optim.AdamW(net.parameters(), lr=lr)
    # 训练网络
    trainDDPM_unet(device, dataloader, epochs, diffusion, net, loss_func, optimizer)

    # 测试网络
    x = diffusion.sample_noise_back_to_image(net, 1)

    x = np.array(x[0].permute(1,2,0).cpu().numpy()*255, np.uint8)
    cv2.imshow("res", x)
    cv2.waitKey(0)