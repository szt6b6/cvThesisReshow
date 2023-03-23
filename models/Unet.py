import torch
from torch import nn

# make modification, only the channels are changed.
# input C1xHxW, output C2xHxW
class Double_Conv(nn.Module):
    def __init__(self, in_channel, out_channel) -> None:
        super().__init__()

        self.seq = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(out_channel, out_channel, 3, 1, 1),
            nn.ReLU()
        )

    def forward(self, x):
        return self.seq(x)


# 无条件DDPM 输入:b,c,h,w 输出:b,c,h,w
class UNet_DDPM(nn.Module):
    def __init__(self, c_in, c_out, time_steps, device="cuda") -> None:
        super().__init__()

        self.device = device
        self.c_in = c_in
        self.c_out = c_out
        self.time_steps = time_steps


        self.dconv1 = Double_Conv(c_in, 64)
        self.dconv2 = Double_Conv(64, 128)
        self.dconv3 = Double_Conv(128, 256)
        self.dconv4 = Double_Conv(256, 512)

        self.down1 = nn.MaxPool2d(2, 2)
        self.down2 = nn.MaxPool2d(2, 2)
        self.down3 = nn.MaxPool2d(2, 2)
        self.down4 = nn.MaxPool2d(2, 2)

        self.up4 = nn.ConvTranspose2d(1024, 512, 2, 2, 0)
        self.up3 = nn.ConvTranspose2d(512, 256, 2, 2, 0)
        self.up2 = nn.ConvTranspose2d(256, 128, 2, 2, 0)
        self.up1 = nn.ConvTranspose2d(128, 64, 2, 2, 0)

        self.dconv4_ = Double_Conv(1024, 512)
        self.dconv3_ = Double_Conv(512, 256)
        self.dconv2_ = Double_Conv(256, 128)
        self.dconv1_ = Double_Conv(128, 64)

        self.bottom_conv = Double_Conv(512, 1024)

        self.linear_project1 = nn.Sequential(nn.Linear(self.time_steps, 128), #放在dconv2后面
                                            nn.ReLU())
        self.linear_project2 = nn.Sequential(nn.Linear(self.time_steps, 1024), #放在bottom_conv后面
                                            nn.ReLU())
        self.linear_project3 = nn.Sequential(nn.Linear(self.time_steps, 128), #放在up2后面
                                            nn.ReLU())
        self.final_conv = nn.Conv2d(64, self.c_out, 1, 1)
    
    # 参考github上的位置信息处理
    def pos_encoding(self, t, channels):
        inv_freq = 1.0 / (
            10000
            ** (torch.arange(0, channels, 2, device=self.device).float() / channels)
        )
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc



    def forward(self, x, t):
        # unsqueeze 表示在末尾扩张
        t = t.unsqueeze(-1).type(torch.float)
        t = self.pos_encoding(t, self.time_steps)

        t1 = self.linear_project1(t)
        t1 = t1.view(t1.shape[0], t1.shape[1], 1, 1)

        t2 = self.linear_project2(t)
        t2 = t2.view(t2.shape[0], t2.shape[1], 1, 1)

        t3 = self.linear_project3(t)
        t3 = t3.view(t3.shape[0], t3.shape[1], 1, 1)

        x1 = self.dconv1(x)
        x2 = self.dconv2(self.down1(x1))
        x2 = x2 + t1
        x3 = self.dconv3(self.down2(x2))
        x4 = self.dconv4(self.down3(x3))
        
        x5 = self.up4(self.bottom_conv(self.down4(x4)) + t2)
        
        x_out = self.up3(self.dconv4_(torch.concat([x4, x5], dim=1)))
        x_out = self.up2(self.dconv3_(torch.concat([x_out, x3], dim=1)))
        x_out = x_out + t3
        x_out = self.up1(self.dconv2_(torch.concat([x_out, x2], dim=1)))
        
        
        x_out = self.final_conv(self.dconv1_(torch.concat([x_out, x1], dim=1)))

        return x_out


# Set the input 3x512x512, then after down: 64x256x256->128x128x128->256x64x64->512x32x32
class UNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.dconv1 = Double_Conv(3, 64)
        self.dconv2 = Double_Conv(64, 128)
        self.dconv3 = Double_Conv(128, 256)
        self.dconv4 = Double_Conv(256, 512)

        self.down1 = nn.MaxPool2d(2, 2)
        self.down2 = nn.MaxPool2d(2, 2)
        self.down3 = nn.MaxPool2d(2, 2)
        self.down4 = nn.MaxPool2d(2, 2)

        self.up4 = nn.ConvTranspose2d(1024, 512, 2, 2, 0)
        self.up3 = nn.ConvTranspose2d(512, 256, 2, 2, 0)
        self.up2 = nn.ConvTranspose2d(256, 128, 2, 2, 0)
        self.up1 = nn.ConvTranspose2d(128, 64, 2, 2, 0)

        self.dconv4_ = Double_Conv(1024, 512)
        self.dconv3_ = Double_Conv(512, 256)
        self.dconv2_ = Double_Conv(256, 128)
        self.dconv1_ = Double_Conv(128, 64)

        self.bottom_conv = Double_Conv(512, 1024)

        
        self.final_conv = nn.Conv2d(64, 3, 1, 1)



    def forward(self, x):
        x1 = self.dconv1(x)
        x2 = self.dconv2(self.down1(x1))
        x3 = self.dconv3(self.down2(x2))
        x4 = self.dconv4(self.down3(x3))
        
        x5 = self.up4(self.bottom_conv(self.down4(x4)))
        
        x_out = self.up3(self.dconv4_(torch.concat([x4, x5], dim=1)))
        x_out = self.up2(self.dconv3_(torch.concat([x_out, x3], dim=1)))
        x_out = self.up1(self.dconv2_(torch.concat([x_out, x2], dim=1)))
        x_out = self.final_conv(self.dconv1_(torch.concat([x_out, x1], dim=1)))

        return x_out


if __name__ == "__main__":
    x = torch.randn((2, 1, 512, 512)).to("cuda")

    t = torch.randint(1, 100, (2,)).to("cuda")


    # Set the C=1, H=512, W=512
    # Have modification refet to U-net img in model_architectures.md 
    unet = UNet_DDPM(1, 1, 100).to("cuda")

    out = unet(x, t)

    print(out.shape)