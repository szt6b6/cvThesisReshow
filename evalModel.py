import torch
from torchvision import transforms
import cv2

if __name__ == "__main__":

    """
        1.读入图片 并转成灰度图且resize到指定输入大小 CHW -> 1x28x28
        2.转换成tensor, 增加padding 1x28x28 -> 1x32x32, 且减去均值再除以方差进行归一化
        3.输入unsqueeze到1x1x32x32, 加载训练好的网络网络, 再计算获得输出
    """
    
    img = cv2.resize(cv2.imread("imgs\\hand_write_5.png", cv2.COLOR_BGR2BGRA), (28, 28)) 

    trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.Pad(2),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

    input = torch.unsqueeze(trans(img), 0).to(device)

    net = torch.load("pre_trained\\Lenet.pth").to(device)
    res = net(input)

    print("the number is %d" % torch.argmax(res).item())