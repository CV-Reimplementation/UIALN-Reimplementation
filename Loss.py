import torch
import torch.nn as nn
import torch.nn.functional as F


# the structure-aware TV loss
class TVLoss(nn.Module):
    def __init__(self, TVLoss_weight=0.1):
        super(TVLoss, self).__init__()
        self.TVLoss_weight = TVLoss_weight
        self.lambda_g = 0.1  # 论文没给
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __call__(self, x, y):
        # 对x, y使用Sobel滤波器求其梯度
        # x, y的shape为[batch_size, 1, height, width]
        # x_grad, y_grad的shape为[batch_size, 2, height, width]
        x_grad_1 = F.conv2d(x, torch.tensor([[[[1, 0, -1], [2, 0, -2], [1, 0, -1]]]], device=self.device, dtype=torch.float), padding=1)
        x_grad_2 = F.conv2d(x, torch.tensor([[[[1, 2, 1], [0, 0, 0], [-1, -2, -1]]]], device=self.device, dtype=torch.float), padding=1)
        y_grad_2 = F.conv2d(y, torch.tensor([[[[1, 2, 1], [0, 0, 0], [-1, -2, -1]]]], device=self.device, dtype=torch.float), padding=1)
        y_grad_1 = F.conv2d(y, torch.tensor([[[[1, 0, -1], [2, 0, -2], [1, 0, -1]]]], device=self.device, dtype=torch.float), padding=1)
        x_gard = torch.cat((x_grad_1, x_grad_2), dim=1)
        y_gard = torch.cat((y_grad_1, y_grad_2), dim=1)
        loss = x_gard * torch.exp(-self.lambda_g * y_gard)
        # 求loss的L1范数
        loss = torch.sum(torch.abs(loss))
        return loss * self.TVLoss_weight


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.randn(1, 1, 256, 256).cuda()
    y = torch.randn(1, 1, 256, 256).cuda()
    tv_loss = TVLoss()
    loss = tv_loss(x, y)
    print(loss)