import torch as th
import torch.nn as nn

class Unet_module(nn.Module):
    def __init__(self, kernel_size, de_kernel_size, channel_list, down_up='down'):
        super(Unet_module, self).__init__()
        self.conv1 = nn.Conv2d(channel_list[0], channel_list[1], kernel_size, 1, (kernel_size - 1) // 2)
        self.conv2 = nn.Conv2d(channel_list[1], channel_list[2], kernel_size, 1, (kernel_size - 1) // 2)
        self.relu1 = nn.PReLU()
        self.relu2 = nn.PReLU()
        self.bn1 = nn.BatchNorm2d(channel_list[1])
        self.bn2 = nn.BatchNorm2d(channel_list[2])
        self.bridge_conv = nn.Conv2d(channel_list[0], channel_list[-1], kernel_size, 1, (kernel_size - 1) // 2)

        if down_up == 'down':
            self.sample = nn.Sequential(
                nn.Conv2d(channel_list[2], channel_list[2], de_kernel_size, 2, (de_kernel_size - 1) // 2, 1),
                nn.BatchNorm2d(channel_list[2]), nn.PReLU())
        else:
            self.sample = nn.Sequential(
                nn.ConvTranspose2d(channel_list[2], channel_list[2], de_kernel_size, 2, (de_kernel_size - 1) // 2),
                nn.BatchNorm2d(channel_list[2]), nn.ReLU())

    def forward(self, x):
        res = self.bridge_conv(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = x + res
        next_layer = self.sample(x)

        return next_layer, x
    
class de_conv_module(nn.Module):
    def __init__(self, kernel_size, de_kernel_size, channel_list, down_up='down'):
        super().__init__()
        self.conv1 = nn.Conv2d(channel_list[0], channel_list[1], kernel_size, 1, (kernel_size - 1) // 2)
        self.conv2 = nn.Conv2d(channel_list[1], channel_list[2], kernel_size, 1, (kernel_size - 1) // 2)
        self.relu1 = nn.PReLU()
        self.relu2 = nn.PReLU()
        self.bn1 = nn.BatchNorm2d(channel_list[1])
        self.bn2 = nn.BatchNorm2d(channel_list[2])
        self.bridge_conv = nn.Conv2d(channel_list[0], channel_list[-1], kernel_size, 1, (kernel_size - 1) // 2)

        if down_up == 'down':
            self.sample = nn.Sequential(
                nn.Conv2d(channel_list[2], channel_list[2], de_kernel_size, 2, (de_kernel_size - 1) // 2, 1),
                nn.BatchNorm2d(channel_list[2]), nn.PReLU())
        else:
            self.sample = nn.Sequential(
                nn.ConvTranspose2d(channel_list[2], channel_list[2], de_kernel_size, 2, (de_kernel_size - 1) // 2),
                nn.BatchNorm2d(channel_list[2]), nn.ReLU())

    def forward(self, x, x1):
        x = th.cat([x, x1], dim=1)
        res = self.bridge_conv(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = x + res
        next_layer = self.sample(x)

        return next_layer

class FCN_2D(nn.Module):
    def __init__(self, in_channel, layers):
        super().__init__()
        # channel=2
        self.conv1 = nn.Sequential(nn.Conv2d(in_channel, layers, 5, 1, padding=2), nn.BatchNorm2d(layers), nn.PReLU())
        self.conv2 = nn.Sequential(nn.Conv2d(layers, layers * 2, 2, 2, padding=0), nn.BatchNorm2d(layers * 2),
                                   nn.PReLU())
        self.conv3 = Unet_module(5, 2, [layers * 2, layers * 2, layers * 4], 'down')
        self.conv4 = Unet_module(5, 2, [layers * 4, layers * 4, layers * 8], 'down')
        self.conv5 = Unet_module(5, 2, [layers * 8, layers * 8, layers * 16], 'down')

        self.de_conv1 = Unet_module(5, 2, [layers * 16, layers * 32, layers * 16], down_up='up')
        self.de_conv2 = de_conv_module(5, 2, [layers * 32, layers * 8, layers * 8], down_up='up')
        self.de_conv3 = de_conv_module(5, 2, [layers * 16, layers * 4, layers * 4], down_up='up')
        self.de_conv4 = de_conv_module(5, 2, [layers * 8, layers * 2, layers], down_up='up')

        self.last_conv = nn.Conv2d(layers * 2, 1, 1, 1, padding=0)

    def forward(self, x):
        x = self.conv1(x)
        x_1 = x
        x = self.conv2(x)
        x, x_2 = self.conv3(x)
        x, x_3 = self.conv4(x)
        x, x_4 = self.conv5(x)

        x, _ = self.de_conv1(x)
        x = self.de_conv2(x, x_4)
        x = self.de_conv3(x, x_3)
        x = self.de_conv4(x, x_2)

        x = th.cat([x, x_1], dim=1)
        output = self.last_conv(x)
        return output
