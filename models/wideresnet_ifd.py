import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, dropRate=0.0):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = dropRate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                               padding=0, bias=False) or None
    def forward(self, x):
        if not self.equalInOut:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)
        return torch.add(x if self.equalInOut else self.convShortcut(x), out)

class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, dropRate=0.0):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, dropRate)
    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, dropRate):
        layers = []
        for i in range(int(nb_layers)):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, dropRate))
        return nn.Sequential(*layers)
    def forward(self, x):
        return self.layer(x)

class WideResNet(nn.Module):
    def __init__(self, depth, in_channels, num_classes, widen_factor, dropRate, mean, std):
        super(WideResNet, self).__init__()
        nChannels = [16, 16*widen_factor, 32*widen_factor, 64*widen_factor]
        assert((depth - 4) % 6 == 0)
        self.n = (depth - 4) / 6
        self.block = BasicBlock

        # Add parameters
        self.in_channels = in_channels
        self.mean = mean
        self.std = std

        # 1st conv before any network block
        self.conv1 = nn.Conv2d(self.in_channels, nChannels[0], kernel_size=3, stride=1, padding=1, bias=False)
        # 1st block
        self.block1 = NetworkBlock(self.n, nChannels[0], nChannels[1], self.block, 1, dropRate)
        # 2nd block
        self.block2 = NetworkBlock(self.n, nChannels[1], nChannels[2], self.block, 2, dropRate)
        # 3rd block
        self.block3 = NetworkBlock(self.n, nChannels[2], nChannels[3], self.block, 2, dropRate)

        self.bn1 = nn.BatchNorm2d(nChannels[3])
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(nChannels[3], num_classes)
        self.nChannels = nChannels[3]


    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x, is_feat):
        out = (x-self.mean) / self.std
        out = self.conv1(out)
        out = self.block1(out)
        feature1 = out
        out = self.block2(out)
        feature2 = out
        out = self.block3(out)
        feature3 = out
        out = self.relu(self.bn1(out))
        out = F.avg_pool2d(out, out.shape[2])
        out = out.view(-1, self.nChannels)
        if is_feat:
            return self.fc(out), torch.cat([feature1, feature2, feature3], dim=1)
        else:
            return self.fc(out)


class WideResNet_Plain(WideResNet):

    def __init__(self, depth, in_channels, num_classes, widen_factor, dropRate, mean, std):
        super(WideResNet_Plain, self).__init__(depth, in_channels, num_classes, widen_factor, dropRate, mean, std)
        self.initialize()

    def get_inference(self, x_adv):
        logit_adv = self(x_adv).detach()
        return logit_adv

class WideResNet_IFD(WideResNet):

    def __init__(self, depth, in_channels, num_classes, widen_factor, dropRate, mean, std):
        super(WideResNet_IFD, self).__init__(depth, in_channels, num_classes, widen_factor, dropRate, mean, std)
        self.initialize()

    def forward(self, x, intermediate_propagate=0, pop=0, is_feat = False):
        if is_feat:
            out = self.conv1(x)
            out = self.block1(out)  #[2, 160, 32, 32]
            feat1 = out
            out = self.block2(out)  #[2, 320, 16, 16]
            feat2 = out
            out = self.block3(out)  #[2, 640, 8, 8]
            feat3 = out
            out = self.relu(self.bn1(out))
            out = F.avg_pool2d(out, 8)  #[2, 640]
            #feat = out.view(out.size(0), -1)
            out = out.view(-1, self.nChannels)
            return self.fc(out), [feat1, feat2, feat3]
    
        # adding important function of pop & intermediate_propagate
        if intermediate_propagate == 0:
            out = (x-self.mean) / self.std
            out = self.conv1(out)
            out = self.block1(out)
            if pop == 1:
                return out
            out = self.block2(out)
            if pop == 2:
                return out
            out = self.block3(out)
            out = self.bn1(out)
            if pop == 3:
                return out
            out = self.relu(out)
            out = F.avg_pool2d(out, out.shape[2])
            out = out.view(-1, self.nChannels)
            return self.fc(out)

        elif intermediate_propagate == 1:
            out = x
            out = self.block2(out)
            out = self.block3(out)
            out = self.relu(self.bn1(out))
            out = F.avg_pool2d(out, out.shape[2])
            out = out.view(-1, self.nChannels)
            return self.fc(out)

        elif intermediate_propagate == 2:
            out = x
            out = self.block3(out)
            out = self.relu(self.bn1(out))
            out = F.avg_pool2d(out, out.shape[2])
            out = out.view(-1, self.nChannels)
            return self.fc(out)

        elif intermediate_propagate == 3:
            out = x.clone()
            out = self.relu(out)
            out = F.avg_pool2d(out, out.shape[2])
            out = out.view(-1, self.nChannels)
            return self.fc(out)

    def get_inference(self, x_adv):
        logit_adv = self(x_adv).detach()
        return logit_adv
