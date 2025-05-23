'''Pre-activation ResNet in PyTorch.
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Identity Mappings in Deep Residual Networks. arXiv:1603.05027
'''
import torch
import torch.nn as nn
import torch.nn.functional as F


class PreActBlock(nn.Module):
    '''Pre-activation version of the BasicBlock.'''
    expansion = 1  

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out += shortcut
        return out


class PreActBottleneck(nn.Module):
    '''Pre-activation version of the original Bottleneck module.'''
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out = self.conv3(F.relu(self.bn3(out)))
        out += shortcut
        return out


class PreActResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(PreActResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.bn = nn.BatchNorm2d(512 * block.expansion)
        self.linear = nn.Linear(512*block.expansion, num_classes)#512

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.relu(self.bn(out))
        #out = F.avg_pool2d(out, 4)
        out = F.adaptive_avg_pool2d(out, 1)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

class PreActResNet_IFD(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(PreActResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.bn = nn.BatchNorm2d(512 * block.expansion)
        self.linear = nn.Linear(512*block.expansion, num_classes)#512

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.relu(self.bn(out))
        #out = F.avg_pool2d(out, 4)
        out = F.adaptive_avg_pool2d(out, 1)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


class PreActResNet_IFD(PreActResNet):
    def __init__(self, block, num_blocks, num_classes=10):
        super(PreActResNet_IFD, self).__init__(block, num_blocks, num_classes)

    def forward(self, x, intermediate_propagate=0, pop=0, is_feat=False):
        if is_feat:
            out = self.conv1(x)
            out = self.layer1(out)
            feat1 = out
            out = self.layer2(out)
            feat2 = out
            out = self.layer3(out)
            feat3 = out
            out = self.layer4(out)
            feat4 = out
            out = F.relu(self.bn(out))
            out = F.adaptive_avg_pool2d(out, 1)
            out = out.view(out.size(0), -1)
            out = self.linear(out)
            return out, [feat1, feat2, feat3, feat4]

        # Default full forward
        if intermediate_propagate == 0:
            out = self.conv1(x)
            out = self.layer1(out)
            if pop == 1:
                return out
            out = self.layer2(out)
            if pop == 2:
                return out
            out = self.layer3(out)
            if pop == 3:
                return out
            out = self.layer4(out)
            if pop == 4:
                return out
            out = F.relu(self.bn(out))
            out = F.adaptive_avg_pool2d(out, 1)
            out = out.view(out.size(0), -1)
            return self.linear(out)

        # Start from layer2
        elif intermediate_propagate == 1:
            out = self.layer2(x)
            out = self.layer3(out)
            out = self.layer4(out)
            out = F.relu(self.bn(out))
            out = F.adaptive_avg_pool2d(out, 1)
            out = out.view(out.size(0), -1)
            return self.linear(out)

        # Start from layer3
        elif intermediate_propagate == 2:
            out = self.layer3(x)
            out = self.layer4(out)
            out = F.relu(self.bn(out))
            out = F.adaptive_avg_pool2d(out, 1)
            out = out.view(out.size(0), -1)
            return self.linear(out)

        # Start from layer4
        elif intermediate_propagate == 3:
            out = self.layer4(x)
            out = F.relu(self.bn(out))
            out = F.adaptive_avg_pool2d(out, 1)
            out = out.view(out.size(0), -1)
            return self.linear(out)

        # Start from bn + relu
        elif intermediate_propagate == 4:
            out = F.relu(x)
            out = F.adaptive_avg_pool2d(out, 1)
            out = out.view(out.size(0), -1)
            return self.linear(out)

    def get_inference(self, x_adv):
        logit_adv = self(x_adv).detach()
        return logit_adv


def PreActResNet34_IFD(num_classes=10):
    return PreActResNet_IFD(PreActBlock, [3,4,6,3], num_classes=num_classes)
