"""
Code adapted from https://github.com/facebookresearch/GradientEpisodicMemory
                    &
                  https://github.com/kuangliu/pytorch-cifar
"""
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.functional import relu, avg_pool2d, max_pool2d
import torch

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class SELayer(nn.Module):                #se
    def __init__(self, channel, reduction=5):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

# class ChannelAttention(nn.Module):
#     def __init__(self, in_planes, ratio=16):
#         super(ChannelAttention, self).__init__()
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.max_pool = nn.AdaptiveMaxPool2d(1)
 
#         self.fc1   = nn.Conv2d(in_planes, in_planes // 5, 1, bias=False)
#         self.relu1 = nn.ReLU()
#         self.fc2   = nn.Conv2d(in_planes // 5, in_planes, 1, bias=False)
 
#         self.sigmoid = nn.Sigmoid()
 
#     def forward(self, x):
#         avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
#         max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
#         out = avg_out + max_out
#         return self.sigmoid(out)

# class SpatialAttention(nn.Module):
#     def __init__(self, kernel_size=7):
#         super(SpatialAttention, self).__init__()
 
#         assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
#         padding = 3 if kernel_size == 7 else 1
 
#         self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
#         self.sigmoid = nn.Sigmoid()
 
#     def forward(self, x):
#         avg_out = torch.mean(x, dim=1, keepdim=True)
#         max_out, _ = torch.max(x, dim=1, keepdim=True)
#         x = torch.cat([avg_out, max_out], dim=1)
#         x = self.conv1(x)
#         return self.sigmoid(x)
    
class ChannelAttention(nn.Module):
    """ Channel Attention Module """
    def __init__(self, in_channels, reduction_ratio=5):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))
        self.max_pool = nn.AdaptiveMaxPool2d((1,1))
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, in_channels//reduction_ratio),
            nn.ReLU(),
            nn.Linear(in_channels//reduction_ratio, in_channels)
        )
    
    def forward(self, x):
        avg_feat = self.mlp(self.avg_pool(x).flatten(1))
        max_feat = self.mlp(self.max_pool(x).flatten(1))
        att_feat = avg_feat + max_feat
        att_weight = torch.sigmoid(att_feat).unsqueeze(2).unsqueeze(3)
        return x*att_weight


class SpatialAttention(nn.Module):
    """ Spatial Attention Module """
    def __init__(self):
        super().__init__()
        self.Conv = nn.Sequential(
            nn.Conv2d(2, 1, 7, stride=1, padding=3, bias=False),
            nn.BatchNorm2d(1)
        )
    
    def forward(self, x):
        max_feat = torch.max(x, dim=1)[0].unsqueeze(1)
        mean_feat = torch.mean(x, dim=1).unsqueeze(1)
        att_feat = torch.cat((max_feat, mean_feat), dim=1)
        att_weight = torch.sigmoid(self.Conv(att_feat))
        return x*att_weight

class CBAM(nn.Module):
    """ Channel Block Attention Module """
    def __init__(self, in_channels, reduction_ratio=5):
        super().__init__()
        self.CA = ChannelAttention(in_channels, reduction_ratio)
        self.SA = SpatialAttention()

    def forward(self, x):
        feat = self.CA(x)
        feat = self.SA(feat)
        return feat

    
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
#         self.se = SELayer(planes)    #se
#         self.cbam = CBAM(planes)         #cbam

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
#         out = self.se(out) #se
#         out = self.cbam(out) #cbam
        out += self.shortcut(x)
        out = relu(out)
        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = relu(self.bn1(self.conv1(x)))
        out = relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes, nf, bias):
        super(ResNet, self).__init__()
        self.in_planes = nf
        self.conv1 = conv3x3(3, nf * 1)
        self.bn1 = nn.BatchNorm2d(nf * 1)
#         self.ca = ChannelAttention(self.in_planes)    #cbam
#         self.sa = SpatialAttention()                 #cbam
#         self.c = CBAM(self.in_planes*8)             #cbam
#         self.se = SELayer(self.in_planes*8, 16) #se
        self.layer1 = self._make_layer(block, nf * 1, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, nf * 2, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, nf * 4, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, nf * 8, num_blocks[3], stride=2)
        self.linear = nn.Linear(nf * 8 * block.expansion, num_classes, bias=bias)


    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def features(self, x):
        '''Features before FC layers'''
#         print(x.shape)         #torch.Size([30, 3, 200, 200])
        out = relu(self.bn1(self.conv1(x)))
#         print(out.shape)        #torch.Size([30, 20, 200, 200])
        out = self.layer1(out)
#         print(out.shape)         #torch.Size([30, 20, 200, 200])
        out = self.layer2(out)
#         print(out.shape)         #torch.Size([30, 40, 100, 100])
        out = self.layer3(out)
#         print(out.shape)         #torch.Size([30, 80, 50, 50])
        out = self.layer4(out)
#         print(out.shape)         #torch.Size([30, 160, 25, 25])
#         out = self.c(out)
#         out = self.se(out)
        out = avg_pool2d(out, 4) 
#         print(out.shape)         #torch.Size([30, 160, 6, 6])
        out = out.view(out.size(0), -1)
#         print(out.shape)         #torch.Size([30, 5760])
        return out

    def logits(self, x):
        '''Apply the last FC linear mapping to get logits'''
        x = self.linear(x)
        return x

    def forward(self, x):
        out = self.features(x)
        logits = self.logits(out)
        return logits


def Reduced_ResNet18(nclasses, nf=20, bias=True):
    """
    Reduced ResNet18 as in GEM MIR(note that nf=20).
    """
    return ResNet(BasicBlock, [2, 2, 2, 2], nclasses, nf, bias)


def ResNet18(nclasses, nf=64, bias=True):
    return ResNet(BasicBlock, [2, 2, 2, 2], nclasses, nf, bias)

'''
See https://github.com/kuangliu/pytorch-cifar/blob/master/models/resnet.py
'''

def ResNet34(nclasses, nf=64, bias=True):
    return ResNet(BasicBlock, [3, 4, 6, 3], nclasses, nf, bias)

def ResNet50(nclasses, nf=64, bias=True):
    return ResNet(Bottleneck, [3, 4, 6, 3], nclasses, nf, bias)


def ResNet101(nclasses, nf=64, bias=True):
    return ResNet(Bottleneck, [3, 4, 23, 3], nclasses, nf, bias)


def ResNet152(nclasses, nf=64, bias=True):
    return ResNet(Bottleneck, [3, 8, 36, 3], nclasses, nf, bias)


# class SupConResNet(nn.Module):
#     """backbone + projection head"""
#     def __init__(self, dim_in=5760, head='mlp', feat_dim=128): #5760  #128
#         super(SupConResNet, self).__init__()
#         #self.encoder = Reduced_ResNet18(100)
#         self.encoder = Reduced_ResNet18(2)
#         #self.encoder = ResNet18(2)
#         if head == 'linear':
#             self.head = nn.Linear(dim_in, feat_dim)
#         elif head == 'mlp':
#             self.head = nn.Sequential(
#                 nn.Linear(dim_in, dim_in),
#                 nn.ReLU(inplace=True),
#                 nn.Linear(dim_in, feat_dim)
#             )
#         elif head == 'None':
#             self.head = None
#         else:
#             raise NotImplementedError(
#                 'head not supported: {}'.format(head))

#     def forward(self, x):
#         feat = self.encoder.features(x)
# #         print(feat.shape)
#         if self.head:
#             feat = F.normalize(self.head(feat), dim=1)
#         else:
#             feat = F.normalize(feat, dim=1)
            
#         return feat

#     def features(self, x):
#         return self.encoder.features(x)

class LinearClassifier(nn.Module):
    """Linear classifier"""
    def __init__(self, feat_dim=5760, num_classes=2):
        super(LinearClassifier, self).__init__()
#         self.fc = nn.Linear(feat_dim, num_classes)
        self.fc = nn.Sequential(
                nn.Linear(feat_dim, 1024),
                nn.ReLU(inplace=True),
                nn.Linear(1024, 128),
                nn.ReLU(inplace=True),
                nn.Linear(128, num_classes)
            )

    def forward(self, features):
        return self.fc(features)
    
class ConvClassifier(nn.Module):
    """conv classifier"""
    def __init__(self, num_classes=2):
        super(ConvClassifier, self).__init__()
#         self.conv1 = conv3x3(160, 80, stride=2)              #3x3 3layer
#         self.bn1 = nn.BatchNorm2d(80)
#         self.conv2 = conv3x3(80, 40, stride=2)
#         self.bn2 = nn.BatchNorm2d(40)
#         self.conv3 = conv3x3(40, num_classes, stride=2)
#         self.bn3 = nn.BatchNorm2d(num_classes)

#         self.conv1 = nn.Conv2d(160, 80, kernel_size=1, stride=1,    #1x1
#                      padding=0, bias=False)
#         self.bn1 = nn.BatchNorm2d(80)
#         self.conv2 = nn.Conv2d(80, 40, kernel_size=1, stride=1,
#                      padding=0, bias=False)
#         self.bn2 = nn.BatchNorm2d(40)
#         self.conv3 = nn.Conv2d(40, num_classes, kernel_size=1, stride=1,
#                      padding=0, bias=False)
#         self.bn3 = nn.BatchNorm2d(num_classes)

#         self.conv1 = conv3x3(160, 80)                           ##3x3 3layer+fc or +ap
#         self.bn1 = nn.BatchNorm2d(80)
#         self.conv2 = conv3x3(80, 40)
#         self.bn2 = nn.BatchNorm2d(40)
#         self.conv3 = conv3x3(40, num_classes)
#         self.bn3 = nn.BatchNorm2d(num_classes)
#         self.fc = nn.Linear(72, 2)

#         self.conv1 = conv3x3(160, 120)          #3x3 5layer+fc
#         self.bn1 = nn.BatchNorm2d(120)
#         self.conv2 = conv3x3(120, 80)
#         self.bn2 = nn.BatchNorm2d(80)
#         self.conv3 = conv3x3(80, 40)
#         self.bn3 = nn.BatchNorm2d(40)
#         self.conv4 = conv3x3(40, 20)
#         self.bn4 = nn.BatchNorm2d(20)
#         self.conv5 = conv3x3(20, num_classes)
#         self.bn5 = nn.BatchNorm2d(num_classes)
#         self.fc = nn.Linear(72, 2)

        self.conv1 = conv3x3(160, 120)          #3x3 5layer+ap+fc
        self.bn1 = nn.BatchNorm2d(120)
        self.conv2 = conv3x3(120, 80)
        self.bn2 = nn.BatchNorm2d(80)
        self.conv3 = conv3x3(80, 40)
        self.bn3 = nn.BatchNorm2d(40)
        self.conv4 = conv3x3(40, 20)
        self.bn4 = nn.BatchNorm2d(20)
        self.conv5 = conv3x3(20, num_classes)
        self.bn5 = nn.BatchNorm2d(num_classes)
        self.fc = nn.Linear(18, 2)

#         self.conv1 = conv3x3(160, 80)                           ##3x3 3layer+ap+fc
#         self.bn1 = nn.BatchNorm2d(80)
#         self.conv2 = conv3x3(80, 40)
#         self.bn2 = nn.BatchNorm2d(40)
#         self.conv3 = conv3x3(40, num_classes)
#         self.bn3 = nn.BatchNorm2d(num_classes)
#         self.fc = nn.Linear(18, 2)

    def forward(self, x):
        x = x.view(x.size(0), 160, 6, 6)
#         print(x.shape)
        x = relu(self.bn1(self.conv1(x)))
#         print(x.shape)
        x = relu(self.bn2(self.conv2(x)))
#         print(x.shape)
        x = relu(self.bn3(self.conv3(x)))
        x = relu(self.bn4(self.conv4(x)))
        x = relu(self.bn5(self.conv5(x)))
#         print(x.shape)
        x = avg_pool2d(x, 2) 
#         print(x.shape)
#         x = x.view(x.size(0), -1)
#         print(x.shape)
        x = x.view(x.size(0), -1)
        x = self.fc(x) 
        return x
    
# class SupConResNet(nn.Module):        #把映射層改成全連接分類器
#     """backbone + projection head"""
#     def __init__(self, dim_in=5760, head='mlp', feat_dim=2): #5760  #128
#         super(SupConResNet, self).__init__()
#         self.encoder = Reduced_ResNet18(2)
#         if head == 'linear':
#             self.head = nn.Linear(dim_in, feat_dim)
#         elif head == 'mlp':
#             self.head = nn.Sequential(
#                 nn.Linear(dim_in, 1024),
#                 nn.ReLU(inplace=True),
#                 nn.Linear(1024, 128),
#                 nn.ReLU(inplace=True),
#                 nn.Linear(128, feat_dim)
#             )
#         elif head == 'None':
#             self.head = None
#         else:
#             raise NotImplementedError(
#                 'head not supported: {}'.format(head))

#     def forward(self, x):
#         feat = self.encoder.features(x)
# #         print(feat.shape)
#         if self.head:
#             feat = F.normalize(self.head(feat), dim=1)
#         else:
#             feat = F.normalize(feat, dim=1)
            
#         return feat

#     def features(self, x):
#         return self.encoder.features(x)

class SupConResNet(nn.Module):        #把映射層改成卷積分類器   lr->0.001
    """backbone + projection head"""
    def __init__(self, dim_in=5760, head='mlp', feat_dim=2): #5760  #128
        super(SupConResNet, self).__init__()
        self.encoder = Reduced_ResNet18(2)
        self.fc = nn.Linear(18, 2)
        if head == 'linear':
            self.head = nn.Linear(dim_in, feat_dim)
        elif head == 'mlp':
            self.head = nn.Sequential(
                conv3x3(160, 120),         
                nn.BatchNorm2d(120),
                conv3x3(120, 80),
                nn.BatchNorm2d(80),
                conv3x3(80, 40),
                nn.BatchNorm2d(40),
                conv3x3(40, 20),
                nn.BatchNorm2d(20),
                conv3x3(20, 2),
                nn.BatchNorm2d(2)
            )
        elif head == 'None':
            self.head = None
        else:
            raise NotImplementedError(
                'head not supported: {}'.format(head))

    def forward(self, x):
        feat = self.encoder.features(x)
        feat = feat.view(feat.size(0), 160, 6, 6)

#         feat = F.normalize(self.head(feat), dim=1)
        feat = self.head(feat)
        feat = avg_pool2d(feat, 2) 
        feat = feat.view(feat.size(0), -1)
        feat = self.fc(feat) 
  
        return feat

    def features(self, x):
        return self.encoder.features(x)