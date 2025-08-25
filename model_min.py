import torch
import torch.nn as nn
import torch.nn.functional as F
_USE_TORCHVISION = False
try:
    from torchvision.models import resnet18, ResNet18_Weights  # training container has this
    _USE_TORCHVISION = True
except Exception:
    from resnet18_def import resnet18  # exporter uses this (no torchvision)
    ResNet18_Weights = None  # just so name exists



class CNN_TSM(nn.Module):
    def __init__(self, sequence_length=4):
        super().__init__()
        self.sequence_length = sequence_length
        C = 512

        if _USE_TORCHVISION and ResNet18_Weights is not None:
            enc = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
            print("using torchvision ")
        else:
            enc = resnet18()  # local copy, random init (we load checkpoint next)
            print("using fallback, no loaded weights")




        enc.fc = nn.Identity()
        self.encoder = enc

        self.convofusion = nn.Conv2d(in_channels=sequence_length * C, out_channels=C, kernel_size=1)

        self.dec1 = nn.Sequential(nn.Conv2d(512, 256, 3, padding=1),
                                  nn.BatchNorm2d(256), nn.ReLU())
        self.dec2 = nn.Sequential(nn.Conv2d(256, 128, 3, padding=1),
                                  nn.BatchNorm2d(128), nn.ReLU())
        self.dec3 = nn.Sequential(nn.Conv2d(128, 64, 3, padding=1),
                                  nn.BatchNorm2d(64), nn.ReLU())
        self.dec4 = nn.Sequential(nn.Conv2d(64, 32, 3, padding=1),
                                  nn.BatchNorm2d(32), nn.ReLU())
        self.out  = nn.Conv2d(32, 3, 3, padding=1)

    def forward(self, x):  # x: [B,T,3,224,224] -> y: [B,3,224,224]
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)

        x = self.encoder.conv1(x); x = self.encoder.bn1(x); x = self.encoder.relu(x); x = self.encoder.maxpool(x)
        x1 = self.encoder.layer1(x)
        x2 = self.encoder.layer2(x1)
        x3 = self.encoder.layer3(x2)
        x4 = self.encoder.layer4(x3)  # [B*T, 512, 7, 7]

        C = x4.shape[1]; H = x4.shape[2]; W = x4.shape[3]
        x4 = x4.reshape(B, T * C, H, W)

        x4 = self.convofusion(x4)
        def up2(z):  # explicit resize keeps shapes static & exportable
            return F.interpolate(z, scale_factor=2, mode='nearest')  # or 'bilinear', align_corners=False

        z = up2(x4); z = self.dec1(z)   # 7→14, 512→256
        z = up2(z);  z = self.dec2(z)   # 14→28, 256→128
        z = up2(z);  z = self.dec3(z)   # 28→56, 128→64
        z = up2(z);  z = self.dec4(z)   # 56→112, 64→32
        z = up2(z);  y = self.out(z)    # 112→224, 32→3

        return y
