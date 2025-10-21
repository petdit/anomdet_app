import os, torch
import torch.nn as nn
import torch.nn.functional as F
_USE_TORCHVISION = False
try:
    from torchvision.models import resnet18, ResNet18_Weights
    _USE_TORCHVISION = True
except Exception:
    from resnet18_def import resnet18
    ResNet18_Weights = None

class CNN_TSM(nn.Module):
    def __init__(self, sequence_length=4):
        super().__init__()
        self.sequence_length = sequence_length
        C = 512

        local_w = os.environ.get("RESNET18_WEIGHTS_PATH")  # e.g. /opt/ml/input/data/pretrained/resnet18.pth
        if local_w and os.path.exists(local_w):
            if _USE_TORCHVISION:
                enc = resnet18(weights=None)     # torchvision factory
            else:
                enc = resnet18()                  # your resnet18_def (no kwargs)
            sd = torch.load(local_w, map_location="cpu")
            missing, unexpected = enc.load_state_dict(sd, strict=False)
            print(f"Loaded ResNet18 weights from {local_w}. Missing={len(missing)} Unexpected={len(unexpected)}")
        elif _USE_TORCHVISION and ResNet18_Weights is not None:
            enc = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
            print("Loaded torchvision IMAGENET1K_V1 weights (network access may be needed)")
        else:
            enc = resnet18()
            print("Using fallback, no pretrained weights")

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

    def _encode_frame(self, x4d: torch.Tensor) -> torch.Tensor:
        # x4d: [B,3,H,W] -> [B,512,7,7]
        x = self.encoder.conv1(x4d)
        x = self.encoder.bn1(x); x = self.encoder.relu(x); x = self.encoder.maxpool(x)
        x1 = self.encoder.layer1(x)
        x2 = self.encoder.layer2(x1)
        x3 = self.encoder.layer3(x2)
        x4 = self.encoder.layer4(x3)
        return x4

    def forward(self, x):  # x: [B,T,3,224,224] -> y: [B,3,224,224]
        B, T, C, H, W = x.shape
        if T != self.sequence_length:
            raise RuntimeError(f"Expected T={self.sequence_length}, got {T}")

        if self.training:
            # ====== TRAINING: original vectorized encoder (fast) ======
            x_bt = x.view(B * T, C, H, W)
            x = self.encoder.conv1(x_bt); x = self.encoder.bn1(x); x = self.encoder.relu(x); x = self.encoder.maxpool(x)
            x1 = self.encoder.layer1(x)
            x2 = self.encoder.layer2(x1)
            x3 = self.encoder.layer3(x2)
            x4 = self.encoder.layer4(x3)            # [B*T,512,7,7]
            x4 = x4.reshape(B, T * 512, 7, 7)          # [B, T*512, 7, 7]
        else:
            # ====== EVAL/INFERENCE: serial T-loop so Conv2d sees N=B (==1 on device) ======
            feats = []
            for xt in torch.unbind(x, dim=1):       # xt: [B,3,H,W]
                f = self._encode_frame(xt)          # [B,512,7,7]
                feats.append(f)
            x4 = torch.cat(feats, dim=1)            # [B, T*512, 7, 7]

        # ====== fusion + decoder (unchanged) ======
        x4 = self.convofusion(x4)

        def up2(z):  # explicit resize keeps shapes static & exportable
            return F.interpolate(z, scale_factor=2, mode='nearest')

        z = up2(x4); z = self.dec1(z)   # 7→14, 512→256
        z = up2(z);  z = self.dec2(z)   # 14→28, 256→128
        z = up2(z);  z = self.dec3(z)   # 28→56, 128→64
        z = up2(z);  z = self.dec4(z)   # 56→112, 64→32
        z = up2(z);  y = self.out(z)    # 112→224, 32→3
        return y




class CNN_TSM_NOT_COMPATIBLE_WITH_EXECUTORCH_ON_ANDROID_BECAUSE_B_LARGER_THAN1_ON_CONV2D(nn.Module):
    def __init__(self, sequence_length=4):
        super().__init__()
        self.sequence_length = sequence_length
        C = 512

        local_w = os.environ.get("RESNET18_WEIGHTS_PATH")  # e.g. /opt/ml/input/data/pretrained/resnet18.pth
        if local_w and os.path.exists(local_w):
            if _USE_TORCHVISION:
                enc = resnet18(weights=None)     # torchvision factory
            else:
                enc = resnet18()                  # your resnet18_def (no kwargs)
            sd = torch.load(local_w, map_location="cpu")
            missing, unexpected = enc.load_state_dict(sd, strict=False)
            print(f"Loaded ResNet18 weights from {local_w}. Missing={len(missing)} Unexpected={len(unexpected)}")
        elif _USE_TORCHVISION and ResNet18_Weights is not None:
            enc = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
            print("Loaded torchvision IMAGENET1K_V1 weights (network access may be needed)")
        else:
            enc = resnet18()
            print("Using fallback, no pretrained weights")

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