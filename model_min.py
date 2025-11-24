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

        # --- encoder (unchanged) ---
        local_w = os.environ.get("RESNET18_WEIGHTS_PATH")
        if local_w and os.path.exists(local_w):
            enc = resnet18(weights=None) if _USE_TORCHVISION else resnet18()
            sd = torch.load(local_w, map_location="cpu")
            enc.load_state_dict(sd, strict=False)
        elif _USE_TORCHVISION and ResNet18_Weights is not None:
            enc = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        else:
            enc = resnet18()
        enc.fc = nn.Identity()
        self.encoder = enc

        # temporal fusion (same as research)
        self.convofusion = nn.Conv2d(in_channels=sequence_length * C,
                                     out_channels=C, kernel_size=1)

        # --- decoder: EXACT research layout (no TSM), keep defaults (bias=True) ---
        # sizes: 7x7 -> 14 -> 28 -> 56 -> 112 -> 224
        self.conv_decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(256), nn.ReLU(),

            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(128), nn.ReLU(),

            nn.ConvTranspose2d(128,  64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d( 64), nn.ReLU(),

            nn.ConvTranspose2d( 64,  32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d( 32), nn.ReLU(),

            nn.ConvTranspose2d( 32,   3, kernel_size=3, stride=2, padding=1, output_padding=1),
            # no activation (matches research)
        )

    def forward(self, x):  # x: [B,T,3,224,224] -> y: [B,3,224,224]
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)

        # encoder
        x = self.encoder.conv1(x); x = self.encoder.bn1(x); x = self.encoder.relu(x); x = self.encoder.maxpool(x)
        x1 = self.encoder.layer1(x)
        x2 = self.encoder.layer2(x1)
        x3 = self.encoder.layer3(x2)
        x4 = self.encoder.layer4(x3)  # [B*T, 512, 7, 7]

        # temporal fusion like research (T collapsed into channels before decoder)
        _, C4, H4, W4 = x4.shape  # 512, 7, 7
        x4 = x4.reshape(B, T * C4, H4, W4)  # [B, 4*512, 7, 7]
        x4 = self.convofusion(x4)           # [B, 512, 7, 7]

        # research adds TSM here, but you asked to skip it
        # x4_shifted = self.temporal_shift(x4); x4 = x4 + x4_shifted   # <-- omitted

        y = self.conv_decoder(x4)           # [B, 3, 224, 224]
        return y
    
class CNN_TSM_NNAPI(nn.Module):
    def __init__(self, sequence_length=4, spatial_size=224):
        super().__init__()
        self.sequence_length = int(sequence_length)
        self.spatial_size = int(spatial_size)
        if self.spatial_size % 32 != 0:
            raise ValueError("spatial_size must be divisible by 32")

        s32 = self.spatial_size // 32  # 224//32 = 7
        C = 512

        # encoder
        local_w = os.environ.get("RESNET18_WEIGHTS_PATH")
        if local_w and os.path.exists(local_w):
            enc = resnet18(weights=None) if _USE_TORCHVISION else resnet18()
            sd = torch.load(local_w, map_location="cpu")
            enc.load_state_dict(sd, strict=False)
        elif _USE_TORCHVISION and ResNet18_Weights is not None:
            enc = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        else:
            enc = resnet18()
        enc.fc = nn.Identity()
        self.encoder = enc

        self.convofusion = nn.Conv2d(in_channels=self.sequence_length * C,
                                     out_channels=C, kernel_size=1)

        self.conv_decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 3, 2, 1, 1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 3, 2, 1, 1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.ConvTranspose2d(128,  64, 3, 2, 1, 1), nn.BatchNorm2d( 64), nn.ReLU(),
            nn.ConvTranspose2d( 64,  32, 3, 2, 1, 1), nn.BatchNorm2d( 32), nn.ReLU(),
            nn.ConvTranspose2d( 32,   3, 3, 2, 1, 1),
        )

    @torch.no_grad()
    def forward(self, x):  # x: [T, 3, S, S] -> y: [3, S, S]
        T = self.sequence_length
        # treat time as batch for encoder
        x = self.encoder.conv1(x); x = self.encoder.bn1(x); x = self.encoder.relu(x); x = self.encoder.maxpool(x)
        x1 = self.encoder.layer1(x)
        x2 = self.encoder.layer2(x1)
        x3 = self.encoder.layer3(x2)
        x4 = self.encoder.layer4(x3)           # [T, 512, 7, 7] for S=224

        x4 = x4.reshape(1, T * 512, 7, 7)      # [1, 4*512, 7, 7]
        x4 = self.convofusion(x4)              # [1, 512, 7, 7]

        y = self.conv_decoder(x4)              # [1, 3, 224, 224]
        return y[0]


"""
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
                enc = resnet18()                  # fallback def
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

        # fuse temporal channels after encoder: [B, T*512, 7, 7] -> [B, 512, 7, 7]
        self.convofusion = nn.Conv2d(in_channels=sequence_length * C, out_channels=C, kernel_size=1)

        # decoder blocks (unchanged structure)
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

        # ResNet18 encoder
        x = self.encoder.conv1(x); x = self.encoder.bn1(x); x = self.encoder.relu(x); x = self.encoder.maxpool(x)
        x1 = self.encoder.layer1(x)
        x2 = self.encoder.layer2(x1)
        x3 = self.encoder.layer3(x2)
        x4 = self.encoder.layer4(x3)  # [B*T, 512, 7, 7] for 224x224 inputs

        C = x4.shape[1]; H = x4.shape[2]; W = x4.shape[3]
        x4 = x4.reshape(B, T * C, H, W)  # safer than .view for non-contiguous

        x4 = self.convofusion(x4)

        # NNAPI-friendly 2D Resize (bilinear, align_corners=False) between decoder blocks
        def up2(z):
            return F.interpolate(z, scale_factor=2, mode='bilinear', align_corners=False)

        z = up2(x4); z = self.dec1(z)   # 7→14, 512→256
        z = up2(z);  z = self.dec2(z)   # 14→28, 256→128
        z = up2(z);  z = self.dec3(z)   # 28→56, 128→64
        z = up2(z);  z = self.dec4(z)   # 56→112, 64→32
        z = up2(z);  y = self.out(z)    # 112→224, 32→3

        return y


class CNN_TSM_NNAPI(nn.Module):
    
    # Export-only variant for NNAPI:
    # - Input: [T, 3, S, S] (no outer batch)
    # - Avoids DepthToSpace; uses 2D Resize + conv blocks (NNAPI-supported)
    # - Keep BN; export in eval() so folding can occur where applicable
    
    def __init__(self, sequence_length=4, spatial_size=224):
        super().__init__()
        self.sequence_length = int(sequence_length)
        self.spatial_size = int(spatial_size)
        if self.spatial_size % 32 != 0:
            raise ValueError(f"spatial_size must be divisible by 32; got {self.spatial_size}")

        C = 512
        s32 = self.spatial_size // 32  # feature map size from ResNet18 for 224→7
        self._feat_h = s32
        self._feat_w = s32

        # encoder (weights layout matches training)
        local_w = os.environ.get("RESNET18_WEIGHTS_PATH")
        if local_w and os.path.exists(local_w):
            if _USE_TORCHVISION:
                enc = resnet18(weights=None)
            else:
                enc = resnet18()
            sd = torch.load(local_w, map_location="cpu")
            enc.load_state_dict(sd, strict=False)
        elif _USE_TORCHVISION and ResNet18_Weights is not None:
            enc = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        else:
            enc = resnet18()
        enc.fc = nn.Identity()
        self.encoder = enc

        # fuse temporal channels: [1, T*512, s32, s32] -> [1, 512, s32, s32]
        self.convofusion = nn.Conv2d(in_channels=self.sequence_length * C, out_channels=C, kernel_size=1, bias=True)

        # decoder blocks (unchanged structure)
        self.dec1 = nn.Sequential(nn.Conv2d(512, 256, 3, padding=1, bias=True),
                                  nn.BatchNorm2d(256), nn.ReLU())
        self.dec2 = nn.Sequential(nn.Conv2d(256, 128, 3, padding=1, bias=True),
                                  nn.BatchNorm2d(128), nn.ReLU())
        self.dec3 = nn.Sequential(nn.Conv2d(128,  64, 3, padding=1, bias=True),
                                  nn.BatchNorm2d( 64), nn.ReLU())
        self.dec4 = nn.Sequential(nn.Conv2d( 64,  32, 3, padding=1, bias=True),
                                  nn.BatchNorm2d( 32), nn.ReLU())
        self.out  = nn.Conv2d(32, 3, 3, padding=1, bias=True)

    @torch.no_grad()
    def forward(self, x):  # x: [T, 3, S, S] -> y: [3, S, S]
        T  = self.sequence_length
        s7 = self._feat_h  # == S//32

        # treat T as the batch for the encoder
        x = self.encoder.conv1(x); x = self.encoder.bn1(x); x = self.encoder.relu(x); x = self.encoder.maxpool(x)
        x1 = self.encoder.layer1(x)
        x2 = self.encoder.layer2(x1)
        x3 = self.encoder.layer3(x2)
        x4 = self.encoder.layer4(x3)  # [T, 512, s7, s7]

        # fuse time into channels, then conv to 512
        x4 = x4.reshape(1, T * 512, s7, s7)
        x4 = self.convofusion(x4)

        def up2(z):
            return F.interpolate(z, scale_factor=2, mode='bilinear', align_corners=False)

        z = up2(x4);  z = self.dec1(z)       # [1,256, s7*2,  s7*2]
        z = up2(z);   z = self.dec2(z)       # [1,128, s7*4,  s7*4]
        z = up2(z);   z = self.dec3(z)       # [1, 64, s7*8,  s7*8]
        z = up2(z);   z = self.dec4(z)       # [1, 32, s7*16, s7*16]
        z = up2(z);   y = self.out(z)        # [1,  3, S,     S]

        return y[0]  # [3, S, S]
"""