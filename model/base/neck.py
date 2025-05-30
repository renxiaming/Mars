import torch
import torch.nn as nn
import torch.nn.functional as F
from .components import Conv, C2f


class Neck(nn.Module):
    """
    Reference: resources/yolov8.jpg
    """
    def __init__(self, w, r, n):
        super().__init__()
        self.kernelSize = 3
        self.stride = 2

        # PANet Neck layers
        # Top-down pathway (FPN)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        
        # Reduce channels for P5 (feat3 -> C)
        self.conv_p5 = Conv(int(512 * w * r), int(512 * w), 1, 1)
        
        # P4 + P5 -> C2f
        self.c2f_p4p5 = C2f(int(512 * w) + int(512 * w), int(512 * w), n, False)
        
        # Reduce channels for P4P5 fusion
        self.conv_p4p5 = Conv(int(512 * w), int(256 * w), 1, 1)
        
        # P3 + P4P5 -> C2f
        self.c2f_p3p4p5 = C2f(int(256 * w) + int(256 * w), int(256 * w), n, False)
        
        # Bottom-up pathway (PANet)
        # P3P4P5 -> P4P5 (X -> Y)
        self.conv_down1 = Conv(int(256 * w), int(256 * w), self.kernelSize, self.stride)
        self.c2f_down1 = C2f(int(256 * w) + int(512 * w), int(512 * w), n, False)
        
        # P4P5 -> P5 (Y -> Z)
        self.conv_down2 = Conv(int(512 * w), int(512 * w), self.kernelSize, self.stride)
        self.c2f_down2 = C2f(int(512 * w) + int(512 * w * r), int(512 * w * r), n, False)

    def forward(self, feat1, feat2, feat3):
        """
        Input shape:
            feat1: (B, 256 * w, 80, 80)
            feat2: (B, 512 * w, 40, 40)
            feat3: (B, 512 * w * r, 20, 20)
        Output shape:
            C: (B, 512 * w, 40, 40)
            X: (B, 256 * w, 80, 80)
            Y: (B, 512 * w, 40, 40)
            Z: (B, 512 * w * r, 20, 20)
        """
        # Top-down pathway (FPN)
        # P5 reduce channels
        p5_reduced = self.conv_p5(feat3)  # (B, 512*w*r, 20, 20) -> (B, 512*w, 20, 20)
        
        # P5 upsampled + P4
        p5_up = self.upsample(p5_reduced)  # (B, 512*w, 20, 20) -> (B, 512*w, 40, 40)
        p4_p5 = torch.cat([feat2, p5_up], dim=1)  # (B, 512*w + 512*w, 40, 40)
        C = self.c2f_p4p5(p4_p5)  # (B, 512*w, 40, 40)
        
        # P4P5 reduce channels and upsample + P3
        p4p5_reduced = self.conv_p4p5(C)  # (B, 512*w, 40, 40) -> (B, 256*w, 40, 40)
        p4p5_up = self.upsample(p4p5_reduced)  # (B, 256*w, 40, 40) -> (B, 256*w, 80, 80)
        p3_p4p5 = torch.cat([feat1, p4p5_up], dim=1)  # (B, 256*w + 256*w, 80, 80)
        X = self.c2f_p3p4p5(p3_p4p5)  # (B, 256*w, 80, 80)
        
        # Bottom-up pathway (PANet)
        # X downsample + C -> Y
        x_down = self.conv_down1(X)  # (B, 256*w, 80, 80) -> (B, 256*w, 40, 40)
        x_c = torch.cat([x_down, C], dim=1)  # (B, 256*w + 512*w, 40, 40)
        Y = self.c2f_down1(x_c)  # (B, 512*w, 40, 40)
        
        # Y downsample + feat3 -> Z
        y_down = self.conv_down2(Y)  # (B, 512*w, 40, 40) -> (B, 512*w, 20, 20)
        y_feat3 = torch.cat([y_down, feat3], dim=1)  # (B, 512*w + 512*w*r, 20, 20)
        Z = self.c2f_down2(y_feat3)  # (B, 512*w*r, 20, 20)
        
        return C, X, Y, Z
