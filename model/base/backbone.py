import torch.nn as nn
from .components import Conv, C2f, SPPF


class Backbone(nn.Module):
    """
    Reference: resources/yolov8.jpg
    """
    def __init__(self, w, r, n):
        super().__init__()
        self.imageChannel = 3
        self.kernelSize = 3
        self.stride = 2

        # YOLOv8 Backbone layers
        # P1/2
        self.conv1 = Conv(self.imageChannel, int(64 * w), self.kernelSize, self.stride)
        
        # P2/4
        self.conv2 = Conv(int(64 * w), int(128 * w), self.kernelSize, self.stride)
        self.c2f1 = C2f(int(128 * w), int(128 * w), n, True)
        
        # P3/8
        self.conv3 = Conv(int(128 * w), int(256 * w), self.kernelSize, self.stride)
        self.c2f2 = C2f(int(256 * w), int(256 * w), 2 * n, True)
        
        # P4/16
        self.conv4 = Conv(int(256 * w), int(512 * w), self.kernelSize, self.stride)
        self.c2f3 = C2f(int(512 * w), int(512 * w), 2 * n, True)
        
        # P5/32
        self.conv5 = Conv(int(512 * w), int(512 * w * r), self.kernelSize, self.stride)
        self.c2f4 = C2f(int(512 * w * r), int(512 * w * r), n, True)
        self.sppf = SPPF(int(512 * w * r), int(512 * w * r), 5)

    def forward(self, x):
        """
        Input shape: (B, 3, 640, 640)
        Output shape:
            feat0: (B, 128 * w, 160, 160)
            feat1: (B, 256 * w, 80, 80)
            feat2: (B, 512 * w, 40, 40)
            feat3: (B, 512 * w * r, 20, 20)
        """
        # P1/2: (B, 3, 640, 640) -> (B, 64*w, 320, 320)
        x = self.conv1(x)
        
        # P2/4: (B, 64*w, 320, 320) -> (B, 128*w, 160, 160)
        x = self.conv2(x)
        feat0 = self.c2f1(x)  # feat0: (B, 128*w, 160, 160)
        
        # P3/8: (B, 128*w, 160, 160) -> (B, 256*w, 80, 80)
        x = self.conv3(feat0)
        feat1 = self.c2f2(x)  # feat1: (B, 256*w, 80, 80)
        
        # P4/16: (B, 256*w, 80, 80) -> (B, 512*w, 40, 40)
        x = self.conv4(feat1)
        feat2 = self.c2f3(x)  # feat2: (B, 512*w, 40, 40)
        
        # P5/32: (B, 512*w, 40, 40) -> (B, 512*w*r, 20, 20)
        x = self.conv5(feat2)
        x = self.c2f4(x)
        feat3 = self.sppf(x)  # feat3: (B, 512*w*r, 20, 20)
        
        return feat0, feat1, feat2, feat3
