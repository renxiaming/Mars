from model.base.yolomodel import YoloModel
from model.base.swin_backbone import SwinBackbone
from model.base.neck import Neck
from model.base.head import DetectHead
import torch.nn as nn


class SwinYoloModel(YoloModel):
    """YOLO model with Swin-Transformer backbone."""
    
    def __init__(self, mcfg):
        # 不调用父类的__init__，因为我们要替换backbone
        nn.Module.__init__(self)
        
        self.mcfg = mcfg
        self.inferenceMode = False

        # model layers - 使用Swin-Transformer backbone
        w, r, n = self.getModelWRN(mcfg.phase)
        self.backbone = SwinBackbone(w, r, n)  # 替换为Swin-Transformer
        self.neck = Neck(w, r, n)
        self.head = DetectHead(w, r, self.mcfg.nc, self.mcfg.regMax)

        # model static data (继承自父类)
        self.layerStrides = [8, 16, 32]
        self.outputShapes = (
            (self.mcfg.nc + self.mcfg.regMax * 4, int(self.mcfg.inputShape[0] / 8), int(self.mcfg.inputShape[1] / 8)),
            (self.mcfg.nc + self.mcfg.regMax * 4, int(self.mcfg.inputShape[0] / 16), int(self.mcfg.inputShape[1] / 16)),
            (self.mcfg.nc + self.mcfg.regMax * 4, int(self.mcfg.inputShape[0] / 32), int(self.mcfg.inputShape[1] / 32)),
        )
        
        # 继续父类的初始化
        from misc.bbox import makeAnchors
        import torch
        
        self.anchorPoints, self.anchorStrides = makeAnchors([x[-2:] for x in self.outputShapes], self.layerStrides, 0.5)
        self.anchorPoints = self.anchorPoints.to(self.mcfg.device)
        self.anchorStrides = self.anchorStrides.to(self.mcfg.device)
        self.proj = torch.arange(self.mcfg.regMax, dtype=torch.float).to(self.mcfg.device)
        self.scaleTensor = torch.tensor(self.mcfg.inputShape, device=self.mcfg.device, dtype=torch.float)[[1, 0, 1, 0]]

    @staticmethod
    def getModelWRN(phase):
        """继承父类的参数计算方法"""
        if phase == "nano":
            return (0.25, 2, 1)
        if phase == "small":
            return (0.5, 2, 1)
        if phase == "medium":
            return (0.75, 1.5, 2)
        if phase == "large":
            return (1, 1, 3)
        if phase == "extended":
            return (1.25, 1, 3)
        raise ValueError("Invalid model phase: {}".format(phase))

    def loadBackboneWeights(self, url):
        """Swin-Transformer预训练权重加载"""
        if url is None:
            return
            
        # 对于Swin-Transformer，我们可能需要不同的预训练权重
        from misc.log import log
        log.yellow("Swin-Transformer backbone: 暂时跳过预训练权重加载")
        log.cyan("建议使用ImageNet预训练的Swin-Transformer权重")
        
        # TODO: 实现Swin-Transformer预训练权重加载
        # 可以从timm或者官方仓库下载Swin-Transformer预训练权重


def modelClass():
    """模型工厂需要的入口函数"""
    return SwinYoloModel 