import os
from config import mconfig


def mcfg(tags):
    mcfg = mconfig.ModelConfig()
    # projectRootDir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # pretrainedFile = os.path.join(projectRootDir, "resources/pretrained/backbone", "backbone_{}.pth".format(mcfg.phase))
    # mcfg.pretrainedBackboneUrl = "file://{}".format(pretrainedFile)

    mcfg.phase = "nano" # DO NOT MODIFY
    mcfg.trainSplitName = "train" # DO NOT MODIFY
    mcfg.validationSplitName = "validation" # DO NOT MODIFY
    mcfg.testSplitName = "test" # DO NOT MODIFY

    # data setup - 使用项目中archive文件夹的数据集
    mcfg.imageDir = "archive/mar20/images"
    mcfg.annotationDir = "archive/mar20/annotations"
    mcfg.classList = ["A{}".format(x) for x in range(1, 21)] # DO NOT MODIFY
    mcfg.subsetMap = { # DO NOT MODIFY
        "train": "archive/mar20/splits/v5/train.txt",
        "validation": "archive/mar20/splits/v5/validation.txt",
        "test": "archive/mar20/splits/v5/test.txt",
        "small": "archive/mar20/splits/v5/small.txt",
    }

    # 根据backbone类型设置预训练权重
    if "swin" in tags:
        # Swin-Transformer不使用YOLO预训练权重
        mcfg.pretrainedBackboneUrl = None
        mcfg.modelName = "swin"  # 使用Swin模型
        mcfg.batchSize = 6  # Swin-Transformer内存需求更高
    else:
        # 启用YOLO预训练权重
        mcfg.pretrainedBackboneUrl = "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt"
        mcfg.modelName = "base"

    if "full" in tags:
        mcfg.maxEpoch = 200
        mcfg.backboneFreezeEpochs = [x for x in range(0, 100)]

    if "teacher" in tags:
        mcfg.maxEpoch = 200
        mcfg.backboneFreezeEpochs = [x for x in range(0, 100)]
        mcfg.trainSelectedClasses = ["A{}".format(x) for x in range(1, 11)] # DO NOT MODIFY

    if "distillation" in tags:
        # 动态获取用户名，兼容Windows和Linux
        username = os.environ.get('USERNAME') or os.environ.get('USER', 'root')
        
        if "swin" in tags:
            # Swin-Transformer蒸馏模式
            mcfg.modelName = "swin"  # 学生模型也用Swin
            mcfg.checkpointModelFile = f"C:/Mars_Output/{username}/c1.nano.swin.teacher/__cache__/best_weights.pth"
            mcfg.teacherModelFile = f"C:/Mars_Output/{username}/c1.nano.swin.teacher/__cache__/best_weights.pth"
        else:
            # 原始蒸馏模式
            mcfg.modelName = "distillation"
            mcfg.checkpointModelFile = f"C:/Mars_Output/{username}/c1.nano.teacher/__cache__/best_weights.pth"
            mcfg.teacherModelFile = f"C:/Mars_Output/{username}/c1.nano.teacher/__cache__/best_weights.pth"
            
        mcfg.distilLossWeights = (1.0, 0.05, 0.001)
        mcfg.maxEpoch = 100
        mcfg.backboneFreezeEpochs = [x for x in range(0, 25)]
        mcfg.epochValidation = False # DO NOT MODIFY
        mcfg.trainSplitName = "small" # DO NOT MODIFY
        mcfg.teacherClassIndexes = [x for x in range(0, 10)] # DO NOT MODIFY

    return mcfg
