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
            # 原始蒸馏模式 - 使用EMA权重
            mcfg.modelName = "distillation"
            # 优先使用EMA权重，如果不存在则使用常规权重
            ema_weights_path = f"C:/Mars_Output/{username}/c1.nano.teacher/__cache__/ema_best_weights.pth"
            regular_weights_path = f"C:/Mars_Output/{username}/c1.nano.teacher/__cache__/best_weights.pth"
            
            # 检查EMA权重是否存在
            if os.path.exists(ema_weights_path):
                mcfg.checkpointModelFile = ema_weights_path
                mcfg.teacherModelFile = ema_weights_path
                print(f"🎯 使用EMA teacher权重: {ema_weights_path}")
            else:
                mcfg.checkpointModelFile = regular_weights_path
                mcfg.teacherModelFile = regular_weights_path
                print(f"⚠️  EMA权重不存在，使用常规权重: {regular_weights_path}")
            
        mcfg.distilLossWeights = (1.0, 0.05, 0.001)
        mcfg.maxEpoch = 100
        mcfg.backboneFreezeEpochs = [x for x in range(0, 25)]
        mcfg.epochValidation = False # DO NOT MODIFY
        mcfg.trainSplitName = "small" # DO NOT MODIFY
        mcfg.teacherClassIndexes = [x for x in range(0, 10)] # DO NOT MODIFY

    # EMA配置支持
    if "ema" in tags:
        mcfg.useEMA = True
        mcfg.emaDecay = 0.9999  # 更高的衰减系数，更稳定
        mcfg.emaWarmupEpochs = 3  # EMA预热期
        # 确保保存EMA权重
        mcfg.epochValidation = True  # 需要验证才能保存最佳权重
        
        # 如果是distillation.ema，强制使用EMA teacher权重
        if "distillation" in tags and not "swin" in tags:
            username = os.environ.get('USERNAME') or os.environ.get('USER', 'root')
            # 使用专门的EMA teacher目录
            ema_teacher_weights = f"C:/Mars_Output/{username}/c1.nano.teacher.ema/__cache__/best_weights.pth"
            fallback_ema_weights = f"C:/Mars_Output/{username}/c1.nano.teacher/__cache__/ema_best_weights.pth"
            
            if os.path.exists(ema_teacher_weights):
                mcfg.checkpointModelFile = ema_teacher_weights
                mcfg.teacherModelFile = ema_teacher_weights
                print(f"🎯 EMA蒸馏模式：使用EMA teacher目录权重: {ema_teacher_weights}")
            elif os.path.exists(fallback_ema_weights):
                mcfg.checkpointModelFile = fallback_ema_weights
                mcfg.teacherModelFile = fallback_ema_weights
                print(f"🎯 EMA蒸馏模式：使用EMA权重文件: {fallback_ema_weights}")
            else:
                print(f"❌ 错误：EMA蒸馏需要EMA teacher权重，但以下文件都不存在:")
                print(f"   - {ema_teacher_weights}")
                print(f"   - {fallback_ema_weights}")

    return mcfg
