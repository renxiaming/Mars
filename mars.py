import sys
from engine.engine import MarsEngine


if __name__ == "__main__":
    mode = "pipe"
    nobuf = False

    for i in range(len(sys.argv)):
        arg = sys.argv[i]
        if arg == "-nobuf":
            nobuf = True
        elif arg == "-train":
            mode = "train"
        elif arg == "-eval":
            mode = "eval"
        elif arg == "-pipe":
            mode = "pipe"

    # 🎯 配置选择 - 选择你要运行的配置
    
    # === 原始YOLO配置 ===
    # cfgname = "c1.nano.full"             # 原始YOLO完整训练
    # cfgname = "c1.nano.teacher"          # 原始YOLO教师模式
    # cfgname = "c1.nano.full.ema"         # 原始YOLO + EMA
    # cfgname = "c1.nano.distillation"     # 原始YOLO蒸馏
    # cfgname = "c1.nano.distillation.ema" # 原始YOLO蒸馏 + EMA增强
    
    # === 🆕 Swin-Transformer配置 ===
    cfgname = "c1.nano.swin.full"          # Swin-Transformer完整训练
    # cfgname = "c1.nano.swin.teacher"     # Swin-Transformer教师模式
    # cfgname = "c1.nano.swin.distillation" # Swin-Transformer蒸馏

    MarsEngine(
        mode=mode,
        cfgname=cfgname,
        root="C:/Mars_Output", # 修改为你的Windows系统路径，用于保存训练结果
        nobuf=nobuf,
    ).run()
