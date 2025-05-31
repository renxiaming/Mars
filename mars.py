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

    MarsEngine(
        mode=mode,
        cfgname="c1.nano.full.ema",  # 从full配置开始，添加EMA功能进行对比
        # cfgname="c1.nano.full.cuda@3",
        # cfgname="c1.nano.teacher.ema",
        # cfgname="c1.nano.distillation",
        root="C:/Mars_Output", # 修改为你的Windows系统路径，用于保存训练结果
        nobuf=nobuf,
    ).run()
