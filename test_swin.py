#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
测试Swin-Transformer YOLO模型
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
from config.mconfig import MarsConfig
from factory.modelfactory import MarsModelFactory
from misc.log import log

def test_swin_model():
    """测试Swin-Transformer模型构建和前向传播"""
    
    log.cyan("=== 🎯 Swin-Transformer YOLO 测试 ===")
    
    # 1. 加载配置
    cfgname = "c1_nano_swin"
    mcfg = MarsConfig(cfgname)
    log.inf(f"配置加载成功: {cfgname}")
    log.inf(f"设备: {mcfg.device}")
    log.inf(f"输入尺寸: {mcfg.inputShape}")
    
    # 2. 创建模型
    try:
        model = MarsModelFactory.loadNewModel(mcfg, None)
        log.green("✅ Swin-Transformer模型创建成功")
        
        # 3. 模型信息
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        log.inf(f"总参数量: {total_params:,}")
        log.inf(f"可训练参数: {trainable_params:,}")
        
        # 4. 模型结构检查
        log.cyan("\n📊 模型结构分析:")
        log.inf(f"Backbone类型: {type(model.backbone).__name__}")
        log.inf(f"Neck类型: {type(model.neck).__name__}")  
        log.inf(f"Head类型: {type(model.head).__name__}")
        
        # 5. 前向传播测试
        model.eval()
        batch_size = 2
        test_input = torch.randn(batch_size, 3, 640, 640).to(mcfg.device)
        
        log.cyan(f"\n🧪 前向传播测试 (batch_size={batch_size}):")
        
        with torch.no_grad():
            # Backbone测试
            log.inf("测试backbone...")
            feat0, feat1, feat2, feat3 = model.backbone(test_input)
            
            log.green(f"feat0 (P2): {feat0.shape}")  # 期望 [2, 32, 160, 160]
            log.green(f"feat1 (P3): {feat1.shape}")  # 期望 [2, 64, 80, 80]
            log.green(f"feat2 (P4): {feat2.shape}")  # 期望 [2, 128, 40, 40]
            log.green(f"feat3 (P5): {feat3.shape}")  # 期望 [2, 256, 20, 20]
            
            # Neck测试
            log.inf("测试neck...")
            C, X, Y, Z = model.neck(feat1, feat2, feat3)
            
            log.green(f"C (40x40): {C.shape}")
            log.green(f"X (80x80): {X.shape}")
            log.green(f"Y (40x40): {Y.shape}")
            log.green(f"Z (20x20): {Z.shape}")
            
            # Head测试
            log.inf("测试head...")
            outputs = model.head([X, Y, Z])
            
            for i, out in enumerate(outputs):
                log.green(f"输出{i+1}: {out.shape}")
            
            # 完整模型测试
            log.inf("测试完整模型...")
            if hasattr(model, 'forward'):
                predictions = model(test_input)
                log.green(f"最终输出数量: {len(predictions)}")
                for i, pred in enumerate(predictions):
                    log.green(f"预测{i+1}: {pred.shape}")
        
        log.green("\n✅ 所有测试通过!")
        
        # 6. 内存使用情况
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated() / 1024**2  # MB
            log.inf(f"GPU内存使用: {memory_allocated:.1f} MB")
        
        return model
        
    except Exception as e:
        log.red(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def compare_with_original():
    """对比Swin-Transformer与原始backbone的性能"""
    
    log.cyan("\n=== 📊 性能对比测试 ===")
    
    # 加载原始模型
    try:
        orig_cfg = MarsConfig("c1.nano.full.ema")
        orig_model = MarsModelFactory.loadNewModel(orig_cfg, None)
        orig_params = sum(p.numel() for p in orig_model.parameters())
        log.inf(f"原始模型参数: {orig_params:,}")
        
        # 加载Swin模型  
        swin_cfg = MarsConfig("c1_nano_swin")
        swin_model = MarsModelFactory.loadNewModel(swin_cfg, None)
        swin_params = sum(p.numel() for p in swin_model.parameters())
        log.inf(f"Swin模型参数: {swin_params:,}")
        
        # 参数对比
        param_ratio = swin_params / orig_params
        log.yellow(f"参数比例: {param_ratio:.2f}x")
        
        # 推理速度测试
        test_input = torch.randn(1, 3, 640, 640).cuda()
        
        # 预热
        for _ in range(10):
            with torch.no_grad():
                _ = orig_model(test_input)
                _ = swin_model(test_input)
        
        # 速度测试
        import time
        
        # 原始模型
        torch.cuda.synchronize()
        start_time = time.time()
        for _ in range(100):
            with torch.no_grad():
                _ = orig_model(test_input)
        torch.cuda.synchronize()
        orig_time = time.time() - start_time
        
        # Swin模型
        torch.cuda.synchronize()
        start_time = time.time()
        for _ in range(100):
            with torch.no_grad():
                _ = swin_model(test_input)
        torch.cuda.synchronize()
        swin_time = time.time() - start_time
        
        log.inf(f"原始模型推理时间: {orig_time*10:.2f}ms per image")
        log.inf(f"Swin模型推理时间: {swin_time*10:.2f}ms per image")
        log.yellow(f"速度比例: {swin_time/orig_time:.2f}x")
        
    except Exception as e:
        log.yellow(f"性能对比跳过: {e}")


if __name__ == "__main__":
    model = test_swin_model()
    if model is not None:
        compare_with_original()
        log.cyan("\n🎉 Swin-Transformer集成测试完成!")
    else:
        log.red("❌ 测试失败") 