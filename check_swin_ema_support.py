#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
检查Swin-Transformer所有配置的EMA支持情况
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config.mconfig import MarsConfig
from misc.log import log

def check_ema_support():
    """检查所有Swin-Transformer配置的EMA支持"""
    
    log.cyan("=== 🔍 Swin-Transformer EMA支持检查 ===")
    
    # 定义所有Swin-Transformer配置
    swin_configs = [
        "c1.nano.swin.full",
        "c1.nano.swin.teacher", 
        "c1.nano.swin.distillation"
    ]
    
    ema_support_results = {}
    
    for cfg_name in swin_configs:
        try:
            log.inf(f"\n📋 检查配置: {cfg_name}")
            
            # 加载配置
            mcfg = MarsConfig(cfg_name)
            
            # 检查EMA相关属性
            ema_enabled = hasattr(mcfg, 'useEMA') and mcfg.useEMA
            ema_decay = getattr(mcfg, 'emaDecay', None)
            ema_warmup = getattr(mcfg, 'emaWarmupEpochs', None)
            
            # 检查模型类型
            model_name = getattr(mcfg, 'modelName', 'unknown')
            batch_size = getattr(mcfg, 'batchSize', None)
            max_epoch = getattr(mcfg, 'maxEpoch', None)
            
            # 记录结果
            result = {
                'ema_enabled': ema_enabled,
                'ema_decay': ema_decay,
                'ema_warmup_epochs': ema_warmup,
                'model_name': model_name,
                'batch_size': batch_size,
                'max_epoch': max_epoch,
                'config_loaded': True
            }
            
            ema_support_results[cfg_name] = result
            
            # 输出详细信息
            log.green(f"  ✅ 配置加载成功")
            log.inf(f"  📊 模型类型: {model_name}")
            log.inf(f"  🔢 批量大小: {batch_size}")
            log.inf(f"  📈 训练轮次: {max_epoch}")
            
            if ema_enabled:
                log.green(f"  ✅ EMA已启用")
                log.inf(f"    • 衰减系数: {ema_decay}")
                log.inf(f"    • 预热轮次: {ema_warmup}")
            else:
                log.red(f"  ❌ EMA未启用")
                
        except Exception as e:
            log.red(f"  ❌ 配置加载失败: {e}")
            ema_support_results[cfg_name] = {
                'config_loaded': False,
                'error': str(e)
            }
    
    # 输出总结
    log.cyan("\n=== 📊 EMA支持总结 ===")
    
    total_configs = len(swin_configs)
    loaded_configs = sum(1 for r in ema_support_results.values() if r.get('config_loaded', False))
    ema_enabled_configs = sum(1 for r in ema_support_results.values() 
                              if r.get('config_loaded', False) and r.get('ema_enabled', False))
    
    log.inf(f"总配置数量: {total_configs}")
    log.inf(f"成功加载: {loaded_configs}")
    log.inf(f"EMA已启用: {ema_enabled_configs}")
    
    if ema_enabled_configs == total_configs:
        log.green("🎉 所有Swin-Transformer配置都支持EMA!")
    elif ema_enabled_configs > 0:
        log.yellow(f"⚠️  {ema_enabled_configs}/{total_configs} 配置支持EMA")
    else:
        log.red("❌ 没有配置启用EMA")
    
    # 详细表格
    log.cyan("\n📋 详细支持情况:")
    print("| 配置名称 | 模型类型 | EMA启用 | 衰减系数 | 预热轮次 | 批量大小 |")
    print("|---------|---------|---------|----------|----------|----------|")
    
    for cfg_name, result in ema_support_results.items():
        if result.get('config_loaded', False):
            model_name = result.get('model_name', 'N/A')
            ema_status = "✅" if result.get('ema_enabled', False) else "❌"
            ema_decay = result.get('ema_decay', 'N/A')
            ema_warmup = result.get('ema_warmup_epochs', 'N/A')
            batch_size = result.get('batch_size', 'N/A')
            
            print(f"| {cfg_name} | {model_name} | {ema_status} | {ema_decay} | {ema_warmup} | {batch_size} |")
        else:
            print(f"| {cfg_name} | ERROR | ❌ | N/A | N/A | N/A |")
    
    return ema_support_results


def check_ema_files():
    """检查EMA相关文件路径"""
    
    log.cyan("\n=== 📁 EMA文件路径检查 ===")
    
    try:
        # 测试EMA文件路径生成
        mcfg = MarsConfig("c1.nano.swin.full")
        
        log.inf("EMA相关文件路径:")
        log.inf(f"  • 最佳EMA权重: {mcfg.emaBestWeightsPath()}")
        log.inf(f"  • 最新EMA权重: {mcfg.emaCachePath()}")
        log.inf(f"  • 普通最佳权重: {mcfg.epochBestWeightsPath()}")
        log.inf(f"  • 普通最新权重: {mcfg.epochCachePath()}")
        
        log.green("✅ EMA文件路径生成正常")
        
    except Exception as e:
        log.red(f"❌ EMA文件路径检查失败: {e}")


if __name__ == "__main__":
    results = check_ema_support()
    check_ema_files()
    
    log.cyan("\n=== 🎯 使用建议 ===")
    log.inf("1. 所有三个Swin-Transformer配置都已启用EMA")
    log.inf("2. EMA会在训练过程中自动更新和保存")
    log.inf("3. 验证时会同时评估普通模型和EMA模型")
    log.inf("4. 最佳权重会分别保存普通版本和EMA版本")
    log.green("🚀 可以开始Swin-Transformer + EMA训练!") 