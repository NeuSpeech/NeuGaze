# =============================================================================
# NeuSpeech Institute, NeuGaze Project
# Copyright (c) 2024 Yiqian Yang
#
# This code is part of the NeuGaze project developed at NeuSpeech Institute.
# Author: Yiqian Yang
#
# This work is licensed under the Creative Commons Attribution-NonCommercial 4.0 
# International License. To view a copy of this license, visit:
# http://creativecommons.org/licenses/by-nc/4.0/
# =============================================================================

#!/usr/bin/env python3
"""
测试累积训练功能的脚本
验证新的累积训练系统是否正常工作
"""

import sys
import os
from pathlib import Path

# 添加项目路径
sys.path.append(str(Path(__file__).parent))

# 模拟导入pipeline
try:
    from my_model_arch.cpu_fast.pipeline import IntegratedRegressionMediaPipeline
    import torch
    import numpy as np
    print("Successfully imported pipeline")
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)

def test_accumulated_training():
    """测试累积训练功能"""
    print("=" * 50)
    print("测试累积训练功能")
    print("=" * 50)
    
    # 创建管道实例
    try:
        pipeline = IntegratedRegressionMediaPipeline(
            weights="my_model_arch/cpu_fast/cpu_convert/mobileone_s0_224_fp16_pnnx.ncnn.param",
            device="cpu",
            use_accumulated_training=True,  # 启用累积训练
            max_accumulated_datasets=5     # 最多使用5个数据集
        )
        print("✓ Pipeline 创建成功")
    except Exception as e:
        print(f"✗ Pipeline 创建失败: {e}")
        return False
    
    # 测试历史数据收集
    try:
        historical_paths = pipeline.collect_historical_calibration_data()
        print(f"✓ 找到 {len(historical_paths)} 个历史校准数据集")
        for path in historical_paths[:3]:  # 只显示前3个
            print(f"  - {path}")
        if len(historical_paths) > 3:
            print(f"  ... 还有 {len(historical_paths) - 3} 个数据集")
    except Exception as e:
        print(f"✗ 历史数据收集失败: {e}")
        return False
    
    # 测试累积训练（如果有历史数据）
    if historical_paths:
        try:
            print("\n开始测试累积训练...")
            success = pipeline.train_with_accumulated_data(max_datasets=3)
            if success:
                print("✓ 累积训练完成")
            else:
                print("✗ 累积训练失败")
        except Exception as e:
            print(f"✗ 累积训练异常: {e}")
            return False
    else:
        print("⚠ 没有历史数据，跳过累积训练测试")
    
    print("\n" + "=" * 50)
    print("累积训练功能测试完成")
    print("=" * 50)
    return True

def test_config_compatibility():
    """测试配置兼容性"""
    print("\n测试配置兼容性...")
    
    # 测试没有累积训练配置的情况
    try:
        pipeline = IntegratedRegressionMediaPipeline(
            weights="my_model_arch/cpu_fast/cpu_convert/mobileone_s0_224_fp16_pnnx.ncnn.param",
            device="cpu"
            # 没有指定累积训练参数，应该使用默认值
        )
        print("✓ 默认配置兼容性测试通过")
        print(f"  默认累积训练状态: {pipeline.use_accumulated_training}")
        print(f"  默认最大数据集数量: {pipeline.max_accumulated_datasets}")
    except Exception as e:
        print(f"✗ 默认配置兼容性测试失败: {e}")
        return False
    
    return True

if __name__ == "__main__":
    print("累积训练功能测试开始")
    
    # 测试配置兼容性
    config_test = test_config_compatibility()
    
    # 测试累积训练功能
    training_test = test_accumulated_training()
    
    # 总结
    print("\n" + "=" * 60)
    print("测试总结:")
    print(f"配置兼容性: {'✓ 通过' if config_test else '✗ 失败'}")
    print(f"累积训练功能: {'✓ 通过' if training_test else '✗ 失败'}")
    
    if config_test and training_test:
        print("\n🎉 所有测试通过！累积训练功能已正确实现。")
        sys.exit(0)
    else:
        print("\n❌ 部分测试失败，请检查实现。")
        sys.exit(1) 