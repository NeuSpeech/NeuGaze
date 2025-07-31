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

import yaml
from typing import Dict, Any

class ExpressionEvaluator:
    def __init__(self, config: Dict[str, Any]):
        self.config = config

    def evaluate_condition(self, condition: Dict[str, Any], features: Dict[str, float]) -> bool:
        feature = condition['feature']
        operator = condition['operator']
        
        if operator == ">":
            return features[feature] > condition['threshold']
        elif operator == "<":
            return features[feature] < condition['threshold']
        elif operator == "BETWEEN":
            return condition['min'] < features[feature] < condition['max']
        elif operator == "DIFF>":
            return (features[feature] - features[condition['compare_to']]) > condition['threshold']
        elif operator == "DIFF<":
            return abs(features[feature] - features[condition['compare_to']]) < condition['threshold']
        return False

    def evaluate_expression(self, expression_config: Dict[str, Any], features: Dict[str, float]) -> bool:
        conditions = expression_config['conditions']
        combine = expression_config.get('combine', 'AND')
        
        results = [self.evaluate_condition(cond, features) for cond in conditions]
        
        if combine == "AND":
            return all(results)
        elif combine == "OR":
            return any(results)
        return False

    def apply_priority_rules(self, keys_dict: Dict[str, bool]) -> Dict[str, bool]:
        """应用优先级规则"""
        for rule in self.config.get('priority_rules', []):
            if rule['when'] == 'any':
                # 特殊处理 left_click 的规则
                if 'left_click' in rule['disable']:
                    # 检查是否有其他键被激活
                    other_keys_active = any(keys_dict[k] for k in keys_dict if k != 'left_click')
                    if other_keys_active:
                        keys_dict['left_click'] = False
            else:
                # 处理其他规则
                if keys_dict.get(rule['when'], False):
                    for key in rule['disable']:
                        keys_dict[key] = False
        return keys_dict

    def evaluate_all(self, features: Dict[str, float]) -> Dict[str, bool]:
        keys_dict = {}
        for expr_name, expr_config in self.config['expressions'].items():
            # print(expr_name,expr_config)
            keys_dict[expr_name] = self.evaluate_expression(expr_config, features)
        
        return self.apply_priority_rules(keys_dict)

if __name__ == "__main__":
    # 定义多组测试数据
    test_cases = [
        {   # 测试用例1：原始数据
            'name': "原始测试数据",
            'features': {
                'jawOpen': 0.5, 'jawLeft': 0.05, 'jawRight': 0.05,
                'mouthRollLower': 0.5, 'mouthRollUpper': 0.5,
                'mouthSmileLeft': 0.35, 'mouthSmileRight': 0.15,
                'mouthLeft': 0.25, 'mouthRight': 0.25,
                'mouthPressLeft': 0.45, 'mouthPressRight': 0.45,
                'mouthUpperUpLeft': 0.6, 'mouthUpperUpRight': 0.6,
                'mouthLowerDownLeft': 0.4, 'mouthLowerDownRight': 0.4,
                'browInnerUp': 0.9, 'mouthFunnel': 0.1,
                'mouthPucker': 0.98, 'eyeBlinkLeft': 0.7, 'eyeBlinkRight': 0.2
            }
        },
        {   # 测试用例2：测试num2和num7的优先级规则
            'name': "优先级规则测试",
            'features': {
                'jawOpen': 0.1, 'jawLeft': 0.05, 'jawRight': 0.05,
                'mouthRollLower': 0.3, 'mouthRollUpper': 0.3,
                'mouthSmileLeft': 0.5, 'mouthSmileRight': 0.5,  # num2应该激活
                'mouthLeft': 0.1, 'mouthRight': 0.1,
                'mouthPressLeft': 0.3, 'mouthPressRight': 0.3,
                'mouthUpperUpLeft': 0.6, 'mouthUpperUpRight': 0.6,  # num7也应该激活
                'mouthLowerDownLeft': 0.4, 'mouthLowerDownRight': 0.4,
                'browInnerUp': 0.3, 'mouthFunnel': 0.1,
                'mouthPucker': 0.5, 'eyeBlinkLeft': 0.2, 'eyeBlinkRight': 0.2
            }
        },
        {   # 测试用例3：测试left_click的优先级规则
            'name': "Left Click测试",
            'features': {
                'jawOpen': 0.5, 'jawLeft': 0.05, 'jawRight': 0.05,
                'mouthRollLower': 0.3, 'mouthRollUpper': 0.3,
                'mouthSmileLeft': 0.1, 'mouthSmileRight': 0.1,
                'mouthLeft': 0.1, 'mouthRight': 0.1,
                'mouthPressLeft': 0.3, 'mouthPressRight': 0.3,
                'mouthUpperUpLeft': 0.1, 'mouthUpperUpRight': 0.1,
                'mouthLowerDownLeft': 0.1, 'mouthLowerDownRight': 0.1,
                'browInnerUp': 0.3, 'mouthFunnel': 0.1,
                'mouthPucker': 0.98, 'eyeBlinkLeft': 0.2, 'eyeBlinkRight': 0.2
            }
        },
        {   # 测试用例4：测试多个表情同时激活
            'name': "多表情测试",
            'features': {
                'jawOpen': 0.5, 'jawLeft': 0.4, 'jawRight': 0.4,
                'mouthRollLower': 0.5, 'mouthRollUpper': 0.5,
                'mouthSmileLeft': 0.5, 'mouthSmileRight': 0.5,
                'mouthLeft': 0.3, 'mouthRight': 0.3,
                'mouthPressLeft': 0.5, 'mouthPressRight': 0.5,
                'mouthUpperUpLeft': 0.6, 'mouthUpperUpRight': 0.6,
                'mouthLowerDownLeft': 0.4, 'mouthLowerDownRight': 0.4,
                'browInnerUp': 0.9, 'mouthFunnel': 0.15,
                'mouthPucker': 0.98, 'eyeBlinkLeft': 0.7, 'eyeBlinkRight': 0.2
            }
        }
    ]

    # 新写法的配置和结果
    test_config = {
        'expressions': {
            'numlock': {
                'conditions': [
                    {'feature': 'jawOpen', 'operator': '>', 'threshold': 0.4},
                    {'feature': 'jawLeft', 'operator': '<', 'threshold': 0.1},
                    {'feature': 'jawRight', 'operator': '<', 'threshold': 0.1}
                ],
                'combine': 'AND'
            },
            'num0': {
                'conditions': [
                    {'feature': 'mouthRollLower', 'operator': '>', 'threshold': 0.45},
                    {'feature': 'mouthRollUpper', 'operator': '>', 'threshold': 0.45}
                ],
                'combine': 'AND'
            },
            'num1': {
                'conditions': [
                    {'feature': 'mouthSmileLeft', 'operator': 'BETWEEN', 'min': 0.25, 'max': 0.45},
                    {'feature': 'mouthSmileLeft', 'operator': 'DIFF>', 'compare_to': 'mouthSmileRight', 'threshold': 0.15}
                ],
                'combine': 'AND'
            },
            'num2': {
                'conditions': [
                    {'feature': 'mouthSmileLeft', 'operator': '>', 'threshold': 0.45},
                    {'feature': 'mouthSmileRight', 'operator': '>', 'threshold': 0.45},
                    {'feature': 'mouthSmileLeft', 'operator': 'DIFF<', 'compare_to': 'mouthSmileRight', 'threshold': 0.2}
                ],
                'combine': 'AND'
            },
            'num3': {
                'conditions': [
                    {'feature': 'mouthSmileRight', 'operator': 'BETWEEN', 'min': 0.25, 'max': 0.45},
                    {'feature': 'mouthSmileRight', 'operator': 'DIFF>', 'compare_to': 'mouthSmileLeft', 'threshold': 0.15}
                ],
                'combine': 'AND'
            },
            'num4': {
                'conditions': [
                    {'feature': 'mouthLeft', 'operator': '>', 'threshold': 0.2},
                    {'feature': 'jawOpen', 'operator': '<', 'threshold': 0.05},
                    {'feature': 'mouthSmileLeft', 'operator': '<', 'threshold': 0.2},
                    {'feature': 'mouthSmileRight', 'operator': '<', 'threshold': 0.2}
                ],
                'combine': 'AND'
            },
            'num5': {
                'conditions': [
                    {'feature': 'mouthPressLeft', 'operator': '>', 'threshold': 0.4},
                    {'feature': 'mouthPressRight', 'operator': '>', 'threshold': 0.4}
                ],
                'combine': 'AND'
            },
            'num6': {
                'conditions': [
                    {'feature': 'mouthRight', 'operator': '>', 'threshold': 0.2},
                    {'feature': 'jawOpen', 'operator': '<', 'threshold': 0.05},
                    {'feature': 'mouthSmileLeft', 'operator': '<', 'threshold': 0.2},
                    {'feature': 'mouthSmileRight', 'operator': '<', 'threshold': 0.2}
                ],
                'combine': 'AND'
            },
            'num7': {
                'conditions': [
                    {'feature': 'mouthUpperUpLeft', 'operator': '>', 'threshold': 0.5},
                    {'feature': 'mouthUpperUpRight', 'operator': '>', 'threshold': 0.5},
                    {'feature': 'mouthLowerDownLeft', 'operator': '>', 'threshold': 0.3},
                    {'feature': 'mouthLowerDownRight', 'operator': '>', 'threshold': 0.3}
                ],
                'combine': 'AND'
            },
            'num8': {
                'conditions': [
                    {'feature': 'browInnerUp', 'operator': '>', 'threshold': 0.8}
                ],
                'combine': 'AND'
            },
            'num9': {
                'conditions': [
                    {'feature': 'mouthFunnel', 'operator': '>', 'threshold': 0.4}
                ],
                'combine': 'AND'
            },
            'right_click': {
                'conditions': [
                    {'feature': 'jawLeft', 'operator': '>', 'threshold': 0.3}
                ],
                'combine': 'AND'
            },
            'mid_click': {
                'conditions': [
                    {'feature': 'jawRight', 'operator': '>', 'threshold': 0.3}
                ],
                'combine': 'AND'
            },
            'left_click': {
                'conditions': [
                    {'feature': 'mouthPucker', 'operator': '>', 'threshold': 0.97},
                    {'feature': 'mouthFunnel', 'operator': '<', 'threshold': 0.2}
                ],
                'combine': 'AND'
            },
            'extra': {
                'conditions': [
                    {'feature': 'eyeBlinkLeft', 'operator': '>', 'threshold': 0.6},
                    {'feature': 'eyeBlinkRight', 'operator': '<', 'threshold': 0.25}
                ],
                'combine': 'AND'
            }
        },
        'priority_rules': [
            {'when': 'num7', 'disable': ['num2']},
            {'when': 'any', 'disable': ['left_click'], 'except': ['left_click']}
        ]
    }

    # 对每个测试用例进行测试
    for test_case in test_cases:
        print(f"\n=== 测试用例: {test_case['name']} ===")
        f = test_case['features']
        
        # 原始写法的结果
        original_keys_dict = {
            'numlock': f['jawOpen'] > 0.4 and f['jawLeft'] < 0.1 and f['jawRight'] < 0.1,
            'num0': f['mouthRollLower'] > 0.45 and f['mouthRollUpper'] > 0.45,
            'num1': 0.25<f['mouthSmileLeft'] < 0.45 and f['mouthSmileLeft'] - f['mouthSmileRight'] > 0.15,
            'num2': f['mouthSmileLeft'] > 0.45 and f['mouthSmileRight'] > 0.45 and abs(
                f['mouthSmileLeft'] - f['mouthSmileRight']) < 0.2,
            'num3': 0.25<f['mouthSmileRight'] < 0.45 and f['mouthSmileRight'] - f['mouthSmileLeft'] > 0.15,
            'num4': f['mouthLeft'] > 0.2 and f['jawOpen'] < 0.05 and
                    f['mouthSmileLeft'] < 0.2 and f['mouthSmileRight'] < 0.2,
            'num5': f['mouthPressLeft'] > 0.4 and f['mouthPressRight'] > 0.4,
            'num6': f['mouthRight'] > 0.2 and f['jawOpen'] < 0.05 and
                    f['mouthSmileLeft'] < 0.2 and f['mouthSmileRight'] < 0.2,
            'num7': f['mouthUpperUpLeft'] > 0.5 and f['mouthUpperUpRight'] > 0.5 and
                    f['mouthLowerDownLeft'] > 0.3 and f['mouthLowerDownRight'] > 0.3,
            'num8': f['browInnerUp'] > 0.8,
            'num9': f['mouthFunnel'] > 0.4,
            'right_click': f['jawLeft'] > 0.3,
            'mid_click': f['jawRight'] > 0.3,
            'left_click': f['mouthPucker'] > 0.97 and f['mouthFunnel'] < 0.2,
            'extra': f['eyeBlinkLeft'] > 0.6 and f['eyeBlinkRight'] < 0.25,
        }

        # 应用原始写法的优先级规则
        if original_keys_dict['num7']:
            original_keys_dict['num2'] = False
        for k, v in original_keys_dict.items():
            if k != 'left_click' and v:
                original_keys_dict['left_click'] = False
                break

        # 使用新写法计算结果
        evaluator = ExpressionEvaluator(test_config)
        new_results = evaluator.evaluate_all(f)

        # 打印不一致的结果
        print("表情\t\t原始写法\t新写法\t\t一致性")
        print("-" * 50)
        has_mismatch = False
        for key in original_keys_dict.keys():
            original = original_keys_dict[key]
            new = new_results[key]
            if original != new:
                has_mismatch = True
                print(f"{key:<12} {str(original):<12} {str(new):<12} ✗")
        
        if not has_mismatch:
            print("所有表情判断结果一致 ✓")