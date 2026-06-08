"""
行人检测实验 - 一键运行脚本
基于YOLOv8的行人检测完整流程
"""

import os
import sys
from pathlib import Path


def main():
    print("=" * 60)
    print("  YOLOv8 行人检测实验 - 一键运行")
    print("=" * 60)

    base_dir = Path(r"d:\RG")

    # ============ Step 1: 数据预处理 ============
    print("\n" + "=" * 60)
    print("Step 1: VOC格式 -> YOLO格式 数据转换")
    print("=" * 60)
    os.system(f"python {base_dir / 'step1_voc_to_yolo.py'}")

    # ============ Step 2: 数据集探索 ============
    print("\n" + "=" * 60)
    print("Step 2: 数据集探索与统计")
    print("=" * 60)
    os.system(f"python {base_dir / 'step5_dataset_explore.py'}")

    # ============ Step 3: 模型训练 ============
    print("\n" + "=" * 60)
    print("Step 3: 模型训练")
    print("=" * 60)
    os.system(f"python {base_dir / 'step2_train.py'}")

    # ============ Step 4: 模型评估 ============
    print("\n" + "=" * 60)
    print("Step 4: 模型评估")
    print("=" * 60)
    os.system(f"python {base_dir / 'step3_evaluate.py'}")

    # ============ Step 5: 可视化预测 ============
    print("\n" + "=" * 60)
    print("Step 5: 可视化预测")
    print("=" * 60)
    os.system(f"python {base_dir / 'step4_visualize.py'}")

    print("\n" + "=" * 60)
    print("  所有步骤执行完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()
