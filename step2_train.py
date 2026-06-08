"""
行人检测实验 - 基于YOLOv8
步骤2：模型训练
"""

import os
from pathlib import Path
from ultralytics import YOLO


def train():
    print("=" * 60)
    print("YOLOv8 行人检测模型训练")
    print("=" * 60)

    # 数据集配置文件路径
    yaml_path = Path(r"d:\RG\person_dataset\person.yaml")

    # 加载预训练模型（YOLOv11n - 小型模型，适合课程实验）
    model = YOLO("yolo11n.pt")

    # 开始训练
    results = model.train(
        data=str(yaml_path),
        epochs=50,           # 训练轮数
        imgsz=640,           # 输入图像尺寸
        batch=8,             # 批次大小
        name="person_detect",  # 实验名称
        project=str(Path(r"d:\RG\runs")),
        device=0,        # 使用CPU训练（如有GPU可改为0）
        patience=15,         # 早停耐心值
        save=True,           # 保存模型
        save_period=10,      # 每10轮保存一次
        plots=True,          # 生成训练图表
        val=True,            # 每轮验证
        verbose=True,        # 详细输出
    )

    print("\n" + "=" * 60)
    print("训练完成!")
    # print(f"最佳模型保存在: {Path(r'd:\RG\runs\person_detect\weights\best.pt')}")
    # print(f"最后模型保存在: {Path(r'd:\RG\runs\person_detect\weights\last.pt')}")
    print("=" * 60)

    return results


if __name__ == "__main__":
    train()
