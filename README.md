# 基于YOLOv8的行人检测系统

基于Ultralytics YOLO框架的行人检测项目，使用Kaggle行人检测数据集训练，支持person和person-like两类目标检测。

## 项目结构

```
d:\RG\
├── archive/                    # 原始数据集（PASCAL VOC格式）
│   ├── Train/Train/
│   │   ├── JPEGImages/         # 训练集图片（944张）
│   │   └── Annotations/        # 训练集XML标注
│   └── Test/Test/
│       ├── JPEGImages/         # 测试集图片（235张）
│       └── Annotations/        # 测试集XML标注
├── person_dataset/             # 转换后的YOLO格式数据集
│   ├── images/train/           # 训练集图片
│   ├── images/val/             # 验证集图片
│   ├── labels/train/           # 训练集YOLO标签
│   ├── labels/val/             # 验证集YOLO标签
│   └── person.yaml             # 数据集配置文件
├── runs/                       # 训练与评估输出
│   ├── person_detect/          # 训练结果
│   ├── person_eval/            # 评估结果
│   └── person_predict/         # 预测结果
├── dataset_samples/            # 数据集示例图片
├── output_images/              # 可视化预测图片
├── step1_voc_to_yolo.py        # Step1: VOC格式转YOLO格式
├── step2_train.py              # Step2: 模型训练
├── step3_evaluate.py           # Step3: 模型评估
├── step4_visualize.py          # Step4: 可视化预测
├── step5_dataset_explore.py    # Step5: 数据集统计分析
├── run_all.py                  # 一键运行脚本
└── baogao.md                   # 项目设计报告
```

## 环境要求

- Python 3.10+
- PyTorch 1.12+
- ultralytics
- OpenCV (cv2)

安装依赖：

```bash
pip install ultralytics opencv-python
```

## 使用方法

### 一键运行

```bash
python run_all.py
```

### 分步运行

**Step 1: 数据预处理** — 将VOC格式转换为YOLO格式

```bash
python step1_voc_to_yolo.py
```

**Step 2: 数据集探索** — 统计数据集信息和保存示例图片

```bash
python step5_dataset_explore.py
```

**Step 3: 模型训练** — 使用YOLOv11n预训练权重微调

```bash
python step2_train.py
```

**Step 4: 模型评估** — 在验证集上计算mAP等指标

```bash
python step3_evaluate.py
```

**Step 5: 可视化预测** — 生成带检测框的预测图片

```bash
python step4_visualize.py
```

## 数据集

- **来源**: [Kaggle Pedestrian Detection](https://www.kaggle.com/datasets/karthika95/pedestrian-detection)
- **格式**: PASCAL VOC (XML标注)
- **大小**: 约166.3MB
- **类别**: person (行人)、person-like (类行人)
- **划分**: 训练集944张 / 测试集235张

## 训练参数

| 参数 | 值 |
|------|-----|
| 模型 | YOLOv11n (预训练) |
| Epochs | 50 |
| 图像尺寸 | 640×640 |
| Batch Size | 8 |
| Device | GPU (device=0) |
| 早停耐心值 | 15 |
| 置信度阈值 | 0.25 |
| IoU阈值 | 0.6 |

## 评估指标

- mAP@0.5
- mAP@0.5:0.95
- Precision
- Recall

训练曲线、PR曲线、混淆矩阵等图表保存在 `runs/person_detect/` 目录下。
