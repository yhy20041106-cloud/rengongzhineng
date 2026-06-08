"""
行人检测实验 - 基于YOLOv8
数据集：PASCAL VOC格式行人检测数据集
步骤1：将VOC格式转换为YOLO格式
"""

import os
import xml.etree.ElementTree as ET
import shutil
from pathlib import Path


# ============ 配置路径 ============
BASE_DIR = Path(r"d:\RG")
ARCHIVE_DIR = BASE_DIR / "archive"
TRAIN_IMG_DIR = ARCHIVE_DIR / "Train" / "Train" / "JPEGImages"
TRAIN_ANN_DIR = ARCHIVE_DIR / "Train" / "Train" / "Annotations"
TEST_IMG_DIR = ARCHIVE_DIR / "Test" / "Test" / "JPEGImages"
TEST_ANN_DIR = ARCHIVE_DIR / "Test" / "Test" / "Annotations"

# YOLO数据集输出目录
YOLO_DATASET_DIR = BASE_DIR / "person_dataset"
YOLO_TRAIN_IMG = YOLO_DATASET_DIR / "images" / "train"
YOLO_VAL_IMG = YOLO_DATASET_DIR / "images" / "val"
YOLO_TRAIN_LABEL = YOLO_DATASET_DIR / "labels" / "train"
YOLO_VAL_LABEL = YOLO_DATASET_DIR / "labels" / "val"


def voc_to_yolo(xml_path, img_width, img_height):
    """将单个VOC XML标注转换为YOLO格式标签"""
    # 类别映射: person -> 0, person-like -> 1
    CLASS_MAP = {"person": 0, "person-like": 1}

    tree = ET.parse(xml_path)
    root = tree.getroot()
    labels = []

    for obj in root.findall("object"):
        name = obj.find("name").text
        if name not in CLASS_MAP:
            continue

        class_id = CLASS_MAP[name]

        difficult = obj.find("difficult")
        if difficult is not None and int(difficult.text) == 1:
            continue

        bndbox = obj.find("bndbox")
        xmin = float(bndbox.find("xmin").text)
        ymin = float(bndbox.find("ymin").text)
        xmax = float(bndbox.find("xmax").text)
        ymax = float(bndbox.find("ymax").text)

        # 裁剪边界框到图像范围内
        xmin = max(0, min(xmin, img_width))
        ymin = max(0, min(ymin, img_height))
        xmax = max(0, min(xmax, img_width))
        ymax = max(0, min(ymax, img_height))

        # 计算YOLO格式：class x_center y_center width height（归一化）
        x_center = (xmin + xmax) / 2.0 / img_width
        y_center = (ymin + ymax) / 2.0 / img_height
        w = (xmax - xmin) / img_width
        h = (ymax - ymin) / img_height

        # 确保值在[0,1]范围内
        x_center = max(0, min(1, x_center))
        y_center = max(0, min(1, y_center))
        w = max(0, min(1, w))
        h = max(0, min(1, h))

        if w > 0 and h > 0:
            labels.append(f"{class_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}")

    return labels


def get_image_size(xml_path):
    """从XML中获取图像尺寸"""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    size = root.find("size")
    if size is not None:
        width = int(size.find("width").text)
        height = int(size.find("height").text)
        return width, height
    return None, None


def convert_dataset(img_dir, ann_dir, output_img_dir, output_label_dir):
    """转换整个数据集"""
    os.makedirs(output_img_dir, exist_ok=True)
    os.makedirs(output_label_dir, exist_ok=True)

    converted = 0
    skipped = 0

    # 获取所有图片文件
    img_files = list(Path(img_dir).glob("*.jpg"))
    print(f"  找到 {len(img_files)} 张图片")

    for img_path in img_files:
        img_name = img_path.stem  # 如 "image (1)"

        # 查找对应的XML标注文件
        xml_path = Path(ann_dir) / f"{img_name}.xml"

        # 复制图片
        dst_img = output_img_dir / img_path.name
        shutil.copy2(img_path, dst_img)

        # 转换标注
        if xml_path.exists():
            width, height = get_image_size(xml_path)
            if width and height:
                labels = voc_to_yolo(xml_path, width, height)
                label_path = output_label_dir / f"{img_name}.txt"
                with open(label_path, "w", encoding="utf-8") as f:
                    f.write("\n".join(labels))
                converted += 1
            else:
                # 无法获取尺寸，创建空标注
                label_path = output_label_dir / f"{img_name}.txt"
                with open(label_path, "w", encoding="utf-8") as f:
                    f.write("")
                skipped += 1
        else:
            # 没有标注文件，创建空标注
            label_path = output_label_dir / f"{img_name}.txt"
            with open(label_path, "w", encoding="utf-8") as f:
                f.write("")
            skipped += 1

    print(f"  转换完成: {converted} 张有标注, {skipped} 张无标注/跳过")
    return converted, skipped


def create_yaml_config():
    """创建YOLOv8数据集配置文件"""
    yaml_content = f"""# Person Detection Dataset
path: {YOLO_DATASET_DIR}
train: images/train
val: images/val

# 类别数量
nc: 2

# 类别名称
names:
  0: person
  1: person-like
"""
    yaml_path = YOLO_DATASET_DIR / "person.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(yaml_content)
    print(f"数据集配置文件已创建: {yaml_path}")
    return yaml_path


def main():
    print("=" * 60)
    print("VOC格式 -> YOLO格式 数据转换")
    print("=" * 60)

    # 转换训练集
    print("\n[1/2] 转换训练集...")
    train_converted, train_skipped = convert_dataset(
        TRAIN_IMG_DIR, TRAIN_ANN_DIR, YOLO_TRAIN_IMG, YOLO_TRAIN_LABEL
    )

    # 转换测试集（作为验证集）
    print("\n[2/2] 转换测试集（作为验证集）...")
    val_converted, val_skipped = convert_dataset(
        TEST_IMG_DIR, TEST_ANN_DIR, YOLO_VAL_IMG, YOLO_VAL_LABEL
    )

    # 创建配置文件
    print("\n[3/3] 创建数据集配置文件...")
    yaml_path = create_yaml_config()

    # 统计信息
    print("\n" + "=" * 60)
    print("数据集统计信息:")
    print(f"  训练集: {train_converted} 张有标注图片, {train_skipped} 张无标注")
    print(f"  验证集: {val_converted} 张有标注图片, {val_skipped} 张无标注")
    print(f"  总计: {train_converted + val_converted} 张有标注图片")
    print(f"  类别: person (行人), person-like (类行人)")
    print(f"  配置文件: {yaml_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
