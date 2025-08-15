# yolo_tool.py
from ultralytics import YOLO
import os
import shutil
from pathlib import Path
from typing import Tuple

# ------------------------------------------------------------------
# 1. 全局配置
# ------------------------------------------------------------------
# 输出根目录
OUTPUT_ROOT = Path("../agents")

# 模型权重
MODEL_PATH = Path("../yolov8x/best.pt")

# ------------------------------------------------------------------
# 2. 工具函数
# ------------------------------------------------------------------
def _clear_predict_dir() -> None:
    """删除旧的 predict 目录，保证每次干净运行"""
    predict_dir = OUTPUT_ROOT / "predict"
    if predict_dir.exists():
        shutil.rmtree(predict_dir)

def run_yolo(image_path: str) -> Tuple[Path, int]:
    """
    对单张图片进行 YOLO 检测并保存结果。

    Args:
        image_path: 图片的完整路径（str 或 Path 均可）

    Returns:
        (labeled_img_path, boxes_count)
        labeled_img_path: 带框图的完整路径
        boxes_count:      检测到的损毁房屋数量
    """
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(image_path)

    # 清理旧目录
    _clear_predict_dir()

    # 加载模型
    model = YOLO(str(MODEL_PATH))

    # 运行预测
    results = model.predict(
        source=str(image_path),
        conf=0.5,
        save=True,
        save_txt=True,
        save_conf=True,
        project=OUTPUT_ROOT,  # 根目录
        name="predict",       # 子目录
        exist_ok=True,
    )

    # 计算检测框数量（损毁房屋）
    total_boxes = sum(len(r.boxes) for r in results)

    # 找到带框图：YOLO 会生成 predict/原文件名.后缀
    labeled_img_path = OUTPUT_ROOT / "predict" / image_path.name

    return labeled_img_path, total_boxes

# ------------------------------------------------------------------
# 3. 示例（仅在直接运行本脚本时执行）
# ------------------------------------------------------------------
if __name__ == "__main__":
    test_img = Path("../data/img/origin.jpg")
    path, count = run_yolo(test_img)
    print("带框图路径:", path)
    print("损毁房屋数量:", count)