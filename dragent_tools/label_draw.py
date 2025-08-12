import cv2
import numpy as np

# 1. 解析 YOLO 标签
def parse_yolo_label(label_path, img_w, img_h):
    boxes = []
    with open(label_path) as f:
        for line in f:
            cls, x_c, y_c, w, h = map(float, line.strip().split())
            # 反归一化到像素坐标
            x1 = int((x_c - w/2) * img_w)
            y1 = int((y_c - h/2) * img_h)
            x2 = int((x_c + w/2) * img_w)
            y2 = int((y_c + h/2) * img_h)
            boxes.append((int(cls), x1, y1, x2, y2))
    return boxes

# 2. 画框函数
def draw_boxes(img, boxes, class_names=None, thickness=2):
    # 预先算出该类会用到的颜色，避免每次重复计算
    max_cls = max([b[0] for b in boxes]) if boxes else 0
    # 生成 HSV 色调：类别越大 -> 色调越接近 0（红色）
    # 区间：黄色≈30，红色≈0；所以映射 30→0
    hues = np.linspace(30, 0, max_cls + 1)  # 0..max_cls 的色调
    colors = {}
    for cls in range(max_cls + 1):
        hsv = np.uint8([[[int(hues[cls]), 255, 255]]])
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
        colors[cls] = tuple(int(v) for v in bgr)

    for cls, x1, y1, x2, y2 in boxes:
        color = colors[cls]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
        label = class_names[cls] if class_names else f'class{cls}'
        cv2.putText(img, label, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, thickness)
    return img

# 3. 主函数
def label_draw(image_path, label_path, output_path, class_names=None):
    # 读入图像
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"无法读取图像文件: {image_path}")
    h, w = img.shape[:2]

    # 解析标签
    boxes = parse_yolo_label(label_path, w, h)

    # 画框
    img_with_boxes = draw_boxes(img, boxes, class_names)

    # 保存结果
    cv2.imwrite(output_path, img_with_boxes)
    print(f"结果已保存到: {output_path}")

# 示例用法
if __name__ == "__main__":
    image_path = "../agents/origin.JPG"  # 输入图像路径
    label_path = "labels.txt"      # YOLO 标签文件路径
    output_path = "output.jpg"     # 输出图像路径
    #class_names = ["class0", "class1", "class2"]  # 类别名称列表，可选

    label_draw(image_path, label_path, output_path) #class_names)