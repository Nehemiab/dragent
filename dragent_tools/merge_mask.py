import cv2
import numpy as np

def blend_mask_to_image(img_path, mask_path,output_path, color=(0, 0, 255), alpha=0.4):
    image = cv2.imread(img_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    """
    将二值掩膜与原图融合，以半透明颜色高亮目标区域。

    参数
    image : np.ndarray
        原图，BGR 格式，shape [H, W, 3]，dtype uint8。
    mask  : np.ndarray
        二值掩膜，shape [H, W]，dtype bool / uint8，非 0 即视为前景。
    color : tuple(int, int, int)
        标记颜色，默认红色 (B, G, R)。
    alpha : float
        掩膜区域的不透明度，0~1，越大越不透明。
    返回
    blended : np.ndarray
        融合后的图像，与原图同尺寸同 dtype。
    """
    if image.shape[:2] != mask.shape[:2]:
        raise ValueError("image 与 mask 尺寸必须一致")

    # mask → uint8 单通道
    mask_u8 = (mask > 0).astype(np.uint8) * 255

    # 生成彩色叠加层
    overlay = np.zeros_like(image)
    overlay[:] = color                    # 全图填充指定颜色

    # 仅保留掩膜区域
    overlay = cv2.bitwise_and(overlay, overlay, mask=mask_u8)

    # 融合：output = alpha * overlay + (1-alpha) * image
    blended = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

    # 掩膜外区域保留原图
    inv_mask = cv2.bitwise_not(mask_u8)
    background = cv2.bitwise_and(image, image, mask=inv_mask)
    blended = cv2.add(blended, background)
    cv2.imwrite(output_path, blended)
    with open(output_path, "rb") as f:
        result = f.read()
    return result


# ---------------------------------------------
# 直接运行的示例
# ---------------------------------------------
if __name__ == "__main__":
    img_path  = "GF2_PMS1__L1A0000564539-MSS1_1_2.png"
    mask_path = "GF2_PMS1__L1A0000564539-MSS1_1_2_mask.png"
    output_path = "result.jpg"
    res=blend_mask_to_image(img_path, mask_path, output_path,color=(0, 255, 255), alpha=0.5)  # 黄色标记
    print("已生成 result.jpg")