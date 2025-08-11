import requests
import base64
from merge_mask import blend_mask_to_image

def gen_mask(image_byte):
    with open("result.jpg", "rb") as f:
        image_bytes = f.read()
    return {'text': '识别出水体', 'result': image_bytes}


def gen_mask_pending(image_byte, server_url="http://localhost:9000/analyze"):
    """
    用法:
    res = gen_mask(image_bytes)
    返回一个字典，包含文本和掩膜字节
    {'text': '生成的文本描述','mask_bytes': b'掩膜的二进制数据'}//res

    """
    file= {'image': ('image.jpg', image_byte, 'image/jpeg')}
    files = {'image': file}
    resp = requests.post(server_url, files=files)

    resp.raise_for_status()
    result = resp.json()

    # 保存掩膜
    mask_b64 = result["mask_base64"]
    mask_bytes = base64.b64decode(mask_b64)
    mask_path = "mask_output.png"
    img_path = "origin.jpg"
    output_path = "result.jpg"
    with open(mask_path, "wb") as f:
        f.write(mask_bytes)
    with open(img_path, "wb") as f:
        f.write(image_byte)
    result = blend_mask_to_image(img_path, mask_path, output_path,color=(0, 255, 255), alpha=0.5)  # 黄色标记
    return {'text': result["text"], 'result': result }



# ===== 示例调用 =====
if __name__ == "__main__":
    with open("origin.JPG", "rb") as f:
        image_bytes = f.read()
    res=gen_mask(image_bytes)
    mask_b64 = res["mask_base64"]
    print("minicpm文本输出:", res["text"])
