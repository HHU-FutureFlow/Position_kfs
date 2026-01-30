import cv2
import numpy as np
from struck.strukt import *

def xywh2xyxy(xywh):
    """
    将xywh（中心+宽高）转xyxy（左上+右下），适配OpenCV画框需求
    :param xywh: numpy数组，shape=(4,)，格式[x_c, y_c, w, h]，int像素坐标
    :return: numpy数组，shape=(4,)，格式[x1, y1, x2, y2]，int像素坐标
    """
    x_c, y_c, w, h = xywh.x, xywh.y, xywh.w, xywh.h
    x1 = int(x_c - w / 2)
    y1 = int(y_c - h / 2)
    x2 = int(x_c + w / 2)
    y2 = int(y_c + h / 2)
    return np.array([x1, y1, x2, y2], dtype=np.int32)


def draw_box(img, xywh, cls_name="target", conf=0.95, color=(255, 0, 0), thickness=2):
    """
    在图片上绘制目标框，支持输入xywh，自动转xyxy
    :param img: 原始图片，numpy数组（OpenCV格式，BGR通道），YOLO检测的orig_img可直接用
    :param xywh: 目标框xywh坐标，numpy数组，shape=(4,)，int像素坐标
    :param cls_name: 类别名称，字符串，默认"target"
    :param conf: 置信度，浮点数，默认0.95
    :param color: 画框颜色，BGR格式，默认红色(0,0,255)
    :param thickness: 框线宽度，int，默认2
    :return: 绘制了框的图片，numpy数组
    """
    # 坐标合法性校验：避免框超出图片范围
    h, w = img.shape[:2]
    xywh.x = np.clip(xywh.x, 0, w)  # 中心x不超出图片宽
    xywh.y = np.clip(xywh.y, 0, h)  # 中心y不超出图片高
    xywh.w = np.clip(xywh.w, 10, w)  # 宽度最小10像素，最大不超图片宽
    xywh.h = np.clip(xywh.h, 10, h)  # 高度最小10像素，最大不超图片高

    # xywh转xyxy，得到OpenCV需要的左上+右下点
    x1, y1, x2, y2 = xywh2xyxy(xywh)

    # 1. 绘制矩形框
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

    # 2. 绘制类别+置信度标签（可选，可注释）
    label = f"{cls_name}: {conf:.2f}"
    # 计算标签背景大小
    (label_w, label_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    # 标签位置：框左上角，避免超出图片
    label_x = max(x1, 0)
    label_y = max(y1 - label_h - baseline, 0)
    # 绘制标签背景（半透明）
    cv2.rectangle(img, (label_x, label_y), (label_x + label_w, label_y + label_h + baseline), color, -1)
    # 绘制标签文字（白色）
    cv2.putText(img, label, (label_x, label_y + label_h), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return img

def drawpoint(img, point):
    point_cords = (int(point.x), int(point.y))
    radius = 3
    color = (0, 0, 255)
    thickness = 1

    cv2.circle(img, point_cords, radius, color, thickness)

    return img


# -------------------------- 测试画框（可选，注释掉不影响主程序）--------------------------
if __name__ == "__main__":
    # 生成测试图片（640x480，黑色背景）
    test_img = np.zeros((480, 640, 3), dtype=np.uint8)
    # 测试xywh（中心320,240，宽100，高80）
    test_xywh = np.array([320, 240, 100, 80], dtype=np.int32)
    # 画框
    img_with_box = draw_box(test_img, test_xywh, cls_name="weapon", conf=0.98)
    # 显示图片
    cv2.imshow("Test Draw Box", img_with_box)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # 保存图片
    cv2.imwrite("test_box.jpg", img_with_box)