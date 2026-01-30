import cv2
import numpy as np
# 1. 加载字典对象
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)

for i in range(4):
    # 2. 创建一个空的 numpy 数组来存放生成的标记
    # 注意：必须先创建这个空数组
    marker_size = 200
    marker_img = np.zeros((marker_size, marker_size), dtype=np.uint8)

    # 3. 使用 generateImageMarker (注意参数顺序和写法)
    # 这是一个“就地”修改的函数，它会把结果写入上面创建的 marker_img 中
    cv2.aruco.generateImageMarker(aruco_dict, i, marker_size, marker_img)

    # 4. 保存图片
    filename = f"{i}.png"
    cv2.imwrite(filename, marker_img)
    print(f"已生成: {filename}")