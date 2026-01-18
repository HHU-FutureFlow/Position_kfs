# 这是一个示例 Python 脚本。

# 按 Shift+F10 执行或将其替换为您的代码。
# 按 双击 Shift 在所有地方搜索类、文件、工具窗口、操作和设置。

#开机，测出给定像素的深度信息，将相机坐标系下三维坐标进行转换变成基坐标系下三维坐标
from concurrent.futures import ThreadPoolExecutor
import Function as fc
from ultralytics import YOLO



if __name__ == '__main__':
    df, cf, fst = fc.RealsenseConfig()
    model = YOLO(r'D:\Position\best.pt')
    print(f"成功加载模型")
    while True:
        result = fc.YoloDetect(model,cf)
        if result[0].boxes == None:
            fc.drawFrame(result, df, [0,0], fst)
            continue
        boxes_xywh = result[0].boxes.xywh.cpu().numpy()
        if len(boxes_xywh) == 0:
            fc.drawFrame(result, df,[0,0], fst)
            continue
        Class_ids, Target_x, Target_y, Target_z = fc.positiondetect(df,result)
        target_position = [Target_x,Target_y,Target_z]
        fc.drawFrame(result, df, target_position, fst)





# 访问 https://www.jetbrains.com/help/pycharm/ 获取 PyCharm 帮助