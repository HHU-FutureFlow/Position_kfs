from struck.strukt import xywh

# ---------------------- 双指数平滑核心超参数（针对你的场景，推荐默认值，可微调） ----------------------
ALPHA = 0.25  # 平滑系数：0.2~0.4（越小抖动抑制越强，推荐0.25）
BETA = 0.2   # 趋势系数：0.1~0.3（越小趋势越稳定，推荐0.15）
# 调参口诀：抖动能看见→减小ALPHA；框跟目标慢→增大BETA（每次微调0.05）

# 全局变量：存储上一帧的平滑值（S_prev）和趋势值（T_prev），全程xywh对象
# 为什么用全局？无需修改主程序，直接在滤波器内部维护状态，对用户透明
S_prev = xywh()  # 上一帧平滑值
T_prev = xywh()  # 上一帧趋势值


def double_exponential_smooth(last_S, last_T, current_val, alpha, beta):
    """
    单值双指数平滑核心函数（x/y/w/h通用）
    :param last_S: 上一帧平滑值
    :param last_T: 上一帧趋势值
    :param current_val: 当前帧检测值（整数）
    :param alpha: 平滑系数
    :param beta: 趋势系数
    :return: (当前帧平滑值, 当前帧趋势值) → 均为整数，适配像素坐标
    """
    # 双指数平滑核心计算
    current_S = alpha * current_val + (1 - alpha) * (last_S + last_T)
    current_T = beta * (current_S - last_S) + (1 - beta) * last_T
    # 取整为整数（像素坐标无小数）
    return round(current_S), round(current_T)


def initialize(last_results, current_results):
    """
    主平滑函数：完全兼容原有调用方式，输入输出均为xywh对象
    """
    global S_prev, T_prev  # 声明使用全局变量，维护滤波器状态

    # 情况1：首次检测/无检测后重新检测（上一帧为初始0值）→ 初始化滤波器状态，直接返回当前值
    if last_results.x == 0 and last_results.y == 0 and last_results.w == 0 and last_results.h == 0:
        # 初始化平滑值=当前检测值，趋势值=0（无历史趋势）
        S_prev.x, T_prev.x = current_results.x, 0
        S_prev.y, T_prev.y = current_results.y, 0
        S_prev.w, T_prev.w = current_results.w, 0
        S_prev.h, T_prev.h = current_results.h, 0
        return current_results  # 首次帧无平滑，返回原始检测值

    # 情况2：正常检测→对x/y/w/h分别做双指数平滑
    final_results = xywh()
    # 对每个分量单独平滑，更新平滑值和趋势值
    final_results.x, T_prev.x = double_exponential_smooth(S_prev.x, T_prev.x, current_results.x, ALPHA, BETA)
    final_results.y, T_prev.y = double_exponential_smooth(S_prev.y, T_prev.y, current_results.y, ALPHA, BETA)
    final_results.w, T_prev.w = double_exponential_smooth(S_prev.w, T_prev.w, current_results.w, ALPHA, BETA)
    final_results.h, T_prev.h = double_exponential_smooth(S_prev.h, T_prev.h, current_results.h, ALPHA, BETA)

    # 更新全局平滑值，供下一帧使用
    S_prev = final_results

    # 返回平滑后的xywh对象（直接传给画框函数）
    return final_results


# 重置滤波器状态的函数（可选，主程序无检测时可调用，增强鲁棒性）
def reset_filter():
    global S_prev, T_prev
    S_prev = xywh()
    T_prev = xywh()