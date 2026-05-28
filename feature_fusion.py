import numpy as np
import cv2
import os

def process_single_npy(input_npy_path, output_png_path, target_size=(640, 640)):
    # 1. 确保输出目录存在，如果不存在则自动创建
    os.makedirs(os.path.dirname(output_png_path), exist_ok=True)

    # 2. 加载 .npy 文件 (加上 allow_pickle=True 防止报错)
    features = np.load(input_npy_path, allow_pickle=True)

    # --- 处理可能被打包的对象，确保提取出纯数值矩阵 ---
    if features.dtype == object and features.size == 1:
        features = features.item()
    if isinstance(features, list):
        features = np.array(features)
    if hasattr(features, 'cpu'):
        features = features.detach().cpu().numpy()
    # --------------------------

    # 3. 去掉 Batch 维度 [1, C, H, W] -> [C, H, W]
    if len(features.shape) == 4:
        features = features[0]

    # 4. 求平均，汇总所有通道特征 -> [H, W]
    summary_feature = np.mean(features, axis=0)

    # 5. 归一化：将数值映射到 0~255 的灰度范围
    summary_feature = (summary_feature - summary_feature.min()) / (summary_feature.max() - summary_feature.min() + 1e-5)
    summary_feature = (summary_feature * 255).astype(np.uint8)

    # 6. 放大尺寸：让小尺寸特征图更清晰
    if target_size:
        summary_feature = cv2.resize(summary_feature, target_size, interpolation=cv2.INTER_LINEAR)

    # 7. 应用伪彩色 ！！！【核心改动：换成 VIRIDIS 冷色调】！！！
    # COLORMAP_VIRIDIS: 从暗紫(低激活)过渡到亮黄/亮绿(高激活)，非常适合深色背景展示
    heatmap = cv2.applyColorMap(summary_feature, cv2.COLORMAP_VIRIDIS)

    # 8. 解决中文路径保存问题 ！！！
    # 将图片编码为 .png 格式的数据流，然后用 tofile 写入包含中文的路径
    is_success, im_buf_arr = cv2.imencode(".png", heatmap)
    if is_success:
        im_buf_arr.tofile(output_png_path)
        print(f"✅ 转换成功！深色调汇总特征图已保存为:\n{output_png_path}")
    else:
        print("❌ 转换失败：图像编码出错。")

# ================= 修改这里的文件名 =================
# 填入你要处理的那个 .npy 文件的实际名称 (注意保留路径前面的 r)
input_file = r"D:\云大\小论文\可视化结果\特征图\B+C\predict\001060\stage1_Conv_features.npy"

# 填入你想保存的图片名称 (建议在名字里加个标识，比如 _viridis，方便区分)
output_file = r"D:\云大\小论文\可视化结果\特征图\B+C\stage1_summary_viridis.png"

# 执行转换
process_single_npy(input_file, output_file)