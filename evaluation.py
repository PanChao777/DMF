import cv2
import os
import numpy as np
from uiqm import getUIQM, getUCIQE  # 确保这两个函数接受uint8类型的输入

# 只需要生成图像的目录
gen_dir = "E:/download/yanyi/project/DICAM-main/results/output_euvp"

# 获取所有PNG图像文件
gen_files = [f for f in os.listdir(gen_dir) if f.endswith(".jpg")]
print(f"找到 {len(gen_files)} 张生成图像")

# 初始化结果列表
UIQM_results = []
UCIQE_results = []

for gen_file in gen_files:
    # 读取生成图像
    img_path = os.path.join(gen_dir, gen_file)
    
    # 直接使用OpenCV读取图像（BGR格式）
    img = cv2.imread(img_path)
    
    if img is None:
        print(f"警告: 无法读取图像 {img_path}")
        continue
    
    # 确保是RGB顺序（有些无参考指标偏好RGB）
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    try:
        # 计算UIQM和UCIQE指标
        UIQM_results.append(getUIQM(img_rgb))
        UCIQE_results.append(getUCIQE(img_rgb))
    except Exception as e:
        print(f"处理图像 {gen_file} 时出错: {str(e)}")

# 输出结果
if UIQM_results:
    print("\n无参考质量评估结果:")
    print(f"UIQM平均值: {np.mean(UIQM_results):.4f} ± {np.std(UIQM_results):.4f}")
    print(f"UCIQE平均值: {np.mean(UCIQE_results):.4f} ± {np.std(UCIQE_results):.4f}")
    
    print("\n图像质量参考:")
    print("UIQM > 3.0 : 优秀水下图像")
    print("UCIQE > 40 : 良好水下图像")
else:
    print("错误: 未能计算任何指标")