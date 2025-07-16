
import os  
from PIL import Image  
  
def resize_images(input_dir, output_dir, size=(256,256), resample=Image.LANCZOS):  
    # 确保输出目录存在  
    if not os.path.exists(output_dir):  
        os.makedirs(output_dir)  
  
    # 遍历输入目录中的所有文件  
    for filename in os.listdir(input_dir):  
        # 构建完整的文件路径  
        file_path = os.path.join(input_dir, filename)  
  
        # 检查文件是否为图像（可以根据扩展名或尝试打开图像）  
        if file_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):  
            try:  
                # 打开图像  
                with Image.open(file_path) as img:  
                    # 调整图像大小  
                    resized_img = img.resize(size, resample=resample)  
  
                    # 构建输出文件路径  
                    output_path = os.path.join(output_dir, filename)  
  
                    # 保存调整后的图像  
                    resized_img.save(output_path)  
                    print(f"Processed {filename}")  
            except Exception as e:  
                print(f"Error processing {filename}: {e}")  
  
# 使用示例  
input_directory = 'E:/download/yanyi/project/DICAM-main/results/hq_r90/'  
output_directory = 'E:/download/yanyi/project/DICAM-main/results/gt_r90/'  
resize_images(input_directory, output_directory)