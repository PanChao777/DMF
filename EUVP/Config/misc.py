import torch
import numpy as np
import os
import re
from Config.euvp_options import opt
from torch.autograd import Variable

# def getLatestCheckpointName():
#     if os.path.exists(opt.checkpoints_dir):
#         file_names = os.listdir(opt.checkpoints_dir)
#         names_ext = [os.path.splitext(x) for x in file_names]
#         checkpoint_names_G = []    
#         l = []
#         for i in range(len(names_ext)):
#             module = names_ext[i][1] == '.pt' and str(names_ext[i][0]).split('_')
#             if module[0] == 'DICAM':
#                 checkpoint_names_G.append(int(module[1]))

#         if len(checkpoint_names_G) == 0 :
#             return None
    
#         g_index = max(checkpoint_names_G)
#         ckp_g = None
    
#         for i in file_names:    
#             if int(str(i).split('_')[1].split('.')[0]) == g_index and str(i).split('_')[0] == 'DICAM':
#                 ckp_g = i
#                 break

#         return ckp_g
def getLatestCheckpointName():
    # 确保检查点目录存在
    if not os.path.exists(opt.checkpoints_dir):
        return None
    
    # 获取所有检查点文件
    all_files = os.listdir(opt.checkpoints_dir)
    
    # 只保留有效的检查点文件（以 .pt 结尾且包含 DICAM_ 的文件）
    checkpoint_files = [f for f in all_files if f.endswith('.pt') and 'DICAM_' in f]
    
    if not checkpoint_files:
        return None
    
    # 按epoch数字排序
    def extract_epoch_number(filename):
        match = re.search(r'DICAM_(\d+)\.pt', filename)
        if match:
            return int(match.group(1))
        return 0
    
    checkpoint_files.sort(key=extract_epoch_number, reverse=True)
    
    # 返回最新的检查点文件名
    return checkpoint_files[0]