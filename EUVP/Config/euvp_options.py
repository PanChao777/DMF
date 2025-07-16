import argparse
import torch
import os

parser = argparse.ArgumentParser()

parser.add_argument('--data_path', default='')

parser.add_argument('--checkpoints_dir', default='')
parser.add_argument('--batch_size', type=int, default=2)
parser.add_argument('--num_images', type=int, default=0)

parser.add_argument('--learning_rate_g', type=float, default=8e-04)

parser.add_argument('--end_epoch', type=int, default=200)
parser.add_argument(
    '--img_extensions', 
    nargs='+',                   
    default=['.png', '.bmp'],     
    help='支持的图像文件扩展名（如 .png .bmp）'
)
parser.add_argument('--image_size', type=int ,default=256)

parser.add_argument('--beta1', type=float ,default=0.9)
parser.add_argument('--beta2', type=float ,default=0.999)
parser.add_argument('--wd_g', type=float ,default=0.00005)
parser.add_argument('--wd_d', type=float ,default=0.00000)

parser.add_argument('--batch_mae_loss', type=float, default=0.0)
parser.add_argument('--total_mae_loss', type=float, default=0.0)

parser.add_argument('--batch_ssim_loss', type=float, default=0.0)
parser.add_argument('--total_ssim_loss', type=float, default=0.0)

parser.add_argument('--batch_vgg_loss', type=float, default=0.0)
parser.add_argument('--total_vgg_loss', type=float, default=0.0)

parser.add_argument('--batch_loss', type=float, default=0.0)
parser.add_argument('--total_loss', type=float, default=0.0)

parser.add_argument('--lambda_mae', type=float, default=1.0)
parser.add_argument('--lambda_vgg', type=float, default=1.0)
parser.add_argument('--lambda_ssim', type=float, default=1.0)

parser.add_argument('--testing_start', type=int, default=1)
parser.add_argument('--testing_end', type=int, default=1)
parser.add_argument('--testing_mode', default="Nat")
parser.add_argument('--testing_dir_inp', default="")
parser.add_argument('--testing_dir_gt', default="")

opt = parser.parse_args()
# print(opt)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# print(device)

if not os.path.exists(opt.checkpoints_dir):
    os.makedirs(opt.checkpoints_dir)