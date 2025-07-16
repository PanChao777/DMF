import torch.nn as nn
import torch.nn.functional as F
import torch
from Config.DWConv import SeparableConv2d
from Config.MSFN import FeedForward

class Flatten(nn.Module):
    def forward(self, inp):
        return inp.view(inp.size(0), -1)

class DAPA(nn.Module):
    def __init__(self, channel, reduction=8):
        super().__init__()
        
        self.channel_att_main = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channel, channel//reduction, 1),
            nn.ReLU(),
            nn.Conv2d(channel//reduction, channel, 1),
            nn.Sigmoid()
        )
        
        
        self.channel_att_cross = nn.Sequential(
            nn.AdaptiveMaxPool2d(1),
            nn.Conv2d(channel, channel//(reduction*2), 1),
            nn.ReLU(),
            nn.Conv2d(channel//(reduction*2), channel, 1),
        )
        
        self.spatial_att = nn.Sequential(
            nn.Conv2d(2, 1, 5, padding=2),
            nn.Sigmoid()
        )
        
    def forward(self, x):
     
        channel_main = self.channel_att_main(x)
        
      
        channel_cross = self.channel_att_cross(x)
        enhanced_channel = torch.sigmoid(channel_main + channel_cross)
        
      
        avg_pool = x.mean(1, True)
        max_pool, _ = torch.max(x, dim=1, keepdim=True)
        spatial = self.spatial_att(torch.cat([avg_pool, max_pool], dim=1))
        
        # 动态加权融合
        combined_att = enhanced_channel * spatial
        return x * combined_att
class DynamicFusionGate(nn.Module):
    def __init__(self, channels_in, channels_out):
        super().__init__()
        self.gate_net = nn.Sequential(
            nn.Conv2d(channels_in * 3, channels_in // 2, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(channels_in // 2, 3, 1),
            nn.Softmax(dim=1)
        )
        self.output_conv = nn.Conv2d(channels_in, channels_out, 1)

    def forward(self, x_r, x_g, x_b):
        combined = torch.cat([x_r, x_g, x_b], dim=1)
        gate_weights = self.gate_net(combined)  # 生成3通道权重
        
        fused = (gate_weights[:, 0:1] * x_r + 
                 gate_weights[:, 1:2] * x_g + 
                 gate_weights[:, 2:3] * x_b)
        
        return self.output_conv(fused)
class LDConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.weight_generator = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_ch, in_ch//4, 1),
            nn.ReLU(),
            nn.Conv2d(in_ch//4, 9*out_ch, 1)
        )
        self.conv = nn.Conv2d(in_ch, out_ch, 3, padding=1)

    def forward(self, x):
        weights = self.weight_generator(x).view(x.size(0), 9, -1)
        dynamic_weight = torch.sigmoid(weights).mean(dim=1)
        return self.conv(x) * dynamic_weight.view(x.size(0), -1, 1, 1)     
class MSFH(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, dim)
        self.conv1 = nn.Conv2d(dim, dim//4, 1)
        
        # 三分支设计（新增Identity分支）
        self.dw_conv3 = nn.Conv2d(dim//4, dim//4, 3, padding=1, groups=dim//4)
        self.dw_conv5 = nn.Conv2d(dim//4, dim//4, 5, padding=2, groups=dim//4)
        self.identity_branch = nn.Identity()  # 新增分支
        self.fft_branch = FFTEnhance(dim//4)
        # 自适应通道调整
        self.out_conv = nn.Conv2d(dim//4 * 4, dim, 1)  # 输入通道调整为3倍

    def forward(self, x):
        identity = x
        x = self.norm1(x)
        x = self.conv1(x)  # [B, C/4, H, W]
        
        # 三分支并行
        x1 = self.dw_conv3(x)
        x2 = self.dw_conv5(x)
        x3 = self.identity_branch(x)  # 新增分支输出
        x4 = self.fft_branch(x)

        fused = torch.cat([x1, x2, x3, x4], dim=1)  # 通道维度拼接
        return identity + self.out_conv(fused)
class FFTEnhance(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, 1),
            nn.GroupNorm(8, channels)
        )

    def forward(self, x):
        fft = torch.fft.fft2(x)
        amp = torch.abs(fft)
        phase = torch.angle(fft)
        # 振幅特征增强
        amp_feat = self.conv(amp)
        # 残差连接保持原有信息
        return x + amp_feat
class DMF(nn.Module):
    def __init__(self):
        super(DMF, self).__init__()
        # 使用1x1卷积升维至256，再应用MixStructureBlock
        self.layer_1_r = nn.Sequential(
            nn.Conv2d(1, 256, kernel_size=1),
            MSFH(256)
        )
        self.layer_1_g = nn.Sequential(
            nn.Conv2d(1, 256, kernel_size=1),
            MSFH(256)
        )
        self.layer_1_b = nn.Sequential(
            nn.Conv2d(1, 256, kernel_size=1),
            MSFH(256)
        )


        self.layer_2_r = DAPA(256,4)
        self.layer_2_g = DAPA(256,4)
        self.layer_2_b = DAPA(256,4)

        self.layer_3 = LDConv(768,256)
        self.layer_4 = DAPA(256,4)

        self.layer_tail = nn.Sequential(
            nn.Conv2d(256,24,kernel_size=3,padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(24,3,kernel_size=1),
            nn.Sigmoid()
        )
        self.fusion_gate = DynamicFusionGate(256, 768)
     
    def forward(self, input):
        input_r = input[:,0:1,:,:]  # 保持维度 [B,1,H,W]
        input_g = input[:,1:2,:,:]
        input_b = input[:,2:3,:,:]
        
        layer_1_r = self.layer_1_r(input_r)
        layer_1_g = self.layer_1_g(input_g)
        layer_1_b = self.layer_1_b(input_b)

        layer_2_r = self.layer_2_r(layer_1_r)
        layer_2_g = self.layer_2_g(layer_1_g)
        layer_2_b = self.layer_2_b(layer_1_b)

        fused_layer_2 = self.fusion_gate(layer_2_r, layer_2_g, layer_2_b)
        layer_3 = self.layer_3(fused_layer_2)
        layer_4 = self.layer_4(layer_3)
        # layer_4 = self.fft_final(layer_4)
        return self.layer_tail(layer_4)  

