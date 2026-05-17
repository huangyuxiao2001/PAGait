import torch
import cv2
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from ..base_model import BaseModel
from ..modules import (HorizontalPoolingPyramid, PackSequenceWrapper, 
                       SeparateFCs, SeparateBNNecks, SetBlockWrapper,
                       conv3x3, conv1x1, BasicBlock2D, BasicBlockP3D)
from einops import rearrange

class CCE(nn.Module): 
    def __init__(self, in_channels=64, squeeze_ratio=16, h=32, w=22):
        super(CCE, self).__init__()
        hidden_dim = int(in_channels / squeeze_ratio)
        self.TP_mean = PackSequenceWrapper(torch.mean)
        self.conv2 = SetBlockWrapper(nn.Sequential(
                conv1x1(in_channels, hidden_dim), 
                nn.BatchNorm2d(hidden_dim), 
                nn.ReLU(inplace=True), 
                conv1x1(hidden_dim, hidden_dim),
                nn.BatchNorm2d(hidden_dim), 
                nn.ReLU(inplace=True), 
                conv1x1(hidden_dim, in_channels), 
            ))
        self.conv1 = SetBlockWrapper(nn.Sequential(
                conv1x1(in_channels, hidden_dim), 
                nn.BatchNorm2d(hidden_dim), 
                nn.ReLU(inplace=True), 
                conv1x1(hidden_dim, hidden_dim), 
                nn.BatchNorm2d(hidden_dim), 
                nn.ReLU(inplace=True), 
                conv1x1(hidden_dim, in_channels), 
            ))
        self.kernel = torch.ones((3,3))

    def channel_normalization(self, masked_attn):
        min_vals = masked_attn.min(dim=1, keepdim=True).values
        max_vals = masked_attn.max(dim=1, keepdim=True).values
        min_vals = min_vals.expand_as(masked_attn)
        max_vals = max_vals.expand_as(masked_attn)
        attn_norm = (masked_attn - min_vals) / (max_vals - min_vals + 1e-6)
        attn_norm = attn_norm.clamp(0, 1)
        return attn_norm

    def forward(self, x1, x2): 
        '''
        x1: silhouette feature [n, c, s, h, w]
        x2: parsing feature    [n, c, s, h, w]
        '''
        attn_x2 = self.conv2(x2) 
        n, c, t, h, w = attn_x2.size()
        
        attn_x1 = self.conv1(x1)
        attn_x = torch.stack((attn_x1, attn_x2), dim=1)

        attn_x = F.softmax(attn_x, dim=1)
        attn_sil = attn_x[:, 0, ...]
        attn_parsing = attn_x[:, 1, ...]
        attn_ = torch.min(attn_sil, attn_parsing)

        attn = self.channel_normalization(attn_)
        common_feature = (x1 + x2) / 2 * attn
        
        return common_feature

def PartPooling(x, with_max_pool=True):
    n_s, c, h, w = x.size()
    z = x.reshape(n_s, c, -1)
    if with_max_pool:
        z = z.mean(-1) + z.max(-1)[0]
    else:
        z = z.mean(-1)
    return z

class RAM(nn.Module):
    def __init__(self, channels, reduction=16, with_max_pool=True, choosed_part=''):
        super(RAM, self).__init__()
        self.choosed_part = choosed_part
        self.with_max_pool = with_max_pool
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        # 可学习权重：目标区域（1.5~2.0）、非目标区域（0.5~1.0）
        # 初始值：目标1.75，非目标0.75（处于合理中间值）
        self.gammas = torch.nn.Parameter(torch.ones(1) * 0.0)  # sigmoid(0)=0.5 → 1.5+0.5*0.5=1.75
        self.non_target_weight = torch.nn.Parameter(torch.ones(1) * 0.0)  # sigmoid(0)=0.5 → 0.5+0.5*0.5=0.75
        
        self.fc = nn.Sequential(
            nn.Linear(channels * 2, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels * 2, bias=False),
            nn.Sigmoid()
        )

    def differential_modulation(self, z, mask_resize):
        """
        基于人体解析掩码的差异化特征调制
        对目标部位和非目标部位应用不同的可学习权重
        """
        split_parts = {
            'up': [1],
            'middle': [2, 3, 4, 5, 6, 11],
            'down': [7, 8, 9, 10],
        }
        choosed_part_list = split_parts[self.choosed_part]
        mask_long = mask_resize.long()
        
        # 生成二值掩码：目标区域 vs 非目标区域
        target_mask = torch.zeros_like(mask_long, dtype=torch.bool)
        for part_i in choosed_part_list:
            target_mask |= (mask_long == part_i)
        non_target_mask = ~target_mask
        
        # 扩展通道维度以匹配特征图
        target_mask = target_mask.unsqueeze(1).float()
        non_target_mask = non_target_mask.unsqueeze(1).float()
        
        # 严格约束权重范围（使用sigmoid保证单调性和边界）
        constrained_target_weight = 1.5 + 0.5 * torch.sigmoid(self.gammas)    # [1.5, 2.0]
        constrained_non_target_weight = 0.5 + 0.5 * torch.sigmoid(self.non_target_weight)  # [0.5, 1.0]
        
        # 执行差异化特征加权
        z_feat = (
            target_mask * z * constrained_target_weight +
            non_target_mask * z * constrained_non_target_weight
        )
        
        return z_feat
    
    def forward(self, x, y, mask_resize):
        n, c, s, h, w = x.size()
        
        x_ = x.transpose(1, 2).reshape(-1, c, h, w)
        y_ = y.transpose(1, 2).reshape(-1, c, h, w)

        # 轮廓图分支全局池化
        x_pool = self.avg_pool(x_).view(n*s, c) + self.max_pool(x_).view(n*s, c)

        # 解析图分支差异化调制+全局池化
        y_feat = self.differential_modulation(y_, mask_resize)
        y_pool = PartPooling(y_feat, with_max_pool=self.with_max_pool)
        
        # 注意力权重生成
        z = torch.cat([x_pool, y_pool], dim=1)
        z = self.fc(z).view(n, s, 2*c, 1, 1).transpose(1, 2)
        
        # 双分支特征加权融合
        return x * z[:,:c,:,:,:] + y * z[:,c:,:,:,:]

# -------------------------- 主模型 --------------------------
class PAGait(BaseModel):
    def build_network(self, model_cfg):
        in_C = model_cfg['Backbone']['in_channels']
        B = model_cfg['Backbone']['blocks']
        C = model_cfg['Backbone']['C']
        self.inference_use_emb = model_cfg.get('use_emb2', False)
        
        # 通道跟踪
        self.inplanes_sil     = 32 * C   # 轮廓图分支
        self.inplanes_parsing = 32 * C   # 人体解析图分支
        self.inplanes_fused   = 3 * 64 * C
        self.inplanes_common  = 64 * C
        
        self.ram_up = RAM(channels=64 * C, **model_cfg['RAMLayersP'], choosed_part='up')
        self.ram_middle = RAM(channels=64 * C, **model_cfg['RAMLayersP'], choosed_part='middle')
        self.ram_down = RAM(channels=64 * C, **model_cfg['RAMLayersP'], choosed_part='down')
        
        # -------------------------- 轮廓图分支（silhouette） --------------------------
        self.sil_layer0 = SetBlockWrapper(nn.Sequential(
            conv3x3(1, self.inplanes_sil, 1),
            nn.BatchNorm2d(self.inplanes_sil),
            nn.ReLU(inplace=True)
        ))
        self.sil_layer1 = SetBlockWrapper(self.make_layer(
            BasicBlock2D, 32 * C, stride=[1, 1], blocks_num=B[0],
            mode='2d', inplanes_track='sil'
        ))
        self.sil_layer2 = self.make_layer(
            BasicBlockP3D, 64 * C, stride=[2, 2], blocks_num=B[1],
            mode='p3d', inplanes_track='sil'
        )
        
        # -------------------------- 人体解析图分支（parsing） --------------------------
        self.parsing_layer0 = SetBlockWrapper(nn.Sequential(
            conv3x3(1, self.inplanes_parsing, 1),
            nn.BatchNorm2d(self.inplanes_parsing),
            nn.ReLU(inplace=True)
        ))
        self.parsing_layer1 = SetBlockWrapper(self.make_layer(
            BasicBlock2D, 32 * C, stride=[1, 1], blocks_num=B[0],
            mode='2d', inplanes_track='parsing'
        ))
        self.parsing_layer2 = self.make_layer(
            BasicBlockP3D, 64 * C, stride=[2, 2], blocks_num=B[1],
            mode='p3d', inplanes_track='parsing'
        )
        
        # 共同特征模块
        self.cce = CCE(
            in_channels=64 * C,
            squeeze_ratio=model_cfg.get('CS', {}).get('squeeze_ratio', 16),
            h=model_cfg.get('CS', {}).get('h', 32),
            w=model_cfg.get('CS', {}).get('w', 22)
        )
        
        # 共同特征分支
        self.common_layer3 = self.make_layer(
            BasicBlockP3D, 128 * C, stride=[2, 2], blocks_num=B[2],
            mode='p3d', inplanes_track='common'
        )
        self.common_layer4 = self.make_layer(
            BasicBlockP3D, 256 * C, stride=[1, 1], blocks_num=B[3],
            mode='p3d', inplanes_track='common'
        )
        
        # CA融合特征分支
        self.fused_layer3 = self.make_layer(
            BasicBlockP3D, 128 * C, stride=[2, 2], blocks_num=B[2],
            mode='p3d', inplanes_track='fused'
        )
        self.fused_layer4 = self.make_layer(
            BasicBlockP3D, 256 * C, stride=[1, 1], blocks_num=B[3],
            mode='p3d', inplanes_track='fused'
        )
        
        # 池化与全连接
        self.TP = PackSequenceWrapper(torch.max)
        self.HPP = HorizontalPoolingPyramid(bin_num=[16])
        
        self.FCs_fused = SeparateFCs(16, 256 * C, 128 * C)
        self.BNNecks_fused = SeparateBNNecks(
            16, 128 * C, class_num=model_cfg['SeparateBNNecks']['class_num']
        )
        
        self.FCs_common = SeparateFCs(16, 256 * C, 128 * C)
        self.BNNecks_common = SeparateBNNecks(
            16, 128 * C, class_num=model_cfg['SeparateBNNecks']['class_num']
        )

    def make_layer(self, block, planes, stride, blocks_num, mode='2d', inplanes_track='sil'):
        # 支持分支：sil / parsing / fused / common
        if inplanes_track == 'sil':
            current_inplanes = self.inplanes_sil
        elif inplanes_track == 'parsing':
            current_inplanes = self.inplanes_parsing
        elif inplanes_track == 'fused':
            current_inplanes = self.inplanes_fused
        elif inplanes_track == 'common':
            current_inplanes = self.inplanes_common
        else:
            raise ValueError(f"Unknown inplanes track: {inplanes_track}")
        
        if max(stride) > 1 or current_inplanes != planes * block.expansion:
            if mode == '3d':
                downsample = nn.Sequential(
                    nn.Conv3d(current_inplanes, planes * block.expansion,
                              kernel_size=[1,1,1], stride=stride, padding=[0,0,0], bias=False),
                    nn.BatchNorm3d(planes * block.expansion)
                )
            elif mode == '2d':
                downsample = nn.Sequential(
                    conv1x1(current_inplanes, planes * block.expansion, stride=stride),
                    nn.BatchNorm2d(planes * block.expansion)
                )
            elif mode == 'p3d':
                downsample = nn.Sequential(
                    nn.Conv3d(current_inplanes, planes * block.expansion,
                              kernel_size=[1,1,1], stride=[1, *stride], padding=[0,0,0], bias=False),
                    nn.BatchNorm3d(planes * block.expansion)
                )
            else:
                raise TypeError('Unknown mode: {}'.format(mode))
        else:
            downsample = lambda x: x
        
        layers = [block(current_inplanes, planes, stride=stride, downsample=downsample)]
        
        # 更新通道数
        if inplanes_track == 'sil':
            self.inplanes_sil = planes * block.expansion
        elif inplanes_track == 'parsing':
            self.inplanes_parsing = planes * block.expansion
        elif inplanes_track == 'fused':
            self.inplanes_fused = planes * block.expansion
        elif inplanes_track == 'common':
            self.inplanes_common = planes * block.expansion
        
        s = [1, 1] if mode in ['2d', 'p3d'] else [1, 1, 1]
        for _ in range(1, blocks_num):
            layers.append(block(
                self.inplanes_sil if inplanes_track == 'sil' else
                self.inplanes_parsing if inplanes_track == 'parsing' else
                self.inplanes_fused if inplanes_track == 'fused' else
                self.inplanes_common,
                planes, stride=s
            ))
        
        return nn.Sequential(*layers)

    def inputs_pretreament(self, inputs):
        """输入：silhouette轮廓图 + parsing人体解析图"""
        parsing_silhouette = inputs[0]
        new_data_list = []
        
        for parsing, sil in zip(parsing_silhouette[0], parsing_silhouette[1]):
            sil = sil[:, np.newaxis, ...]
            parsing = parsing[:, np.newaxis, ...]
            
            sil_h, sil_w = sil.shape[-2], sil.shape[-1]
            parsing_h, parsing_w = parsing.shape[-2], parsing.shape[-1]
            
            if parsing_h != parsing_w and sil_h == sil_w:
                cutting = (parsing_h - parsing_w) // 2
                sil = sil[..., cutting:-cutting]
            
            cat_data = np.concatenate([sil, parsing], axis=1)
            new_data_list.append(cat_data)
        
        new_inputs = [[new_data_list], inputs[1], inputs[2], inputs[3], inputs[4]]
        return super().inputs_pretreament(new_inputs)

    def forward(self, inputs):
        ipts, labs, _, _, seqL = inputs
        
        # 输入 = 轮廓图 + 人体解析图 拼接
        x = ipts[0].transpose(1, 2).contiguous()
        assert x.size(-1) in [44, 64, 88, 96]
        
        # 分离双输入
        parsing_input    = x[:, :1, ...]
        sil_input = x[:, 1:2, ...] 
        del ipts
        
        # ==================== 1. 提取双分支基础特征 ====================
        # 轮廓图分支
        sil0 = self.sil_layer0(sil_input)
        sil1 = self.sil_layer1(sil0)
        sil2 = self.sil_layer2(sil1)
        
        # 人体解析图分支
        parsing0 = self.parsing_layer0(parsing_input)
        parsing1 = self.parsing_layer1(parsing0)
        parsing2 = self.parsing_layer2(parsing1)
        
        # ==================== 2. 人体解析mask缩放 ====================
        n, c_mid, s, h_mid, w_mid = sil2.shape
        mask_resize = F.interpolate(parsing_input.squeeze(1), size=(h_mid, w_mid), mode='nearest')
        mask_resize = mask_resize.transpose(1, 2).reshape(n * s, h_mid, w_mid)
        
        # ==================== 3. 区域注意力融合（RAM） ====================
        up_fused    = self.ram_up(sil2, parsing2, mask_resize)
        mid_fused   = self.ram_middle(sil2, parsing2, mask_resize)
        down_fused  = self.ram_down(sil2, parsing2, mask_resize)
        fused_mid   = torch.cat([up_fused, mid_fused, down_fused], dim=1)
        
        fused3  = self.fused_layer3(fused_mid)
        fused4  = self.fused_layer4(fused3)
        
        # ==================== 4. 共同特征提取（CS） ====================
        common_feat = self.cce(sil2, parsing2)
        common3     = self.common_layer3(common_feat)
        common4     = self.common_layer4(common3)
        
        # ==================== 5. 特征编码 ====================
        # CA分支
        fused_tp    = self.TP(fused4, seqL, options={"dim": 2})[0]
        fused_hpp   = self.HPP(fused_tp)
        embed_fused = self.FCs_fused(fused_hpp)
        _, logits_fused = self.BNNecks_fused(embed_fused)
        
        # CS分支
        common_tp    = self.TP(common4, seqL, options={"dim": 2})[0]
        common_hpp   = self.HPP(common_tp)
        embed_common = self.FCs_common(common_hpp)
        _, logits_common = self.BNNecks_common(embed_common)
        
        # ==================== 6. 双分支特征拼接 ====================
        embed_cat  = torch.cat((embed_fused, embed_common), dim=-1)
        logits_cat = torch.cat((logits_fused, logits_common), dim=-1)
        embed = embed_cat
        
        # ==================== Output ====================
        n_total, _, s_total, h_total, w_total = sil_input.size()
        retval = {
            'training_feat': {
                'triplet': {'embeddings': embed_cat, 'labels': labs},
                'softmax': {'logits': logits_cat, 'labels': labs}
            },
            'visual_summary': {
                'image/sils': sil_input.reshape(n_total*s_total, 1, h_total, w_total)
            },
            'inference_feat': {
                'embeddings': embed
            }
        }
        return retval