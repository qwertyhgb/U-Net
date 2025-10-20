#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
预测脚本 - 使用训练好的UNet模型对输入图像进行分割预测
"""

import argparse
import logging
import os
from typing import List, Tuple, Union, Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from utils.data_loading import BasicDataset
from unet import UNet
from utils.utils import plot_img_and_mask


def predict_img(
    net: UNet,
    full_img: Image.Image,
    device: torch.device,
    scale_factor: float = 1.0,
    out_threshold: float = 0.5
) -> np.ndarray:
    """
    使用UNet模型预测图像的分割掩码
    
    参数:
        net: 训练好的UNet模型
        full_img: 输入的PIL图像
        device: 计算设备(CPU/GPU)
        scale_factor: 图像缩放因子，用于减少内存占用
        out_threshold: 二值化阈值，仅在二分类时使用
        
    返回:
        预测的分割掩码(numpy数组)
    """
    # 设置为评估模式
    net.eval()
    
    # 预处理图像
    img = torch.from_numpy(BasicDataset.preprocess(None, full_img, scale_factor, is_mask=False))
    img = img.unsqueeze(0)  # 添加批次维度 [B,C,H,W]
    img = img.to(device=device, dtype=torch.float32)

    # 使用torch.no_grad()减少内存使用并加速推理
    with torch.no_grad():
        # 前向传播
        output = net(img).cpu()
        
        # 将输出调整为原始图像大小
        output = F.interpolate(output, (full_img.size[1], full_img.size[0]), mode='bilinear', align_corners=False)
        
        # 根据分类数量处理输出
        if net.n_classes > 1:
            # 多分类情况：取最大值所在的类别
            mask = output.argmax(dim=1)
        else:
            # 二分类情况：使用sigmoid函数和阈值
            mask = torch.sigmoid(output) > out_threshold

    # 转换为numpy数组并返回
    return mask[0].long().squeeze().numpy()


def get_args() -> argparse.Namespace:
    """
    解析命令行参数
    
    返回:
        解析后的参数对象
    """
    parser = argparse.ArgumentParser(description='使用UNet模型预测图像分割掩码')
    
    # 模型参数
    parser.add_argument('--model', '-m', default='MODEL.pth', metavar='FILE',
                        help='指定存储模型的文件路径')
    parser.add_argument('--bilinear', action='store_true', default=False, 
                        help='使用双线性上采样而非转置卷积')
    parser.add_argument('--classes', '-c', type=int, default=2, 
                        help='分类数量')
    
    # 输入输出参数
    parser.add_argument('--input', '-i', metavar='INPUT', nargs='+', 
                        help='输入图像的文件名', required=True)
    parser.add_argument('--output', '-o', metavar='OUTPUT', nargs='+', 
                        help='输出图像的文件名')
    parser.add_argument('--no-save', '-n', action='store_true', 
                        help='不保存输出掩码')
    
    # 预处理和后处理参数
    parser.add_argument('--mask-threshold', '-t', type=float, default=0.5,
                        help='二值化掩码的概率阈值')
    parser.add_argument('--scale', '-s', type=float, default=0.5,
                        help='输入图像的缩放因子')
    
    # 可视化参数
    parser.add_argument('--viz', '-v', action='store_true',
                        help='可视化处理过程中的图像')
    
    return parser.parse_args()


def get_output_filenames(args: argparse.Namespace) -> List[str]:
    """
    生成输出文件名列表
    
    参数:
        args: 命令行参数对象
        
    返回:
        输出文件名列表
    """
    def _generate_name(fn: str) -> str:
        """生成默认输出文件名"""
        return f'{os.path.splitext(fn)[0]}_OUT.png'

    # 如果提供了输出文件名，则使用提供的名称；否则生成默认名称
    return args.output or list(map(_generate_name, args.input))


def mask_to_image(mask: np.ndarray, mask_values: List) -> Image.Image:
    """
    将预测的掩码转换为可保存的图像
    
    参数:
        mask: 预测的掩码数组
        mask_values: 掩码值列表，用于映射类别索引到像素值
        
    返回:
        可保存的PIL图像
    """
    # 根据掩码值类型创建输出数组
    if isinstance(mask_values[0], list):
        # 多通道输出（如RGB）
        out = np.zeros((mask.shape[-2], mask.shape[-1], len(mask_values[0])), dtype=np.uint8)
    elif mask_values == [0, 1]:
        # 二值掩码
        out = np.zeros((mask.shape[-2], mask.shape[-1]), dtype=bool)
    else:
        # 灰度掩码
        out = np.zeros((mask.shape[-2], mask.shape[-1]), dtype=np.uint8)

    # 处理多通道掩码
    if mask.ndim == 3:
        mask = np.argmax(mask, axis=0)

    # 将类别索引映射到对应的像素值
    for i, v in enumerate(mask_values):
        out[mask == i] = v

    # 转换为PIL图像
    return Image.fromarray(out)


def main():
    """主函数：加载模型并处理所有输入图像"""
    # 解析命令行参数
    args = get_args()
    
    # 配置日志
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    logger = logging.getLogger(__name__)

    # 获取输入和输出文件列表
    in_files = args.input
    out_files = get_output_filenames(args)

    # 初始化模型
    logger.info(f'初始化UNet模型 (类别数: {args.classes}, 双线性上采样: {args.bilinear})')
    net = UNet(n_channels=3, n_classes=args.classes, bilinear=args.bilinear)

    # 选择设备(GPU/CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'加载模型: {args.model}')
    logger.info(f'使用设备: {device}')

    # 加载模型权重
    net.to(device=device)
    try:
        state_dict = torch.load(args.model, map_location=device)
        # 从状态字典中提取掩码值，如果不存在则使用默认值[0, 1]
        mask_values = state_dict.pop('mask_values', [0, 1])
        net.load_state_dict(state_dict)
        logger.info('模型加载成功!')
    except Exception as e:
        logger.error(f'加载模型失败: {e}')
        return

    # 处理每个输入图像
    for i, filename in enumerate(in_files):
        logger.info(f'正在预测图像 {filename} ...')
        
        try:
            # 打开图像
            img = Image.open(filename)
            
            # 预测分割掩码
            mask = predict_img(
                net=net,
                full_img=img,
                scale_factor=args.scale,
                out_threshold=args.mask_threshold,
                device=device
            )

            # 保存结果
            if not args.no_save:
                out_filename = out_files[i]
                result = mask_to_image(mask, mask_values)
                result.save(out_filename)
                logger.info(f'掩码已保存至 {out_filename}')

            # 可视化结果
            if args.viz:
                logger.info(f'正在可视化图像 {filename} 的结果，关闭窗口继续...')
                plot_img_and_mask(img, mask)
                
        except Exception as e:
            logger.error(f'处理图像 {filename} 时出错: {e}')


if __name__ == '__main__':
    main()
